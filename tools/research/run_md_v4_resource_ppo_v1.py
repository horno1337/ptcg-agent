"""Run the prospectively locked MD-v4 resource-aware PPO v1 experiment.

The runner is research-only. It warm-starts the exact failed epoch-4 MD-v4
state, uses an explicit-FP32 Torch actor for gradient updates, anchors every
rollout row to the exact deployed NumPy MD-v3 policy, and exports only the
fixed terminal update-24 actor. The separate critic is never exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card as CARD  # noqa: E402
from agent import model, qu_v2_features as DEPLOYED_FEATURES  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research import (  # noqa: E402
    eval_md_v2_card_v1_gameplay as LAYERED,
)
from tools.research import (  # noqa: E402
    eval_md_v4_deployed_parent_correction as CORRECTION,
)
from tools.research import (  # noqa: E402
    eval_md_v4_numpy_deployable as NUMPY_EVAL,
)
from tools.research import lock_md_v4_resource_ppo_v1 as LOCK  # noqa: E402
from tools.research import md_v3_ppo_v2_population as OLD_POP  # noqa: E402
from tools.research import md_v4_explicit_reference as EXPLICIT  # noqa: E402
from tools.research import md_v4_features as MF  # noqa: E402
from tools.research import md_v4_model as MM  # noqa: E402
from tools.research import md_v4_runtime as RUNTIME  # noqa: E402
from tools.research import train_md_v3_ppo as PPO_V1  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as PPO_V2  # noqa: E402
from tools.research import train_md_v4 as MDV4_TRAIN  # noqa: E402
from tools.research import train_qu_v2a as BC  # noqa: E402
from tools.rl_env import (  # noqa: E402
    EpisodeSpec,
    OpponentSpec,
    PTCGRLEnv,
    SelectionSpec,
)


RESULT_SCHEMA = "ptcg.md-v4.resource-ppo-training-result.v1"
RECOVERY_SCHEMA = "ptcg.md-v4.resource-ppo-recovery.v1"
RECOVERY_MANIFEST_SCHEMA = "ptcg.md-v4.resource-ppo-recovery-manifest.v1"
TERMINAL_CHECKPOINT_SCHEMA = "ptcg.md-v4.resource-ppo-checkpoint.v1"
ATTEMPT_SCHEMA = "ptcg.md-v4.resource-ppo-attempt.v1"
RESUME_CONSUMPTION_SCHEMA = "ptcg.md-v4.resource-ppo-resume-consumption.v1"
RETIREMENT_SCHEMA = "ptcg.md-v4.resource-ppo-retirement.v1"
COMPLETION_SCHEMA = "ptcg.md-v4.resource-ppo-completion.v1"
DEFAULT_OUTPUT = LOCK.RUN_ROOT / "training"
EXPECTED_ACTOR_PARAMETERS = 31_148
CRITIC_INPUT = 192
ATTEMPT_FILE = "ATTEMPT-CONSUMED.json"
RETIREMENT_FILE = "RETIRED.json"
COMPLETION_FILE = "COMPLETED.json"
RESUME_CONSUMPTION_FILE = "RESUME-CONSUMED.json"
WORKER_LOCK_SUFFIX = ".WORKER.lock"


class ResourcePPORunnerError(RuntimeError):
    """The locked resource-aware PPO run violated its contract."""


def _worker_lock_path(output: Path) -> Path:
    output = output.expanduser().resolve()
    return output.parent / f".{output.name}{WORKER_LOCK_SUFFIX}"


@contextmanager
def _exclusive_worker(output: Path):
    """Hold one kernel-backed worker lease for the complete run lifecycle."""
    output = _assert_output(output, must_not_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _worker_lock_path(output)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourcePPORunnerError(
                f"another official worker holds {lock_path}"
            ) from error
        metadata = json.dumps(
            {
                "candidate": LOCK.CANDIDATE,
                "output": _display_path(output),
                "pid": os.getpid(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, metadata)
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _hash_training_object(value: Any) -> str:
    """Deterministically hash nested optimizer/checkpoint state."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(b"\0")
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            ordered = sorted(
                item.items(),
                key=lambda pair: (
                    type(pair[0]).__name__,
                    repr(pair[0]),
                ),
            )
            digest.update(str(len(ordered)).encode())
            digest.update(b"\0")
            for key, nested in ordered:
                visit(key)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            digest.update(
                b"list\0" if isinstance(item, list) else b"tuple\0"
            )
            digest.update(str(len(item)).encode())
            digest.update(b"\0")
            for nested in item:
                visit(nested)
        elif item is None:
            digest.update(b"none\0")
        elif isinstance(item, bool):
            digest.update(b"bool\0" + (b"1" if item else b"0"))
        elif isinstance(item, int):
            digest.update(b"int\0" + str(item).encode() + b"\0")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ResourcePPORunnerError(
                    "cannot hash non-finite training state"
                )
            digest.update(
                b"float\0" + item.hex().encode() + b"\0"
            )
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"str\0" + str(len(encoded)).encode() + b"\0")
            digest.update(encoded)
        else:
            raise ResourcePPORunnerError(
                f"unsupported training-state value {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def _self_hashed_payload(
    payload: Mapping[str, Any],
    *,
    hash_key: str,
) -> dict[str, Any]:
    result = dict(payload)
    result[hash_key] = LOCK.canonical_sha256(result)
    return result


def _load_self_hashed_json(
    path: Path,
    *,
    schema: str,
    hash_key: str,
) -> dict[str, Any]:
    payload = LOCK._strict_json(path)
    recorded = payload.get(hash_key)
    body = {key: value for key, value in payload.items() if key != hash_key}
    if (
        payload.get("schema") != schema
        or recorded != LOCK.canonical_sha256(body)
    ):
        raise ResourcePPORunnerError(f"{path.name} identity is invalid")
    return payload


def _write_self_hashed_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_key: str,
) -> dict[str, Any]:
    result = _self_hashed_payload(payload, hash_key=hash_key)
    PPO_V1.atomic_json(path, result)
    return result


@dataclass
class ResourceDecision:
    """One learner ST_MAIN macro-decision with immutable rollout anchors."""

    features: MF.PublicResourceWindowFeatures
    critic_features: np.ndarray
    deployed_parent_logits: np.ndarray
    picks: tuple[int, ...]
    n_options: int
    min_count: int
    max_count: int
    old_logp: float
    old_value: float
    episode_id: int
    decision_index: int
    learner_select_index: int
    transition_steps: int = 0
    reward: float = 0.0
    terminal: bool = False

    @property
    def spec(self) -> SelectionSpec:
        return SelectionSpec(
            self.n_options,
            self.min_count,
            self.max_count,
        )


class ResourceCritic(nn.Module):
    """Training-only value estimator over a detached rollout snapshot."""

    def __init__(self, input_size: int = CRITIC_INPUT, hidden: int = 64):
        super().__init__()
        if input_size != CRITIC_INPUT or hidden != LOCK.CRITIC_HIDDEN:
            raise ValueError("resource critic architecture is fixed")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden)
        self.hidden = nn.Linear(input_size, hidden, bias=True)
        self.output = nn.Linear(hidden, 1, bias=True)
        nn.init.orthogonal_(self.hidden.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.hidden.bias)
        nn.init.orthogonal_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim != 2
            or values.shape[1] != self.input_size
            or values.dtype != torch.float32
        ):
            raise ResourcePPORunnerError(
                "critic input must be float32 [batch,192]"
            )
        result = torch.tanh(self.output(F.relu(self.hidden(values)))).squeeze(-1)
        if not bool(torch.isfinite(result).all()):
            raise ResourcePPORunnerError("critic produced non-finite values")
        return result


@dataclass(frozen=True)
class ParameterScopes:
    actor: tuple[torch.nn.Parameter, ...]
    critic: tuple[torch.nn.Parameter, ...]
    frozen: tuple[torch.nn.Parameter, ...]
    actor_names: tuple[str, ...]
    critic_names: tuple[str, ...]
    frozen_names: tuple[str, ...]


@dataclass
class FrozenPopulation:
    opponents: list[OpponentSpec]
    controllers: dict[str, Any]
    manifest: dict[str, Any]


def configure_parameter_scopes(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
) -> ParameterScopes:
    """Train all and only MD-v4 actor parameters plus the external critic."""
    if not isinstance(net, EXPLICIT.TorchMDV4ExplicitFP32):
        raise TypeError("resource PPO requires the explicit-FP32 MD-v4 actor")
    if not isinstance(critic, ResourceCritic):
        raise TypeError("resource PPO requires its training-only critic")
    actor: list[torch.nn.Parameter] = []
    frozen: list[torch.nn.Parameter] = []
    actor_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in net.named_parameters():
        if name.startswith("parent."):
            parameter.requires_grad_(False)
            frozen.append(parameter)
            frozen_names.append(name)
            continue
        module = name.split(".", 1)[0]
        if module not in LOCK.ACTOR_MODULES:
            raise ResourcePPORunnerError(
                f"unclassified MD-v4 actor parameter {name!r}"
            )
        parameter.requires_grad_(True)
        actor.append(parameter)
        actor_names.append(name)
    critic_parameters = tuple(critic.parameters())
    critic_names = tuple(f"critic.{name}" for name, _ in critic.named_parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    if (
        not actor
        or not frozen
        or not critic_parameters
        or sum(parameter.numel() for parameter in actor)
        != EXPECTED_ACTOR_PARAMETERS
        or set(map(id, actor)) & set(map(id, critic_parameters))
        or set(map(id, actor)) & set(map(id, frozen))
        or set(map(id, critic_parameters)) & set(map(id, frozen))
    ):
        raise ResourcePPORunnerError("actor/critic/frozen scopes are invalid")
    if tuple(sorted(set(name.split(".", 1)[0] for name in actor_names))) != (
        tuple(sorted(LOCK.ACTOR_MODULES))
    ):
        raise ResourcePPORunnerError("one or more MD-v4 actor modules are absent")
    return ParameterScopes(
        actor=tuple(actor),
        critic=critic_parameters,
        frozen=tuple(frozen),
        actor_names=tuple(actor_names),
        critic_names=critic_names,
        frozen_names=tuple(frozen_names),
    )


def make_optimizer(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> tuple[torch.optim.Adam, ParameterScopes]:
    if (
        not math.isfinite(actor_learning_rate)
        or actor_learning_rate <= 0.0
        or not math.isfinite(critic_learning_rate)
        or critic_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be finite and positive")
    scopes = configure_parameter_scopes(net, critic)
    optimizer = torch.optim.Adam([
        {
            "params": list(scopes.actor),
            "lr": float(actor_learning_rate),
            "name": "actor",
        },
        {
            "params": list(scopes.critic),
            "lr": float(critic_learning_rate),
            "name": "critic",
        },
    ])
    return optimizer, scopes


def _validate_optimizer(
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    *,
    actor_learning_rate: float | None = None,
    critic_learning_rate: float | None = None,
) -> None:
    groups = {str(group.get("name")): group for group in optimizer.param_groups}
    if set(groups) != {"actor", "critic"}:
        raise ResourcePPORunnerError(
            "optimizer must have exact actor and critic groups"
        )
    for name, parameters in (
        ("actor", scopes.actor),
        ("critic", scopes.critic),
    ):
        expected_lr = (
            actor_learning_rate
            if name == "actor" else critic_learning_rate
        )
        if (
            {id(parameter) for parameter in groups[name]["params"]}
            != {id(parameter) for parameter in parameters}
            or (
                expected_lr is not None
                and float(groups[name].get("lr", math.nan))
                != float(expected_lr)
            )
            or tuple(groups[name].get("betas", ())) != (0.9, 0.999)
            or float(groups[name].get("eps", math.nan)) != 1e-8
            or float(groups[name].get("weight_decay", math.nan)) != 0.0
            or bool(groups[name].get("amsgrad", False))
        ):
            raise ResourcePPORunnerError(
                f"optimizer {name} parameter group or hyperparameters drifted"
            )


def _training_config(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    config = lock.get("training")
    expected = LOCK._training_contract()
    if not isinstance(config, Mapping) or dict(config) != expected:
        raise ResourcePPORunnerError("locked training configuration drifted")
    return config


def _assert_output(path: Path, *, must_not_exist: bool = True) -> Path:
    output = path.expanduser().resolve()
    for protected in (ROOT / "agent", ROOT / "decks"):
        try:
            output.relative_to(protected.resolve())
        except ValueError:
            continue
        raise ResourcePPORunnerError(
            f"research output cannot be below {protected}"
        )
    if output == ROOT.resolve():
        raise ResourcePPORunnerError("research output cannot be repository root")
    if must_not_exist and output.exists():
        raise ResourcePPORunnerError(f"refusing to overwrite {output}")
    return output


def _create_attempt(
    output: Path,
    *,
    lock_sha256: str,
) -> dict[str, Any]:
    """Consume the sole fresh-run attempt before any engine outcome."""
    output = _assert_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return _write_self_hashed_json(
        output / ATTEMPT_FILE,
        {
            "schema": ATTEMPT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "written_before_first_engine_outcome": True,
            "candidate": LOCK.CANDIDATE,
            "lock_sha256": str(lock_sha256),
            "output": _display_path(output),
            "fresh_attempt_consumed": True,
            "native_engine_rng_seedable": False,
        },
        hash_key="attempt_sha256",
    )


def _load_attempt(
    output: Path,
    *,
    lock_sha256: str,
) -> dict[str, Any]:
    attempt = _load_self_hashed_json(
        output / ATTEMPT_FILE,
        schema=ATTEMPT_SCHEMA,
        hash_key="attempt_sha256",
    )
    if (
        attempt.get("candidate") != LOCK.CANDIDATE
        or attempt.get("lock_sha256") != lock_sha256
        or attempt.get("output") != _display_path(output)
        or attempt.get("fresh_attempt_consumed") is not True
        or attempt.get("written_before_first_engine_outcome") is not True
        or attempt.get("native_engine_rng_seedable") is not False
    ):
        raise ResourcePPORunnerError("official attempt marker drifted")
    return attempt


def _assert_attempt_open(output: Path) -> None:
    """Reject work after any terminal lifecycle marker becomes visible."""
    terminal = tuple(
        name
        for name in (RETIREMENT_FILE, COMPLETION_FILE, "result.json")
        if (output / name).exists()
    )
    if terminal:
        raise ResourcePPORunnerError(
            "official attempt is no longer open: " + ", ".join(terminal)
        )


def _write_retirement(
    output: Path,
    *,
    lock_sha256: str,
    attempt_sha256: str,
    error: BaseException,
    resume_consumption_sha256: str | None,
) -> None:
    """Immutably retire a consumed attempt after a controlled failure."""
    destination = output / RETIREMENT_FILE
    if destination.exists():
        return
    _write_self_hashed_json(
        destination,
        {
            "schema": RETIREMENT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": LOCK.CANDIDATE,
            "lock_sha256": str(lock_sha256),
            "attempt_sha256": str(attempt_sha256),
            "resume_consumption_sha256": resume_consumption_sha256,
            "retired": True,
            "reason_type": type(error).__name__,
            "reason": str(error)[:2000],
            "new_fresh_run_authorized": False,
            "promotion_authority": False,
        },
        hash_key="retirement_sha256",
    )


def _write_completion(
    output: Path,
    *,
    lock_sha256: str,
    attempt_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = output / "result.json"
    if not result_path.is_file():
        raise ResourcePPORunnerError("completion has no terminal result")
    recorded_result_sha256 = result.get("result_sha256")
    result_body = {
        key: value
        for key, value in result.items()
        if key != "result_sha256"
    }
    if (
        recorded_result_sha256 != LOCK.canonical_sha256(result_body)
        or LOCK._strict_json(result_path) != dict(result)
    ):
        raise ResourcePPORunnerError("terminal result identity is invalid")
    return _write_self_hashed_json(
        output / COMPLETION_FILE,
        {
            "schema": COMPLETION_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": LOCK.CANDIDATE,
            "lock_sha256": str(lock_sha256),
            "attempt_sha256": str(attempt_sha256),
            "result_sha256": recorded_result_sha256,
            "result_file_sha256": PPO_V1.sha256_file(result_path),
            "completed": True,
            "fixed_terminal_update": LOCK.UPDATES,
            "promotion_authority": False,
        },
        hash_key="completion_sha256",
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ResourcePPORunnerError(f"{path} is not a Torch mapping")
    return payload


def load_warm_start(
    paths: Mapping[str, Path],
    *,
    device: torch.device,
) -> tuple[
    EXPLICIT.TorchMDV4ExplicitFP32,
    Mapping[str, Any],
    dict[str, np.ndarray],
]:
    """Strictly reconstruct the exact epoch-4 state in the explicit actor."""
    if LOCK.file_sha256(paths["warm_checkpoint"]) != LOCK.WARM_CHECKPOINT_SHA256:
        raise ResourcePPORunnerError("warm checkpoint bytes drifted")
    # This reviewed loader validates checkpoint/state/parent/mapping identity.
    standard, checkpoint, initial_arrays = NUMPY_EVAL._load_candidate()
    explicit = EXPLICIT.TorchMDV4ExplicitFP32(standard.parent).eval()
    explicit.load_state_dict(standard.state_dict(), strict=True)
    if (
        MDV4_TRAIN._parameter_state_sha256(explicit.state_dict())
        != LOCK.WARM_STATE_SHA256
        or MM.frozen_parent_state_sha256(explicit)
        != LOCK.FROZEN_PARENT_STATE_SHA256
        or MM._mapping_sha256(MM.export_numpy_weights(explicit))
        != LOCK.WARM_NUMPY_MAPPING_SHA256
    ):
        raise ResourcePPORunnerError("explicit warm-start identity drifted")
    return explicit.to(device), checkpoint, initial_arrays


def _deployed_parent(
    paths: Mapping[str, Path],
) -> model.QuV2Net:
    if (
        LOCK.file_sha256(paths["deployed_parent_weights"])
        != LOCK.DEPLOYED_PARENT_WEIGHTS_SHA256
    ):
        raise ResourcePPORunnerError("deployed parent artifact drifted")
    parent = CORRECTION._load_deployed_parent()
    if not isinstance(parent, model.QuV2Net):
        raise ResourcePPORunnerError("deployed parent is not QuV2Net")
    return parent


def _critic_representation(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the detached 160+32 representation at rollout time."""
    with torch.no_grad():
        state = net.parent.state_vector(batch)
        resource = net._resource_summary(batch)
        history = net._log_summary(batch)
        fused = F.relu(net.fusion(torch.cat([
            resource,
            history,
            batch["resource_prompt_features"],
            batch["log_prompt_features"],
        ], dim=-1)))
        result = torch.cat([state, fused], dim=-1).detach()
    if (
        result.dtype != torch.float32
        or result.ndim != 2
        or result.shape[1] != CRITIC_INPUT
        or not bool(torch.isfinite(result).all())
    ):
        raise ResourcePPORunnerError("rollout critic representation is invalid")
    return result


def _same_mapping(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> tuple[bool, list[str]]:
    mismatches = []
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            mismatches.append(name)
            continue
        first = np.asarray(left[name])
        second = np.asarray(right[name])
        if (
            first.dtype != second.dtype
            or first.shape != second.shape
            or not first.flags.c_contiguous
            or not second.flags.c_contiguous
            or np.ascontiguousarray(first).tobytes()
            != np.ascontiguousarray(second).tobytes()
        ):
            mismatches.append(name)
    return not mismatches, mismatches


def _load_qu_net(path: Path) -> model.QuV2Net:
    with np.load(path, allow_pickle=False) as archive:
        net = model.QuV2Net(archive)
    if not getattr(net, "is_qu_v2", False):
        raise ResourcePPORunnerError(f"{path} is not a Qu-v2 network")
    return net


def _layered_move(controller: Any):
    def move(observation: dict, rng: Any) -> list[int]:
        del rng
        if hasattr(controller, "opponent_move"):
            return controller.opponent_move(observation, None)
        return controller.act(observation)
    return move


def build_population(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    initial_candidate: MM.NumpyMDV4,
) -> FrozenPopulation:
    """Construct controllers in the exact order bound by the lock."""
    static = lock.get("population", {}).get("manifest")
    rows = static.get("opponents") if isinstance(static, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ResourcePPORunnerError("lock has no population opponents")
    md_v3 = _load_qu_net(paths["deployed_parent_weights"])
    card = _load_qu_net(paths["card_weights"])
    qu = _load_qu_net(paths["qu_v2b_weights"])
    md_v1 = _load_qu_net(paths["md_v1_weights"])
    ppo_v2 = _load_qu_net(paths["ppo_v2_weights"])
    controllers: dict[str, Any] = {}
    opponents: list[OpponentSpec] = []
    for index, row in enumerate(rows):
        group = str(row["schedule_group"])
        deck = tuple(int(card_id) for card_id in row["deck"])
        key = str(row["key"])
        if group == "mirror_md_v3":
            controller = LAYERED.LayeredMirrorCardController(
                md_v3, card, qu, "population/md-v3", deck
            )
        elif group == "mirror_md_v4_epoch4":
            controller = RUNTIME.LayeredMDV4Controller(
                initial_candidate,
                md_v3,
                card,
                qu,
                "population/md-v4-epoch4",
                deck,
            )
        elif group == "mirror_ppo_v2":
            controller = LAYERED.LayeredMirrorCardController(
                ppo_v2, card, qu, "population/retired-ppo-v2", deck
            )
        elif group == "mirror_md_v1":
            controller = LAYERED.LayeredMirrorCardController(
                md_v1, None, qu, "population/md-v1", deck
            )
        elif group == "mirror_qu_v2b":
            controller = LAYERED.LayeredMirrorCardController(
                qu, None, qu, "population/qu-v2b-grim", deck
            )
        elif group == "field_qu_v2b":
            controller = LAYERED.LayeredMirrorCardController(
                qu, None, qu, f"population/qu-v2b/{key}", deck
            )
        elif group == "field_rules":
            controller = OLD_POP.DiagnosticRulesController(
                f"population/rules/{key}"
            )
        else:
            raise ResourcePPORunnerError(
                f"unknown population group {group!r}"
            )
        controller_key = f"{index:02d}:{group}:{key}"
        controllers[controller_key] = controller
        move = (
            controller.move
            if isinstance(controller, OLD_POP.DiagnosticRulesController)
            else _layered_move(controller)
        )
        opponents.append(OpponentSpec(
            key=key,
            deck=deck,
            move=move,
            weight=float(row["weight"]),
            policy_id=str(row["policy_id"]),
            schedule_group=group,
        ))
    runtime_manifest = {
        "schema": LOCK.POPULATION_SCHEMA,
        "pilot_mass": dict(LOCK.PILOT_MASS),
        "opponents": [{
            "key": row.key,
            "policy_id": row.policy_id,
            "schedule_group": row.schedule_group,
            "weight": row.weight,
            "deck": list(row.deck),
            "deck_sha256": row.deck_sha256,
        } for row in opponents],
    }
    expected_rows = static["opponents"]
    if runtime_manifest["opponents"] != expected_rows:
        raise ResourcePPORunnerError(
            "runtime opponent population differs from lock"
        )
    return FrozenPopulation(opponents, controllers, runtime_manifest)


def enforce_schedule(
    lock: Mapping[str, Any],
    population: FrozenPopulation,
    *,
    update: int,
) -> list[EpisodeSpec]:
    config = _training_config(lock)
    schedule, actual = LOCK.schedule_contract(
        population.opponents,
        update=update,
        rollout_seed=int(config["rollout_seeds"][update - 1]),
        ppo_seed=int(config["ppo_seeds"][update - 1]),
    )
    expected = lock["schedules"]["updates"][update - 1]
    if actual != expected:
        raise ResourcePPORunnerError(
            f"update {update} schedule differs from lock"
        )
    return schedule


def _controller_snapshot(population: FrozenPopulation) -> dict[str, dict]:
    return {
        key: controller.diagnostics()
        for key, controller in population.controllers.items()
    }


def _counter_delta(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, int]:
    return {
        str(key): int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(after) | set(before))
        if int(after.get(key, 0)) - int(before.get(key, 0))
    }


def _controller_deltas(
    population: FrozenPopulation,
    before: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    scalar_names = (
        "calls",
        "main_routes",
        "parent_main_routes",
        "candidate_attempts",
        "candidate_routes",
        "candidate_fallbacks",
        "card_routes",
        "qu_routes",
        "off_deck_main_routes",
        "off_deck_card_routes",
        "fallbacks",
        "repairs",
    )
    for key, controller in population.controllers.items():
        after = controller.diagnostics()
        prior = before[key]
        result[key] = {
            name: int(after.get(name, 0)) - int(prior.get(name, 0))
            for name in scalar_names
        }
        result[key]["exceptions"] = _counter_delta(
            after.get("exceptions", {}),
            prior.get("exceptions", {}),
        )
        result[key]["fallback_reasons"] = _counter_delta(
            after.get("fallback_reasons", {}),
            prior.get("fallback_reasons", {}),
        )
        result[key]["candidate_fallback_reasons"] = _counter_delta(
            after.get("candidate_fallback_reasons", {}),
            prior.get("candidate_fallback_reasons", {}),
        )
    return result


def _frozen_non_main_action(
    observation: dict,
    deck: Sequence[int],
    card_net: model.QuV2Net,
    qu_net: model.QuV2Net,
) -> list[int]:
    view = ObsView(observation)
    # Non-ST_MAIN routing is the deployed MD-v3/ST_CARD/Qu path.  Encode it
    # directly with the deployed base encoder; the MD-v4 resource window is
    # deliberately out of scope and is not guaranteed to support this prompt.
    deployed = DEPLOYED_FEATURES.encode_public_observation(observation, deck)
    active = card_net if CARD.supports_view(view, deck) else qu_net
    logits, _ = active.forward(deployed)
    return model.decode_qu_v2(
        logits,
        len(view.options),
        view.min_count,
        view.max_count,
    )


def collect_population_games(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
    deployed_parent: model.QuV2Net,
    card_net: model.QuV2Net,
    qu_net: model.QuV2Net,
    deck: Sequence[int],
    population: FrozenPopulation,
    schedule: Sequence[EpisodeSpec],
    *,
    seed: int,
    device: torch.device,
    env_factory: Callable[..., PTCGRLEnv] = PTCGRLEnv,
) -> tuple[list[ResourceDecision], dict[str, Any]]:
    if not schedule:
        raise ResourcePPORunnerError("cannot collect an empty schedule")
    if [row.episode_id for row in schedule] != list(range(len(schedule))):
        raise ResourcePPORunnerError("episode IDs are not consecutive")
    before = _controller_snapshot(population)
    env = env_factory(
        deck,
        population.opponents,
        observation_encoder=lambda observation: observation,
        fault_mode="truncate",
        max_selects=5000,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    decisions: list[ResourceDecision] = []
    outcomes: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    opponents_seen: Counter[str] = Counter()
    seats: Counter[str] = Counter()
    zero_main_games = 0
    net.eval()
    critic.eval()
    try:
        for episode in schedule:
            trajectory: list[ResourceDecision] = []
            observation, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            opponents_seen[
                population.opponents[episode.opponent_index].key
            ] += 1
            seats[str(episode.learner_seat)] += 1
            reward = float(info.get("reward", 0.0))
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            while observation is not None:
                raw = env.raw_observation
                if raw is None:
                    raise ResourcePPORunnerError(
                        "environment lost learner observation"
                    )
                view = ObsView(raw)
                select_index = int(
                    info["seat_selects"][episode.learner_seat]
                )
                if view.select_type == ST_MAIN:
                    routes["st_main"] += 1
                    features = MF.encode_runtime_observation(raw, deck)
                    batch = MM.collate([features], device=device)
                    with torch.no_grad():
                        logits, _ignored_frozen_value = net(batch)
                        representation = _critic_representation(net, batch)
                        old_value = critic(representation)[0]
                        picks, logp, _ = PPO_V1.sample_selection(
                            logits[0],
                            SelectionSpec.from_observation(raw),
                            generator,
                        )
                    deployed_features = CORRECTION._to_deployed_base_features(
                        features
                    )
                    parent_logits, _parent_value = deployed_parent.forward(
                        deployed_features
                    )
                    parent_logits = np.asarray(
                        parent_logits,
                        dtype=np.float32,
                    )
                    if (
                        parent_logits.shape != (len(view.options) + 1,)
                        or not np.isfinite(parent_logits).all()
                    ):
                        raise ResourcePPORunnerError(
                            "deployed parent rollout logits are invalid"
                        )
                    trajectory.append(ResourceDecision(
                        features=features,
                        critic_features=np.array(
                            representation[0].cpu().numpy(),
                            dtype=np.float32,
                            copy=True,
                        ),
                        deployed_parent_logits=np.array(
                            parent_logits,
                            copy=True,
                        ),
                        picks=tuple(int(index) for index in picks),
                        n_options=len(view.options),
                        min_count=view.min_count,
                        max_count=view.max_count,
                        old_logp=float(logp.cpu()),
                        old_value=float(old_value.cpu()),
                        episode_id=episode.episode_id,
                        decision_index=len(trajectory),
                        learner_select_index=select_index,
                    ))
                    action = picks
                else:
                    routes["frozen_non_main"] += 1
                    action = _frozen_non_main_action(
                        raw,
                        deck,
                        card_net,
                        qu_net,
                    )
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
            clean = (
                terminated
                and not truncated
                and info.get("reason") == "engine_terminal"
                and info.get("agent_error") is None
                and info.get("engine_error") is None
                and info.get("result") in ("win", "draw", "loss")
            )
            if not clean:
                outcomes["invalid"] += 1
                invalid_reasons[str(info.get("reason") or "unknown")] += 1
                continue
            outcomes[str(info["result"])] += 1
            if not trajectory:
                zero_main_games += 1
                continue
            PPO_V2._finish_trajectory(
                trajectory,
                final_learner_selects=int(
                    info["seat_selects"][episode.learner_seat]
                ),
                terminal_reward=float(reward),
            )
            decisions.extend(trajectory)
    finally:
        env.close()
    diagnostics = _controller_deltas(population, before)
    controller_faults = sum(
        row["fallbacks"]
        + row["repairs"]
        + row["candidate_fallbacks"]
        + sum(abs(value) for value in row["exceptions"].values())
        for row in diagnostics.values()
    )
    return decisions, {
        "games": len(schedule),
        "outcomes": dict(outcomes),
        "invalid": int(outcomes["invalid"]),
        "invalid_reasons": dict(invalid_reasons),
        "zero_main_games": zero_main_games,
        "routes": dict(routes),
        "opponents": dict(opponents_seen),
        "learner_seats": dict(seats),
        "controller_faults": controller_faults,
        "controllers": diagnostics,
        "deployed_parent_anchor_rows": len(decisions),
        "cached_parent_logits_reads": 0,
    }


def ppo_update(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    decisions: Sequence[ResourceDecision],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    minibatch_size: int,
    clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    parent_kl_coefficient: float,
    gamma: float,
    gae_lambda: float,
    gradient_norm: float,
) -> dict[str, float]:
    if not decisions:
        raise ResourcePPORunnerError("PPO update has no decisions")
    if epochs <= 0 or minibatch_size <= 0:
        raise ValueError("epochs/minibatch size must be positive")
    if not 0.0 <= clip < 1.0 or gradient_norm <= 0.0:
        raise ValueError("clip/gradient norm are invalid")
    _validate_optimizer(optimizer, scopes)
    targets = PPO_V2.compute_smdp_gae(
        decisions,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    raw_advantages = torch.tensor(
        [target.advantage for target in targets],
        dtype=torch.float32,
    )
    advantages = (
        raw_advantages - raw_advantages.mean()
    ) / raw_advantages.std(unbiased=False).clamp(min=1e-6)
    value_targets = torch.tensor(
        [target.value_target for target in targets],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    totals: Counter[str] = Counter()
    batches = 0
    net.train()
    critic.train()
    for _epoch in range(epochs):
        for indices in PPO_V1.minibatches(
            len(decisions),
            minibatch_size,
            generator,
        ):
            rows = [decisions[int(index)] for index in indices]
            batch = MM.collate(
                [row.features for row in rows],
                device=device,
            )
            logits, _ignored_value = net(batch)
            terms = []
            for row_index, row in enumerate(rows):
                parent_logits = torch.as_tensor(
                    row.deployed_parent_logits,
                    dtype=logits.dtype,
                    device=device,
                )
                terms.append(PPO_V1.sequence_statistics(
                    logits[row_index, :row.n_options + 1],
                    row.picks,
                    row.spec,
                    parent_logits=parent_logits,
                ))
            logp = torch.stack([term[0] for term in terms])
            entropy = torch.stack([term[1] for term in terms])
            parent_kl = torch.stack([term[2] for term in terms])
            old_logp = torch.tensor(
                [row.old_logp for row in rows],
                dtype=logp.dtype,
                device=device,
            )
            selected_advantages = advantages[indices].to(device)
            ratio = torch.exp((logp - old_logp).clamp(-20.0, 20.0))
            clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
            policy_loss = -torch.minimum(
                ratio * selected_advantages,
                clipped_ratio * selected_advantages,
            ).mean()
            actor_loss = (
                policy_loss
                - entropy_coefficient * entropy.mean()
                + parent_kl_coefficient * parent_kl.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if any(parameter.grad is not None for parameter in scopes.critic):
                raise ResourcePPORunnerError(
                    "actor loss reached critic parameters"
                )
            actor_gradient = float(torch.nn.utils.clip_grad_norm_(
                scopes.actor,
                gradient_norm,
            ))
            optimizer.step()

            snapshots = torch.as_tensor(
                np.stack([row.critic_features for row in rows]),
                dtype=torch.float32,
                device=device,
            )
            values = critic(snapshots)
            value_loss = (
                values - value_targets[indices].to(device)
            ).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (value_coefficient * value_loss).backward()
            if any(parameter.grad is not None for parameter in scopes.actor):
                raise ResourcePPORunnerError(
                    "critic loss reached actor parameters"
                )
            if any(parameter.grad is not None for parameter in scopes.frozen):
                raise ResourcePPORunnerError(
                    "critic loss reached frozen parent"
                )
            critic_gradient = float(torch.nn.utils.clip_grad_norm_(
                scopes.critic,
                gradient_norm,
            ))
            optimizer.step()
            metrics = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.mean().detach().cpu()),
                "deployed_parent_kl": float(parent_kl.mean().detach().cpu()),
                "ratio": float(ratio.mean().detach().cpu()),
                "actor_gradient_norm": actor_gradient,
                "critic_gradient_norm": critic_gradient,
            }
            totals.update(metrics)
            batches += 1
    if batches <= 0:
        raise ResourcePPORunnerError("PPO update produced no minibatches")
    result = {
        name: float(value / batches)
        for name, value in totals.items()
    }
    result.update({
        "advantage_mean_raw": float(raw_advantages.mean()),
        "advantage_std_raw": float(
            raw_advantages.std(unbiased=False)
        ),
        "value_target_mean": float(value_targets.mean()),
        "cached_parent_logits_reads": 0.0,
    })
    if not all(math.isfinite(value) for value in result.values()):
        raise ResourcePPORunnerError("PPO update produced non-finite metrics")
    return result


def measure_deployed_parent_kl(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    decisions: Sequence[ResourceDecision],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    if not decisions or batch_size <= 0:
        raise ResourcePPORunnerError("cannot measure KL without decisions")
    values = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            rows = decisions[start:start + batch_size]
            logits, _ignored = net(MM.collate(
                [row.features for row in rows],
                device=device,
            ))
            for index, row in enumerate(rows):
                parent = torch.as_tensor(
                    row.deployed_parent_logits,
                    dtype=logits.dtype,
                    device=device,
                )
                _, _, kl = PPO_V1.sequence_statistics(
                    logits[index, :row.n_options + 1],
                    row.picks,
                    row.spec,
                    parent_logits=parent,
                )
                values.append(float(kl.cpu()))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ResourcePPORunnerError("deployed-parent KL is non-finite")
    return {
        "mean": float(array.mean()),
        "maximum": float(array.max()),
        "decisions": int(array.size),
        "cached_parent_logits_reads": 0,
    }


def _parent_state_snapshot(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.parent.state_dict().items()
    }


def _assert_parent_unchanged(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    before: Mapping[str, torch.Tensor],
) -> None:
    current = net.parent.state_dict()
    if (
        set(current) != set(before)
        or any(
            not torch.equal(current[name].detach().cpu(), before[name])
            for name in current
        )
        or MM.frozen_parent_state_sha256(net)
        != LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise ResourcePPORunnerError("embedded MD-v3 parent mutated")


def _rollout_is_clean(
    rollout: Mapping[str, Any],
    *,
    decisions: int,
) -> bool:
    mirror_rows = [
        row
        for key, row in rollout.get("controllers", {}).items()
        if ":mirror_" in str(key)
    ]
    return (
        rollout.get("games") == LOCK.GAMES_PER_UPDATE
        and rollout.get("invalid") == 0
        and rollout.get("controller_faults") == 0
        and rollout.get("cached_parent_logits_reads") == 0
        and rollout.get("deployed_parent_anchor_rows") == decisions
        and sum(
            int(value)
            for value in rollout.get("outcomes", {}).values()
        ) == LOCK.GAMES_PER_UPDATE
        and decisions >= LOCK.MIN_ST_MAIN_PER_UPDATE
        and rollout.get("learner_seats") == LOCK.SEATS_PER_UPDATE
        and bool(mirror_rows)
        and all(
            row.get("off_deck_main_routes") == 0
            and row.get("off_deck_card_routes") == 0
            and row.get("candidate_fallbacks") == 0
            for row in mirror_rows
        )
    )


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    return MDV4_TRAIN._parameter_state_sha256(state)


def _write_recovery(
    output: Path,
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    *,
    update: int,
    lock_sha256: str,
    attempt_sha256: str,
    previous_checkpoint_sha256: str | None,
    resume_consumption_sha256: str | None,
    update_rows: Sequence[Mapping[str, Any]],
) -> Path:
    directory = output / "recovery" / f"update-{update:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / f"RECOVERY-ONLY-update-{update:02d}.pt"
    actor_state = {
        name: tensor.detach().cpu()
        for name, tensor in net.state_dict().items()
    }
    critic_state = {
        name: tensor.detach().cpu()
        for name, tensor in critic.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()
    rows = list(update_rows)
    payload: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "candidate_only": True,
        "recovery_only": True,
        "selection_eligible": False,
        "fixed_terminal_selection_update": LOCK.UPDATES,
        "completed_updates": int(update),
        "lock_sha256": str(lock_sha256),
        "attempt_sha256": str(attempt_sha256),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "resume_consumption_sha256": resume_consumption_sha256,
        "warm_state_dict_sha256": LOCK.WARM_STATE_SHA256,
        "actor_state_dict": actor_state,
        "actor_state_dict_sha256": _state_dict_sha256(actor_state),
        "critic_state_dict": critic_state,
        "critic_state_dict_sha256": _state_dict_sha256(critic_state),
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_dict_sha256":
            _hash_training_object(optimizer_state),
        "actor_parameter_names": list(scopes.actor_names),
        "critic_parameter_names": list(scopes.critic_names),
        "frozen_parameter_names": list(scopes.frozen_names),
        "update_rows": rows,
        "update_rows_sha256": _hash_training_object(rows),
        "deployable_numpy_weights": None,
    }
    payload["checkpoint_payload_sha256"] = _hash_training_object(payload)
    PPO_V1.atomic_torch(path, payload)
    _write_self_hashed_json(
        directory / "RECOVERY-ONLY.json",
        {
            "schema": RECOVERY_MANIFEST_SCHEMA,
            "candidate_only": True,
            "recovery_only": True,
            "selection_eligible": False,
            "completed_updates": int(update),
            "fixed_terminal_selection_update": LOCK.UPDATES,
            "lock_sha256": str(lock_sha256),
            "attempt_sha256": str(attempt_sha256),
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
            "resume_consumption_sha256": resume_consumption_sha256,
            "checkpoint": {
                "path": _display_path(path),
                "sha256": PPO_V1.sha256_file(path),
                "payload_sha256": payload["checkpoint_payload_sha256"],
            },
            "actor_state_dict_sha256":
                payload["actor_state_dict_sha256"],
            "critic_state_dict_sha256":
                payload["critic_state_dict_sha256"],
            "optimizer_state_dict_sha256":
                payload["optimizer_state_dict_sha256"],
            "update_rows_sha256": payload["update_rows_sha256"],
            "deployable_numpy_weights": None,
        },
        hash_key="manifest_sha256",
    )
    return path


def _restore_recovery(
    path: Path,
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    critic: ResourceCritic,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    *,
    output: Path,
    lock_sha256: str,
    attempt_sha256: str,
) -> tuple[int, list[dict[str, Any]], str, dict[str, Any]]:
    path = path.expanduser().resolve()
    output = output.expanduser().resolve()
    try:
        relative = path.relative_to((output / "recovery").resolve())
    except ValueError as exc:
        raise ResourcePPORunnerError(
            "recovery checkpoint is outside the official output tree"
        ) from exc
    if (
        len(relative.parts) != 2
        or not relative.parts[0].startswith("update-")
        or relative.parts[1]
        != f"RECOVERY-ONLY-{relative.parts[0]}.pt"
    ):
        raise ResourcePPORunnerError("recovery checkpoint path is not canonical")
    manifest = _load_self_hashed_json(
        path.parent / "RECOVERY-ONLY.json",
        schema=RECOVERY_MANIFEST_SCHEMA,
        hash_key="manifest_sha256",
    )
    checkpoint_sha256 = PPO_V1.sha256_file(path)
    checkpoint_record = manifest.get("checkpoint")
    if (
        manifest.get("candidate_only") is not True
        or manifest.get("recovery_only") is not True
        or manifest.get("selection_eligible") is not False
        or manifest.get("fixed_terminal_selection_update") != LOCK.UPDATES
        or manifest.get("lock_sha256") != lock_sha256
        or manifest.get("attempt_sha256") != attempt_sha256
        or not isinstance(checkpoint_record, Mapping)
        or checkpoint_record.get("path") != _display_path(path)
        or checkpoint_record.get("sha256") != checkpoint_sha256
        or manifest.get("deployable_numpy_weights") is not None
    ):
        raise ResourcePPORunnerError("recovery manifest contract drifted")
    payload = _load_torch_payload(path)
    recorded_payload_sha256 = payload.get("checkpoint_payload_sha256")
    payload_body = {
        key: value
        for key, value in payload.items()
        if key != "checkpoint_payload_sha256"
    }
    completed = payload.get("completed_updates")
    if (
        payload.get("schema") != RECOVERY_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("recovery_only") is not True
        or payload.get("selection_eligible") is not False
        or payload.get("fixed_terminal_selection_update") != LOCK.UPDATES
        or payload.get("lock_sha256") != lock_sha256
        or payload.get("attempt_sha256") != attempt_sha256
        or payload.get("warm_state_dict_sha256") != LOCK.WARM_STATE_SHA256
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 1 <= completed < LOCK.UPDATES
        or payload.get("actor_parameter_names") != list(scopes.actor_names)
        or payload.get("critic_parameter_names") != list(scopes.critic_names)
        or payload.get("frozen_parameter_names") != list(scopes.frozen_names)
        or payload.get("deployable_numpy_weights") is not None
        or recorded_payload_sha256 != _hash_training_object(payload_body)
        or checkpoint_record.get("payload_sha256")
        != recorded_payload_sha256
        or manifest.get("completed_updates") != completed
        or manifest.get("previous_checkpoint_sha256")
        != payload.get("previous_checkpoint_sha256")
        or manifest.get("resume_consumption_sha256")
        != payload.get("resume_consumption_sha256")
    ):
        raise ResourcePPORunnerError("recovery contract drifted")
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if (
        not isinstance(actor_state, Mapping)
        or payload.get("actor_state_dict_sha256")
        != _state_dict_sha256(actor_state)
        or not isinstance(critic_state, Mapping)
        or payload.get("critic_state_dict_sha256")
        != _state_dict_sha256(critic_state)
        or manifest.get("actor_state_dict_sha256")
        != payload.get("actor_state_dict_sha256")
        or manifest.get("critic_state_dict_sha256")
        != payload.get("critic_state_dict_sha256")
    ):
        raise ResourcePPORunnerError("recovery tensor hash is invalid")
    optimizer_state = payload.get("optimizer_state_dict")
    if (
        not isinstance(optimizer_state, Mapping)
        or payload.get("optimizer_state_dict_sha256")
        != _hash_training_object(optimizer_state)
        or manifest.get("optimizer_state_dict_sha256")
        != payload.get("optimizer_state_dict_sha256")
    ):
        raise ResourcePPORunnerError("recovery optimizer hash is invalid")
    rows = payload.get("update_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != completed
        or [row.get("update") for row in rows]
        != list(range(1, completed + 1))
        or payload.get("update_rows_sha256")
        != _hash_training_object(rows)
        or manifest.get("update_rows_sha256")
        != payload.get("update_rows_sha256")
    ):
        raise ResourcePPORunnerError("recovery history is invalid")
    net.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    _validate_optimizer(
        optimizer,
        scopes,
        actor_learning_rate=LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=LOCK.CRITIC_LEARNING_RATE,
    )
    if (
        _state_dict_sha256(net.state_dict())
        != payload["actor_state_dict_sha256"]
        or _state_dict_sha256(critic.state_dict())
        != payload["critic_state_dict_sha256"]
        or _hash_training_object(optimizer.state_dict())
        != payload["optimizer_state_dict_sha256"]
    ):
        raise ResourcePPORunnerError("restored recovery state changed identity")
    return (
        completed,
        [dict(row) for row in rows],
        checkpoint_sha256,
        manifest,
    )


def _consume_resume(
    output: Path,
    resume_from: Path,
    *,
    lock_sha256: str,
    attempt_sha256: str,
) -> dict[str, Any]:
    """Consume exactly the latest complete recovery before loading it."""
    output = output.expanduser().resolve()
    resume_from = resume_from.expanduser().resolve()
    recovery_root = output / "recovery"
    if not recovery_root.is_dir():
        raise ResourcePPORunnerError("official output has no recovery tree")
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for directory in sorted(recovery_root.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("update-"):
            raise ResourcePPORunnerError("recovery tree contains an alien entry")
        try:
            update = int(directory.name.removeprefix("update-"))
        except ValueError as exc:
            raise ResourcePPORunnerError(
                "recovery directory has a non-integer update"
            ) from exc
        if directory.name != f"update-{update:02d}":
            raise ResourcePPORunnerError(
                "recovery directory path is not canonical"
            )
        checkpoint = directory / f"RECOVERY-ONLY-update-{update:02d}.pt"
        manifest_path = directory / "RECOVERY-ONLY.json"
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise ResourcePPORunnerError(
                "recovery tree contains an incomplete update"
            )
        manifest = _load_self_hashed_json(
            manifest_path,
            schema=RECOVERY_MANIFEST_SCHEMA,
            hash_key="manifest_sha256",
        )
        complete.append((update, checkpoint.resolve(), manifest))
    if not complete:
        raise ResourcePPORunnerError("official output has no complete recovery")
    complete.sort(key=lambda row: row[0])
    if [row[0] for row in complete] != list(
        range(1, complete[-1][0] + 1)
    ):
        raise ResourcePPORunnerError("recovery lineage is not contiguous")
    consumption_by_sha: dict[str, dict[str, Any]] = {}
    for completed_update, checkpoint, manifest in complete:
        consumption_path = checkpoint.parent / RESUME_CONSUMPTION_FILE
        if consumption_path.exists():
            consumption = _load_self_hashed_json(
                consumption_path,
                schema=RESUME_CONSUMPTION_SCHEMA,
                hash_key="resume_consumption_sha256",
            )
            if (
                consumption.get("candidate") != LOCK.CANDIDATE
                or consumption.get("lock_sha256") != lock_sha256
                or consumption.get("attempt_sha256") != attempt_sha256
                or consumption.get("completed_updates") != completed_update
                or consumption.get("checkpoint_path")
                != _display_path(checkpoint)
                or consumption.get("checkpoint_sha256")
                != PPO_V1.sha256_file(checkpoint)
                or consumption.get("manifest_sha256")
                != manifest.get("manifest_sha256")
                or consumption.get("written_before_restore") is not True
                or consumption.get("consumed") is not True
                or consumption.get("reusable") is not False
            ):
                raise ResourcePPORunnerError(
                    "resume-consumption lineage is invalid"
                )
            consumption_by_sha[
                str(consumption["resume_consumption_sha256"])
            ] = consumption
    previous_sha256: str | None = None
    for completed_update, checkpoint, manifest in complete:
        checkpoint_record = manifest.get("checkpoint")
        resume_sha256 = manifest.get("resume_consumption_sha256")
        if (
            manifest.get("completed_updates") != completed_update
            or manifest.get("lock_sha256") != lock_sha256
            or manifest.get("attempt_sha256") != attempt_sha256
            or manifest.get("previous_checkpoint_sha256")
            != previous_sha256
            or not isinstance(checkpoint_record, Mapping)
            or checkpoint_record.get("path") != _display_path(checkpoint)
            or checkpoint_record.get("sha256")
            != PPO_V1.sha256_file(checkpoint)
            or (
                resume_sha256 is not None
                and (
                    resume_sha256 not in consumption_by_sha
                    or int(
                        consumption_by_sha[resume_sha256][
                            "completed_updates"
                        ]
                    ) >= completed_update
                )
            )
        ):
            raise ResourcePPORunnerError(
                "recovery checkpoint lineage is invalid"
            )
        previous_sha256 = str(checkpoint_record["sha256"])
    update, checkpoint, manifest = max(complete, key=lambda row: row[0])
    if resume_from != checkpoint:
        raise ResourcePPORunnerError(
            "resume must consume the latest complete recovery"
        )
    if manifest.get("completed_updates") != update:
        raise ResourcePPORunnerError("latest recovery update identity drifted")
    checkpoint_record = manifest.get("checkpoint")
    if (
        manifest.get("lock_sha256") != lock_sha256
        or manifest.get("attempt_sha256") != attempt_sha256
        or not isinstance(checkpoint_record, Mapping)
        or checkpoint_record.get("path") != _display_path(checkpoint)
        or checkpoint_record.get("sha256") != PPO_V1.sha256_file(checkpoint)
    ):
        raise ResourcePPORunnerError("latest recovery manifest is invalid")
    consumption_path = checkpoint.parent / RESUME_CONSUMPTION_FILE
    if consumption_path.exists():
        raise ResourcePPORunnerError("latest recovery was already consumed")
    return _write_self_hashed_json(
        consumption_path,
        {
            "schema": RESUME_CONSUMPTION_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "written_before_restore": True,
            "candidate": LOCK.CANDIDATE,
            "lock_sha256": str(lock_sha256),
            "attempt_sha256": str(attempt_sha256),
            "completed_updates": update,
            "checkpoint_path": _display_path(checkpoint),
            "checkpoint_sha256": checkpoint_record["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "consumed": True,
            "reusable": False,
        },
        hash_key="resume_consumption_sha256",
    )


def _initial_delta_population(
    candidate: MM.NumpyMDV4,
    initial: MM.NumpyMDV4,
    samples,
    *,
    expected_callbacks: int,
    expected_games: int,
) -> dict[str, Any]:
    callbacks = 0
    games: set[str] = set()
    changed_games: set[str] = set()
    disagreements = 0
    faults = 0
    for minibatch in MDV4_TRAIN._batches(samples):
        for sample in minibatch:
            callbacks += 1
            games.add(sample.game_uid)
            try:
                candidate_logits, _ = candidate.forward(sample.features)
                initial_logits, _ = initial.forward(sample.features)
                left = CORRECTION._production_action(
                    candidate_logits,
                    sample,
                )
                right = CORRECTION._production_action(
                    initial_logits,
                    sample,
                )
            except Exception:
                faults += 1
                continue
            if left != right:
                disagreements += 1
                changed_games.add(sample.game_uid)
    return {
        "callbacks": callbacks,
        "games": len(games),
        "disagreements": disagreements,
        "disagreement_rate": (
            disagreements / callbacks if callbacks else 0.0
        ),
        "games_touched": len(changed_games),
        "games_touched_rate": (
            len(changed_games) / len(games) if games else 0.0
        ),
        "faults": faults,
        "passed": (
            callbacks == expected_callbacks
            and len(games) == expected_games
            and faults == 0
            and disagreements >= LOCK.MIN_INITIAL_DISAGREEMENTS
            and len(changed_games) >= LOCK.MIN_INITIAL_GAMES_TOUCHED
        ),
    }


def offline_gate_report(
    *,
    conformance: Mapping[str, Any],
    final_validation: Mapping[str, Any] | None,
    initial_delta: Mapping[str, Any],
) -> dict[str, Any]:
    parent_passed = bool(
        final_validation is not None
        and final_validation.get("samples") == LOCK.EXPECTED_VALIDATION_CALLBACKS
        and final_validation.get("games") == LOCK.EXPECTED_VALIDATION_GAMES
        and float(final_validation.get("parent_kl", math.inf))
        <= LOCK.MAX_PARENT_KL
        and int(final_validation.get("greedy_disagreements", -1))
        >= LOCK.MIN_PARENT_DISAGREEMENTS
        and int(final_validation.get("games_touched", -1))
        >= LOCK.MIN_PARENT_GAMES_TOUCHED
    )
    initial_passed = bool(initial_delta.get("passed") is True)
    passed = bool(
        conformance.get("passed") is True
        and conformance.get("cached_parent_logits_reads") == 0
        and parent_passed
        and initial_passed
    )
    return {
        "passed": passed,
        "role": "rejection/behavior-sizing only",
        "full_deployed_parent_runtime_conformance_passed":
            conformance.get("passed") is True,
        "cached_parent_logits_reads":
            conformance.get("cached_parent_logits_reads"),
        "deployed_parent": {
            "maximum_kl_inclusive": LOCK.MAX_PARENT_KL,
            "minimum_disagreements_inclusive":
                LOCK.MIN_PARENT_DISAGREEMENTS,
            "minimum_games_touched_inclusive":
                LOCK.MIN_PARENT_GAMES_TOUCHED,
            "passed": parent_passed,
        },
        "warm_start": {
            "minimum_disagreements_inclusive":
                LOCK.MIN_INITIAL_DISAGREEMENTS,
            "minimum_games_touched_inclusive":
                LOCK.MIN_INITIAL_GAMES_TOUCHED,
            "passed": initial_passed,
        },
    }


def _validation_context():
    config = MDV4_TRAIN.TrainingConfig(
        lock_path=MDV4_TRAIN.TRAINING_LOCK_PATH,
        device="cuda",
    )
    MDV4_TRAIN._validate_config(config)
    plan = MDV4_TRAIN.load_locked_corpus(config)
    cache = MDV4_TRAIN.create_thin_cache(config, plan)
    _, training_lock_sha256 = MDV4_TRAIN.load_training_lock(
        config,
        plan,
        cache,
    )
    base_index = MDV4_TRAIN.index_base_cache(config, plan)
    materialization, records = MDV4_TRAIN._load_materialization_payload(
        cache,
        plan,
        training_lock_sha256,
    )
    summary = materialization["summary"]["by_split"]["validation"]
    if (
        summary["target_decisions"] != LOCK.EXPECTED_VALIDATION_CALLBACKS
        or summary["games"] != LOCK.EXPECTED_VALIDATION_GAMES
    ):
        raise ResourcePPORunnerError("validation population drifted")

    def samples():
        return MDV4_TRAIN.iter_split_samples(
            config,
            cache,
            base_index,
            records,
            training_lock_sha256,
            "validation",
            epoch=None,
        )

    return samples


def evaluate_terminal_numpy(
    exported: Mapping[str, np.ndarray],
    initial_arrays: Mapping[str, np.ndarray],
    weights_path: Path,
    deployed_parent: model.QuV2Net,
) -> dict[str, Any]:
    """Run complete staged-NumPy gates against deployed MD-v3 and warm start."""
    with np.load(weights_path, allow_pickle=False) as archive:
        staged_arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    mapping_passed, mapping_mismatches = _same_mapping(
        exported,
        staged_arrays,
    )
    if not mapping_passed:
        raise ResourcePPORunnerError(
            "terminal staged NumPy mapping differs from export"
        )
    research = MM.NumpyMDV4(exported)
    staged = MM.NumpyMDV4(staged_arrays)
    initial = MM.NumpyMDV4(initial_arrays)
    sample_factory = _validation_context()
    conformance, final_validation = (
        CORRECTION._identity_and_corrected_offline_population(
            research,
            staged,
            deployed_parent,
            sample_factory(),
            expected_callbacks=LOCK.EXPECTED_VALIDATION_CALLBACKS,
            expected_games=LOCK.EXPECTED_VALIDATION_GAMES,
        )
    )
    initial_delta = _initial_delta_population(
        staged,
        initial,
        sample_factory(),
        expected_callbacks=LOCK.EXPECTED_VALIDATION_CALLBACKS,
        expected_games=LOCK.EXPECTED_VALIDATION_GAMES,
    )
    offline = offline_gate_report(
        conformance=conformance,
        final_validation=final_validation,
        initial_delta=initial_delta,
    )
    return {
        "array_identity": {
            "passed": mapping_passed,
            "mismatched_fields": mapping_mismatches,
            "in_memory_mapping_sha256": MM._mapping_sha256(exported),
            "staged_mapping_sha256": MM._mapping_sha256(staged_arrays),
        },
        "deployed_parent_conformance": conformance,
        "final_validation": final_validation,
        "warm_start_delta": initial_delta,
        "offline_rejection_gates": offline,
        "cached_parent_logits_used": False,
        "passed": bool(mapping_passed and offline["passed"]),
    }


def _scope_delta(
    current: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    changed = 0
    count = 0
    for name in names:
        difference = current[name].detach().cpu() - initial[name].detach().cpu()
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(torch.count_nonzero(difference))
        count += difference.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": maximum,
        "changed_parameters": changed,
        "total_parameters": count,
    }


def _terminal_checkpoint_payload(
    net: EXPLICIT.TorchMDV4ExplicitFP32,
    scopes: ParameterScopes,
    *,
    lock_sha256: str,
) -> dict[str, Any]:
    """Build the actor-only fixed-terminal deployment checkpoint."""
    terminal_state = {
        name: tensor.detach().cpu()
        for name, tensor in net.state_dict().items()
    }
    return {
        "schema": TERMINAL_CHECKPOINT_SCHEMA,
        "candidate_only": True,
        # Eligibility is recorded only after the staged NumPy screens.
        "selection_eligible": False,
        "offline_gates_pending_at_write": True,
        "completed_updates": LOCK.UPDATES,
        "lock_sha256": lock_sha256,
        "warm_state_dict_sha256": LOCK.WARM_STATE_SHA256,
        "actor_state_dict": terminal_state,
        "actor_state_dict_sha256": _state_dict_sha256(terminal_state),
        "training_only_critic_included": False,
        "optimizer_included": False,
        "deployment_candidate": True,
        "actor_parameter_names": list(scopes.actor_names),
        "frozen_parameter_names": list(scopes.frozen_names),
    }


def _execute_training_body(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_dir: Path,
    *,
    device: torch.device,
    resume_from: Path | None = None,
    attempt_sha256: str,
    resume_consumption: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = _training_config(lock)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ResourcePPORunnerError(
            "the locked scientific run requires available CUDA"
        )
    output = _assert_output(output_dir, must_not_exist=False)
    if not output.is_dir():
        raise ResourcePPORunnerError("official attempt directory is absent")
    seed = int(config["rollout_seed_base"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (
        torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise ResourcePPORunnerError("CUDA matmul TF32 must be disabled")

    required = set(LOCK.REQUIRED_ARTIFACTS)
    if set(paths) != required:
        raise ResourcePPORunnerError("verified lock artifact set is incomplete")
    if (
        paths["runner"] != Path(__file__).resolve()
        or paths["lock_builder"] != Path(LOCK.__file__).resolve()
        or paths["md_v4_explicit_reference"]
        != Path(EXPLICIT.__file__).resolve()
    ):
        raise ResourcePPORunnerError("lock names different implementations")
    input_hashes_before = {
        name: PPO_V1.sha256_file(path)
        for name, path in paths.items()
    }
    deck = tuple(
        int(line)
        for line in paths["deck"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if deck != LOCK.TARGET_DECK:
        raise ResourcePPORunnerError("learner deck drifted")
    net, warm_checkpoint, initial_arrays = load_warm_start(
        paths,
        device=device,
    )
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
    }
    parent_before = _parent_state_snapshot(net)
    critic = ResourceCritic().to(device)
    optimizer, scopes = make_optimizer(
        net,
        critic,
        actor_learning_rate=float(config["actor_learning_rate"]),
        critic_learning_rate=float(config["critic_learning_rate"]),
    )
    deployed_parent = _deployed_parent(paths)
    initial_numpy = MM.NumpyMDV4(initial_arrays)
    card_net = _load_qu_net(paths["card_weights"])
    qu_net = _load_qu_net(paths["qu_v2b_weights"])
    population = build_population(
        lock,
        paths,
        initial_candidate=initial_numpy,
    )
    completed = 0
    update_rows: list[dict[str, Any]] = []
    previous_checkpoint_sha256: str | None = None
    resume_consumption_sha256 = (
        str(resume_consumption["resume_consumption_sha256"])
        if resume_consumption is not None else None
    )
    if resume_from is not None:
        (
            completed,
            update_rows,
            previous_checkpoint_sha256,
            _,
        ) = _restore_recovery(
            resume_from.expanduser().resolve(),
            net,
            critic,
            optimizer,
            scopes,
            output=output,
            lock_sha256=str(lock["lock_sha256"]),
            attempt_sha256=attempt_sha256,
        )
        _assert_parent_unchanged(net, parent_before)

    started = time.time()
    last_decisions: list[ResourceDecision] = []
    for update in range(completed + 1, LOCK.UPDATES + 1):
        _assert_attempt_open(output)
        schedule = enforce_schedule(lock, population, update=update)
        update_started = time.time()
        last_decisions, rollout = collect_population_games(
            net,
            critic,
            deployed_parent,
            card_net,
            qu_net,
            deck,
            population,
            schedule,
            seed=int(config["rollout_seeds"][update - 1]),
            device=device,
        )
        if not _rollout_is_clean(
            rollout,
            decisions=len(last_decisions),
        ):
            raise ResourcePPORunnerError(
                f"update {update} rollout failed cleanliness gate"
            )
        optimizer_identity = id(optimizer)
        metrics = ppo_update(
            net,
            critic,
            optimizer,
            scopes,
            last_decisions,
            device=device,
            seed=int(config["ppo_seeds"][update - 1]),
            epochs=int(config["ppo_epochs"]),
            minibatch_size=int(config["minibatch_size"]),
            clip=float(config["clip"]),
            value_coefficient=float(config["value_coefficient"]),
            entropy_coefficient=float(config["entropy_coefficient"]),
            parent_kl_coefficient=float(
                config["deployed_parent_kl_coefficient"]
            ),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            gradient_norm=float(config["gradient_norm"]),
        )
        if id(optimizer) != optimizer_identity:
            raise ResourcePPORunnerError("persistent optimizer was replaced")
        _assert_parent_unchanged(net, parent_before)
        post_kl = measure_deployed_parent_kl(
            net,
            last_decisions,
            device=device,
        )
        if (
            float(metrics["deployed_parent_kl"]) > LOCK.MAX_PARENT_KL
            or post_kl["mean"] > LOCK.MAX_PARENT_KL
        ):
            raise ResourcePPORunnerError(
                f"update {update} exceeded deployed-parent KL ceiling"
            )
        row = {
            "update": update,
            "rollout_seed": int(config["rollout_seeds"][update - 1]),
            "ppo_seed": int(config["ppo_seeds"][update - 1]),
            "st_main_decisions": len(last_decisions),
            "rollout": rollout,
            "ppo": metrics,
            "post_update_deployed_parent_kl": post_kl,
            "timing_seconds": time.time() - update_started,
        }
        update_rows.append(row)
        recovery = None
        if update < LOCK.UPDATES:
            recovery = _write_recovery(
                output,
                net,
                critic,
                optimizer,
                scopes,
                update=update,
                lock_sha256=str(lock["lock_sha256"]),
                attempt_sha256=attempt_sha256,
                previous_checkpoint_sha256=previous_checkpoint_sha256,
                resume_consumption_sha256=resume_consumption_sha256,
                update_rows=update_rows,
            )
            previous_checkpoint_sha256 = PPO_V1.sha256_file(recovery)
        print(json.dumps({
            "update": update,
            "games": rollout["games"],
            "st_main_decisions": len(last_decisions),
            "deployed_parent_kl": post_kl["mean"],
            "recovery_only": (
                str(recovery) if recovery is not None else None
            ),
        }, sort_keys=True), flush=True)

    if len(update_rows) != LOCK.UPDATES or not last_decisions:
        raise ResourcePPORunnerError("fixed terminal update was not reached")
    _assert_attempt_open(output)
    _assert_parent_unchanged(net, parent_before)
    terminal_dir = output / "terminal-update-24-candidate"
    terminal_dir.mkdir(parents=True, exist_ok=False)
    weights_path = terminal_dir / "candidate-md-v4-resource-ppo-weights.npz"
    exported = {
        name: np.array(value, copy=True, order="C")
        for name, value in MM.export_numpy_weights(net).items()
    }
    if not all(np.asarray(value).flags.c_contiguous for value in exported.values()):
        raise ResourcePPORunnerError("terminal export is not C-contiguous")
    PPO_V1.atomic_npz(weights_path, exported)
    checkpoint_path = (
        terminal_dir / "candidate-md-v4-resource-ppo-checkpoint.pt"
    )
    checkpoint = _terminal_checkpoint_payload(
        net,
        scopes,
        lock_sha256=str(lock["lock_sha256"]),
    )
    PPO_V1.atomic_torch(checkpoint_path, checkpoint)
    offline = evaluate_terminal_numpy(
        exported,
        initial_arrays,
        weights_path,
        deployed_parent,
    )
    current = net.state_dict()
    parameter_delta = {
        "actor": _scope_delta(
            current,
            initial_state,
            scopes.actor_names,
        ),
        "frozen": _scope_delta(
            current,
            initial_state,
            scopes.frozen_names,
        ),
    }
    if (
        parameter_delta["actor"]["changed_parameters"] <= 0
        or parameter_delta["frozen"]["changed_parameters"] != 0
    ):
        raise ResourcePPORunnerError("terminal parameter scope gate failed")
    verified_after = LOCK.verify_bound_artifacts(lock)
    input_hashes_after = {
        name: PPO_V1.sha256_file(path)
        for name, path in verified_after.items()
    }
    if input_hashes_before != input_hashes_after:
        raise ResourcePPORunnerError("one or more locked inputs changed")
    result = {
        "schema": RESULT_SCHEMA,
        "lock_sha256": lock["lock_sha256"],
        "candidate": LOCK.CANDIDATE,
        "candidate_only": True,
        "attempt_sha256": attempt_sha256,
        "resume_consumption_sha256": resume_consumption_sha256,
        "selection": {
            "eligible": bool(offline["passed"]),
            "selected_update": LOCK.UPDATES,
            "fixed_terminal_update": LOCK.UPDATES,
            "intermediate_recovery_eligible": False,
            "checkpoint_cherry_picking": False,
        },
        "device": str(device),
        "rollout_actor": (
            "tools.research.md_v4_explicit_reference."
            "TorchMDV4ExplicitFP32"
        ),
        "cuda_matmul_tf32": False,
        "training_only_critic_exported": False,
        "updates": update_rows,
        "total_games": sum(
            row["rollout"]["games"] for row in update_rows
        ),
        "total_st_main_decisions": sum(
            row["st_main_decisions"] for row in update_rows
        ),
        "parameter_delta": parameter_delta,
        "offline": offline,
        "artifacts": {
            "weights": {
                "path": _display_path(weights_path),
                "sha256": PPO_V1.sha256_file(weights_path),
                "mapping_sha256": MM._mapping_sha256(exported),
                "contains_critic": False,
            },
            "checkpoint": {
                "path": _display_path(checkpoint_path),
                "sha256": PPO_V1.sha256_file(checkpoint_path),
                "contains_critic": False,
                "contains_optimizer": False,
            },
        },
        "timing_seconds": {"total": time.time() - started},
        "training_gate": {
            "passed": bool(offline["passed"]),
            "zero_invalid_or_controller_faults": True,
            "minimum_st_main_per_update": LOCK.MIN_ST_MAIN_PER_UPDATE,
            "frozen_parent_byte_unchanged": True,
            "actor_scope_changed": True,
            "maximum_parent_kl": LOCK.MAX_PARENT_KL,
            "terminal_selection_only": True,
            "cached_parent_logits_used": False,
        },
        "temporal_archive_opened": False,
        "promotion_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = LOCK.canonical_sha256(result)
    PPO_V1.atomic_json(output / "result.json", result)
    return result


def execute_training(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_dir: Path,
    *,
    device: torch.device,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Execute the sole attempt, with immutable fresh/resume consumption."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ResourcePPORunnerError(
            "the locked scientific run requires available CUDA"
        )
    output = output_dir.expanduser().resolve()
    lock_sha256 = str(lock["lock_sha256"])
    with _exclusive_worker(output):
        resume_consumption: dict[str, Any] | None = None
        if resume_from is None:
            attempt = _create_attempt(output, lock_sha256=lock_sha256)
        else:
            output = _assert_output(output, must_not_exist=False)
            if not output.is_dir():
                raise ResourcePPORunnerError("official attempt is absent")
            _assert_attempt_open(output)
            attempt = _load_attempt(output, lock_sha256=lock_sha256)
            resume_consumption = _consume_resume(
                output,
                resume_from,
                lock_sha256=lock_sha256,
                attempt_sha256=str(attempt["attempt_sha256"]),
            )
        attempt_sha256 = str(attempt["attempt_sha256"])
        try:
            result = _execute_training_body(
                lock,
                paths,
                output,
                device=device,
                resume_from=resume_from,
                attempt_sha256=attempt_sha256,
                resume_consumption=resume_consumption,
            )
            _write_completion(
                output,
                lock_sha256=lock_sha256,
                attempt_sha256=attempt_sha256,
                result=result,
            )
            return result
        except BaseException as error:
            _write_retirement(
                output,
                lock_sha256=lock_sha256,
                attempt_sha256=attempt_sha256,
                error=error,
                resume_consumption_sha256=(
                    str(resume_consumption["resume_consumption_sha256"])
                    if resume_consumption is not None else None
                ),
            )
            raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(LOCK.OFFICIAL_TRAINING_OUTPUT),
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    lock_path = Path(args.lock).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output != LOCK.OFFICIAL_TRAINING_OUTPUT.resolve():
        parser.error(
            "the official one-shot output path is fixed to "
            f"{LOCK.OFFICIAL_TRAINING_OUTPUT}"
        )
    lock = LOCK.load_lock(lock_path, verify_artifacts=True)
    paths = LOCK.verify_bound_artifacts(lock)
    device = torch.device(
        "cuda" if args.device == "auto" else args.device
    )
    if device.type != "cuda":
        parser.error("the official scientific run is CUDA-only")
    result = execute_training(
        lock,
        paths,
        output,
        device=device,
        resume_from=(
            Path(args.resume_from).expanduser().resolve()
            if args.resume_from else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
