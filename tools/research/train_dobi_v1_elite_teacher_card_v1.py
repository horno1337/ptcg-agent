"""Train the two locked selective elite-teacher ST_CARD arms."""

from __future__ import annotations

from collections import Counter
import copy
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as PROD_MODEL, qu_v2_features as PROD_QF  # noqa: E402
from tools.research import (  # noqa: E402
    lock_dobi_v1_elite_teacher_card_v1 as LOCK,
    prepare_dobi_v1_elite_teacher_card_v1 as PREP,
    prepare_dobi_v1_elite_teacher_main_v1 as MAIN_PREP,
    qu_v2a_features as QF,
    qu_v2a_model as QM,
    train_dobi_v1_elite_teacher_main_v1 as CORE,
    train_qu_v2a as BASE_TRAIN,
)


RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.training-result.v1"
CHECKPOINT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.checkpoint.v1"
HISTORY_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.history.v1"
ARM_FILENAMES = (
    "checkpoint.pt", "candidate-qu-v2a-weights.npz", "training-history.json",
)


class CardTrainingError(RuntimeError):
    """The locked ST_CARD optimizer failed closed."""


def _contract(row: Mapping[str, Any]) -> tuple[int, int, int]:
    observation = row.get("observation")
    select = observation.get("select") if isinstance(observation, Mapping) else None
    if not isinstance(select, Mapping) or int(select.get("type", -1)) != 1:
        raise CardTrainingError("training row is not ST_CARD")
    options = select.get("option")
    if not isinstance(options, list) or not options:
        raise CardTrainingError("training row has no option menu")
    return len(options), int(select.get("minCount", 1)), int(select.get("maxCount", 1))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CardTrainingError(f"non-object JSONL row {number}")
                rows.append(row)
    except (OSError, TypeError, ValueError) as error:
        raise CardTrainingError(f"cannot read {path}: {error}") from error
    return rows


def _inventory(lock: Mapping[str, Any], cohort: str) -> dict[int, Mapping[str, Any]]:
    cohorts = lock.get("cohorts")
    value = cohorts.get(cohort) if isinstance(cohorts, Mapping) else None
    games = value.get("games") if isinstance(value, Mapping) else None
    if not isinstance(games, list):
        raise CardTrainingError(f"missing locked {cohort} inventory")
    result = {int(row["episode_id"]): row for row in games}
    if len(result) != len(games):
        raise CardTrainingError(f"duplicate episode in {cohort} inventory")
    return result


def _encode_twins(observation: Mapping[str, Any], deck: tuple[int, ...]):
    research = QF.encode_public_observation(observation, deck)
    production = PROD_QF.encode_public_observation(observation, deck)
    left, right = research.arrays(), production.arrays()
    if set(left) != set(right) or any(
        left[name].dtype != right[name].dtype
        or left[name].shape != right[name].shape
        or not np.array_equal(left[name], right[name])
        for name in left
    ):
        raise CardTrainingError("research and production features differ")
    return research, production


def _load_parent() -> tuple[QM.TorchQuV2A, PROD_MODEL.QuV2Net, tuple[int, ...]]:
    if LOCK.sha256_file(LOCK.PARENT_CHECKPOINT) != LOCK.PARENT_CHECKPOINT_SHA256:
        raise CardTrainingError("parent checkpoint drifted")
    payload = torch.load(
        LOCK.PARENT_CHECKPOINT, map_location="cpu", weights_only=True,
    )
    if (
        payload.get("schema") != BASE_TRAIN.TRAINING_SCHEMA
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("feature_dependency_fingerprint")
            != BASE_TRAIN._feature_contract_fingerprint()
        or payload.get("model_schema") != QM.MODEL_SCHEMA
        or payload.get("model_implementation_sha256")
            != BASE_TRAIN._model_implementation_sha256()
        or payload.get("state_dict_sha256")
            != BASE_TRAIN._state_dict_sha256(payload.get("state_dict", {}))
    ):
        raise CardTrainingError("parent checkpoint contract failed")
    architecture = tuple(int(value) for value in payload.get("architecture", ()))
    if len(architecture) != 5:
        raise CardTrainingError("parent architecture is invalid")
    torch_parent = QM.TorchQuV2A(*architecture)
    torch_parent.load_state_dict(payload["state_dict"], strict=True)
    torch_parent.eval()
    if LOCK.sha256_file(LOCK.PARENT_NPZ) != LOCK.PARENT_NPZ_SHA256:
        raise CardTrainingError("parent NumPy artifact drifted")
    with np.load(LOCK.PARENT_NPZ, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = QM.export_numpy_weights(torch_parent)
    if set(arrays) != set(expected) or any(
        arrays[name].dtype != expected[name].dtype
        or arrays[name].shape != expected[name].shape
        or not np.array_equal(arrays[name], expected[name])
        for name in expected
    ):
        raise CardTrainingError("parent Torch/NumPy artifacts differ")
    return torch_parent, PROD_MODEL.QuV2Net(arrays), architecture


def load_states(
    lock: Mapping[str, Any], extraction: Mapping[str, Any],
) -> tuple[list[CORE.PreferenceState], list[CORE.PolicyState]]:
    if (
        LOCK.sha256_file(LOCK.PREFERENCES)
            != extraction.get("preferences", {}).get("compressed_sha256")
        or LOCK.sha256_file(LOCK.PRESERVATION)
            != extraction.get("preservation", {}).get("compressed_sha256")
    ):
        raise CardTrainingError("extracted JSONL artifacts drifted")
    deck = LOCK.target_deck()
    teacher_games = _inventory(lock, "teacher")
    dobi_games = _inventory(lock, "preservation")
    preferences: list[CORE.PreferenceState] = []
    preservation: list[CORE.PolicyState] = []
    pref_seen: set[tuple[int, int]] = set()
    preserve_seen: set[tuple[int, int]] = set()
    split_preferences: Counter[str] = Counter()
    family_preferences: Counter[str] = Counter()
    split_preservation: Counter[str] = Counter()

    for row in _read_jsonl(LOCK.PREFERENCES):
        if row.get("schema") != PREP.PREFERENCE_SCHEMA:
            raise CardTrainingError("preference schema drifted")
        episode_id = int(row.get("episode_id", -1))
        prompt_index = int(row.get("prompt_index", -1))
        game = teacher_games.get(episode_id)
        key = (episode_id, prompt_index)
        if game is None or prompt_index < 0 or key in pref_seen:
            raise CardTrainingError("preference prompt identity is invalid")
        pref_seen.add(key)
        split = str(row.get("supervision_split"))
        family = str(row.get("family"))
        if (
            game.get("outcome") != "win"
            or split != game.get("supervision_split")
            or family not in LOCK.FAMILY_WEIGHTS
            or row.get("teacher_outcome") != "win"
            or row.get("exact_mirror") is not True
        ):
            raise CardTrainingError("preference metadata disagrees with lock")
        n_options, min_count, max_count = _contract(row)
        preferred = MAIN_PREP.validate_action_sequence(
            row.get("preferred", ()), n_options, min_count, max_count,
        )
        rejected = MAIN_PREP.validate_action_sequence(
            row.get("rejected", ()), n_options, min_count, max_count,
        )
        parent_action = MAIN_PREP.validate_action_sequence(
            row.get("parent_action", ()), n_options, min_count, max_count,
        )
        view = __import__("agent.obsview", fromlist=["ObsView"]).ObsView(
            row["observation"]
        )
        semantics = __import__(
            "tools.research.analyze_dobi_v1_elite_card_disagreement",
            fromlist=["semantic_action_signature"],
        )
        if (
            tuple(sorted(preferred)) != preferred
            or tuple(sorted(rejected)) != rejected
            or semantics.semantic_action_signature(view, preferred)[1]
                == semantics.semantic_action_signature(view, rejected)[1]
            or semantics.semantic_action_signature(view, parent_action)[1]
                != semantics.semantic_action_signature(view, rejected)[1]
        ):
            raise CardTrainingError("preference is not a semantic disagreement")
        weight = float(row.get("preference_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise CardTrainingError("preference weight is invalid")
        research, production = _encode_twins(row["observation"], deck)
        preferences.append(CORE.PreferenceState(
            features=research,
            parent_action=parent_action,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            split=split,
            episode_id=episode_id,
            matchup="exact Grimmsnarl mirror",
            exact_mirror=True,
            preferred=preferred,
            rejected=rejected,
            weight=weight,
            families=(family,),
            production_features=production,
        ))
        split_preferences[split] += 1
        family_preferences[f"{split}:{family}"] += 1

    for row in _read_jsonl(LOCK.PRESERVATION):
        if (
            row.get("schema") != PREP.PRESERVATION_SCHEMA
            or row.get("training_kl_state") is not True
        ):
            raise CardTrainingError("preservation schema drifted")
        episode_id = int(row.get("episode_id", -1))
        prompt_index = int(row.get("prompt_index", -1))
        game = dobi_games.get(episode_id)
        key = (episode_id, prompt_index)
        if game is None or prompt_index < 0 or key in preserve_seen:
            raise CardTrainingError("preservation prompt identity is invalid")
        preserve_seen.add(key)
        split = str(row.get("supervision_split"))
        if split != game.get("supervision_split") or row.get("exact_mirror") is not True:
            raise CardTrainingError("preservation metadata disagrees with lock")
        n_options, min_count, max_count = _contract(row)
        parent_action = MAIN_PREP.validate_action_sequence(
            row.get("parent_action", ()), n_options, min_count, max_count,
        )
        research, production = _encode_twins(row["observation"], deck)
        preservation.append(CORE.PolicyState(
            features=research,
            parent_action=parent_action,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            split=split,
            episode_id=episode_id,
            matchup="exact Grimmsnarl mirror",
            exact_mirror=True,
            production_features=production,
        ))
        split_preservation[split] += 1

    expected_pref = extraction.get("preference_counts", {})
    expected_preserve = extraction.get("preservation", {}).get("counts", {})
    if (
        len(preferences) != extraction.get("preferences", {}).get("lines")
        or split_preferences["train"] != int(expected_pref.get("split_train", -1))
        or split_preferences["validation"]
            != int(expected_pref.get("split_validation", -1))
        or family_preferences != Counter({
            key: int(value)
            for key, value in extraction.get("family_counts", {}).items()
            if key.startswith("train:") or key.startswith("validation:")
        })
        or len(preservation) != extraction.get("preservation", {}).get("lines")
        or split_preservation["train"]
            != int(expected_preserve.get("split_train", -1))
        or split_preservation["validation"]
            != int(expected_preserve.get("split_validation", -1))
    ):
        raise CardTrainingError("extracted row inventory drifted")
    if not preferences or not preservation:
        raise CardTrainingError("training inventory is empty")
    return preferences, preservation


def _fixed_arms(lock: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    expected = (
        {"name": "kl1", "kl_coefficient": 1.0},
        {"name": "kl3", "kl_coefficient": 3.0},
    )
    training = lock.get("training", {})
    actual = tuple(dict(row) for row in training.get("arms", ()))
    fixed = {
        "device": "cpu", "epochs": 5, "batch_size": 64,
        "learning_rate": 1e-5, "weight_decay": 1e-5,
        "gradient_clip": 1.0, "seed": 202608071,
    }
    if actual != expected or any(training.get(key) != value for key, value in fixed.items()):
        raise CardTrainingError("locked optimizer configuration drifted")
    return actual


def _publish_arm(temporary: Path, output: Path) -> None:
    if output.exists():
        raise CardTrainingError(f"refusing existing arm output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        for name in ARM_FILENAMES:
            os.link(temporary / name, output / name)
    except BaseException:
        for name in ARM_FILENAMES:
            (output / name).unlink(missing_ok=True)
        output.rmdir()
        raise


def _train_arm(
    arm: Mapping[str, Any], parent: QM.TorchQuV2A,
    architecture: tuple[int, ...], preferences: Sequence[CORE.PreferenceState],
    preservation: Sequence[CORE.PolicyState], orders, lock: Mapping[str, Any],
) -> dict[str, Any]:
    device = torch.device("cpu")
    net = copy.deepcopy(parent).to(device)
    trainable = CORE.freeze_for_residual(net)
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    training = lock["training"]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    history: list[dict[str, Any]] = []
    for epoch, (preference_order, preservation_order) in enumerate(orders, 1):
        batch_count = math.ceil(len(preference_order) / batch_size)
        kl_chunks = CORE.preservation_chunks(preservation_order, batch_count)
        net.train()
        pair_sum = kl_sum = 0.0
        preference_seen = preservation_seen = 0
        for batch_index, begin in enumerate(range(0, len(preference_order), batch_size)):
            pref_group = [
                preferences[int(index)]
                for index in preference_order[begin:begin + batch_size]
            ]
            preserve_group = [
                preservation[int(index)] for index in kl_chunks[batch_index]
            ]
            pair = CORE._preference_loss(
                CORE._candidate_logits(net, pref_group, device),
                pref_group, device, batch_size,
            )
            kl = CORE._kl_loss(
                CORE._candidate_logits(net, preserve_group, device),
                preserve_group, device,
            )
            loss = pair + float(arm["kl_coefficient"]) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            optimizer.step()
            pair_sum += float(pair.detach()) * batch_size
            kl_sum += float(kl.detach()) * len(preserve_group)
            preference_seen += len(pref_group)
            preservation_seen += len(preserve_group)
        if preference_seen != len(preferences) or preservation_seen != len(preservation):
            raise CardTrainingError("epoch did not consume every locked row")
        row = {
            "epoch": epoch,
            "preference_examples": preference_seen,
            "preservation_examples": preservation_seen,
            "mean_pair_loss": pair_sum / preference_seen,
            "mean_parent_kl": kl_sum / preservation_seen,
        }
        history.append(row)
        print(json.dumps({"arm": arm["name"], **row}), flush=True)

    output = LOCK.RUN / "arms" / str(arm["name"])
    if output.exists():
        raise CardTrainingError(f"refusing existing arm: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    net = net.cpu().eval()
    audit = CORE.audit_frozen_tensors(parent.cpu().eval(), net)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "arm": dict(arm),
        "architecture": architecture,
        "trainable_parameters": list(trainable),
        "frozen_tensor_audit": audit,
        "state_dict": state_dict,
        "state_dict_sha256": BASE_TRAIN._state_dict_sha256(state_dict),
        "history": history,
        "promotion_authority": False,
        "upload_authority": False,
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{arm['name']}.", dir=output.parent))
    try:
        BASE_TRAIN._atomic_torch_save(checkpoint, temporary / ARM_FILENAMES[0])
        BASE_TRAIN._atomic_npz(QM.export_numpy_weights(net), temporary / ARM_FILENAMES[1])
        BASE_TRAIN._atomic_json({
            "schema": HISTORY_SCHEMA, "arm": dict(arm), "history": history,
        }, temporary / ARM_FILENAMES[2])
        _publish_arm(temporary, output)
        descriptor = {
            "name": arm["name"],
            "kl_coefficient": arm["kl_coefficient"],
            "checkpoint": str((output / ARM_FILENAMES[0]).resolve()),
            "checkpoint_sha256": LOCK.sha256_file(output / ARM_FILENAMES[0]),
            "weights": str((output / ARM_FILENAMES[1]).resolve()),
            "weights_sha256": LOCK.sha256_file(output / ARM_FILENAMES[1]),
            "history": history,
            "frozen_tensor_audit": audit,
        }
        return descriptor
    finally:
        for name in ARM_FILENAMES:
            (temporary / name).unlink(missing_ok=True)
        temporary.rmdir()


def train(lock_path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    lock = PREP.load_and_verify_lock(lock_path)
    PREP.verify_bound_artifacts(lock)
    arms = _fixed_arms(lock)
    if LOCK.TRAINING_RESULT.exists() or any(
        (LOCK.RUN / "arms" / arm["name"]).exists() for arm in arms
    ):
        raise CardTrainingError("refusing to overwrite training outcomes")
    try:
        extraction = json.loads(LOCK.EXTRACTION_RESULT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise CardTrainingError("extraction result is unavailable") from error
    body = {key: value for key, value in extraction.items() if key != "result_sha256"}
    if (
        extraction.get("schema") != PREP.RESULT_SCHEMA
        or extraction.get("result_sha256") != LOCK.canonical_sha256(body)
        or extraction.get("cohort_lock_sha256") != lock["lock_sha256"]
    ):
        raise CardTrainingError("extraction result contract failed")
    preferences, preservation = load_states(lock, extraction)
    train_preferences = [state for state in preferences if state.split == "train"]
    train_preservation = [state for state in preservation if state.split == "train"]
    if (
        {state.episode_id for state in train_preferences}
        & {state.episode_id for state in preferences if state.split == "validation"}
        or {state.episode_id for state in train_preservation}
        & {state.episode_id for state in preservation if state.split == "validation"}
    ):
        raise CardTrainingError("game split contamination detected")
    scale = CORE.normalize_preference_weights(train_preferences)
    parent, parent_runtime, architecture = _load_parent()
    CORE.attach_parent_logits(parent_runtime, train_preferences)
    CORE.attach_parent_logits(parent_runtime, train_preservation)
    orders = CORE.epoch_orders(
        len(train_preferences), len(train_preservation),
        int(lock["training"]["epochs"]), int(lock["training"]["seed"]),
    )
    descriptors = [
        # populated transactionally below
    ]
    # Every arm output was confirmed absent above, so register all paths as
    # transaction-owned before the first publication.  This closes the window
    # where _train_arm() could publish successfully and then fail while building
    # its descriptor, leaving an orphan that a rerun could neither trust nor
    # overwrite.
    owned_arms = [
        LOCK.RUN / "arms" / str(arm["name"]) for arm in arms
    ]
    try:
        for arm in arms:
            descriptor = _train_arm(
                arm, parent, architecture, train_preferences,
                train_preservation, orders, lock,
            )
            descriptors.append(descriptor)
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "cohort_lock_sha256": lock["lock_sha256"],
            "extraction_result_sha256": extraction["result_sha256"],
            "device": "cpu",
            "train_preferences": len(train_preferences),
            "validation_preferences_excluded": len(preferences) - len(train_preferences),
            "train_parent_kl_states": len(train_preservation),
            "validation_parent_kl_states_excluded": len(preservation) - len(train_preservation),
            "global_train_preference_weight_scale": scale,
            "globally_normalized_train_preference_mass": sum(
                state.weight for state in train_preferences
            ),
            "same_data_and_order_across_arms": True,
            "every_training_kl_state_used_once_per_epoch": True,
            "production_numpy_parent_authoritative": True,
            "research_production_feature_twin_audit": True,
            "arms": descriptors,
            "promotion_authority": False,
            "upload_authority": False,
        }
        result["result_sha256"] = LOCK.canonical_sha256(result)
        LOCK.write_new(LOCK.TRAINING_RESULT, result)
        return result
    except BaseException:
        # Once the complete no-clobber result exists, preserve the entire
        # transaction even if an asynchronous interruption lands while the
        # caller is returning.  Otherwise roll back only outputs owned here.
        if not LOCK.TRAINING_RESULT.exists():
            for output in reversed(owned_arms):
                for filename in ARM_FILENAMES:
                    (output / filename).unlink(missing_ok=True)
                try:
                    output.rmdir()
                except OSError:
                    pass
            try:
                (LOCK.RUN / "arms").rmdir()
            except OSError:
                pass
        raise


def main() -> int:
    try:
        result = train()
    except (OSError, TypeError, ValueError, CardTrainingError,
            CORE.EliteTrainingError, PREP.ExtractionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "arms": [row["weights_sha256"] for row in result["arms"]],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
