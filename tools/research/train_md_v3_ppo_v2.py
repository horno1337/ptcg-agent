"""Corrected PPO mechanics for an isolated MD-v3 ST_MAIN experiment.

This module is intentionally a research library, not an experiment
preregistration or a production builder.  It keeps the deployed Qu-v2A/MD-v3
architecture, freezes the shared representation, and permits updates only to
the policy path (``option1/context1/policy``) and private value head
(``value1/value2``).

The important differences from :mod:`train_md_v3_ppo` are:

* callers create one optimizer and retain it across rollout updates;
* terminal rewards are propagated with same-episode semi-MDP GAE between
  consecutive learner ST_MAIN decisions instead of being copied to every row;
* actor and critic use separate backward passes, and critic inputs are
  detached from the frozen trunk;
* the policy remains KL-anchored to the byte-frozen MD-v3 parent.

No function in this module writes production files.  Candidate checkpoint
output is explicitly restricted away from ``agent/`` and ``decks/``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from agent import model, qu_v2_features as RUNTIME_QF
from agent.obsview import ObsView, ST_MAIN
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED
from tools.research import qu_v2a_features as TRAIN_QF
from tools.research import qu_v2a_model as QM
from tools.research import train_md_v3_ppo as V1
from tools.research import train_qu_v2a as BC
from tools.rl_env import (
    OpponentSpec,
    PTCGRLEnv,
    SelectionSpec,
    build_paired_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SCHEMA = "ptcg.md-v3.st-main-ppo-checkpoint.v2"
ACTOR_MODULES = ("option1", "context1", "policy")
CRITIC_MODULES = ("value1", "value2")
FROZEN_MODULES = (
    "embedding", "board1", "board_relation", "state1", "state2",
)


class PPOV2Error(RuntimeError):
    """The v2 research contract was violated."""


@dataclass
class Decision:
    """One learner ST_MAIN macro-decision.

    ``learner_select_index`` is the learner's primitive select count just
    before this ST_MAIN action.  ``transition_steps`` spans from this decision
    to the next learner ST_MAIN decision (or terminal), including intervening
    frozen ST_CARD/residual selections.  Opponent actions are internal to the
    single-learner semi-MDP and are not counted as learner steps.
    """

    features: TRAIN_QF.PublicFeatures
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
        return SelectionSpec(self.n_options, self.min_count, self.max_count)


@dataclass(frozen=True)
class GAETarget:
    advantage: float
    value_target: float
    td_error: float
    transition_discount: float


@dataclass(frozen=True)
class ParameterScopes:
    actor: tuple[torch.nn.Parameter, ...]
    critic: tuple[torch.nn.Parameter, ...]
    frozen: tuple[torch.nn.Parameter, ...]
    actor_names: tuple[str, ...]
    critic_names: tuple[str, ...]
    frozen_names: tuple[str, ...]


def configure_parameter_scopes(net: QM.TorchQuV2A) -> ParameterScopes:
    """Freeze the representation and expose disjoint actor/critic heads."""
    if not isinstance(net, QM.TorchQuV2A):
        raise TypeError("PPO-v2 requires the current TorchQuV2A architecture")
    actor: list[torch.nn.Parameter] = []
    critic: list[torch.nn.Parameter] = []
    frozen: list[torch.nn.Parameter] = []
    actor_names: list[str] = []
    critic_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in net.named_parameters():
        module = name.split(".", 1)[0]
        if module in ACTOR_MODULES:
            parameter.requires_grad_(True)
            actor.append(parameter)
            actor_names.append(name)
        elif module in CRITIC_MODULES:
            parameter.requires_grad_(True)
            critic.append(parameter)
            critic_names.append(name)
        elif module in FROZEN_MODULES:
            parameter.requires_grad_(False)
            frozen.append(parameter)
            frozen_names.append(name)
        else:
            raise PPOV2Error(f"unclassified Qu-v2A parameter {name!r}")
    if not actor or not critic or not frozen:
        raise PPOV2Error("Qu-v2A actor/critic/frozen scopes are incomplete")
    if set(map(id, actor)) & set(map(id, critic)):
        raise PPOV2Error("actor and critic parameter scopes overlap")
    return ParameterScopes(
        tuple(actor), tuple(critic), tuple(frozen),
        tuple(actor_names), tuple(critic_names), tuple(frozen_names),
    )


def make_optimizer(
    net: QM.TorchQuV2A,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> tuple[torch.optim.Adam, ParameterScopes]:
    """Create the single optimizer that must persist across PPO updates."""
    if (
        not math.isfinite(actor_learning_rate) or actor_learning_rate <= 0
        or not math.isfinite(critic_learning_rate) or critic_learning_rate <= 0
    ):
        raise ValueError("actor and critic learning rates must be finite and positive")
    scopes = configure_parameter_scopes(net)
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
    optimizer: torch.optim.Optimizer, scopes: ParameterScopes,
) -> None:
    groups = {str(group.get("name")): group for group in optimizer.param_groups}
    if set(groups) != {"actor", "critic"}:
        raise PPOV2Error("optimizer must contain named actor and critic groups")
    expected = {
        "actor": {id(parameter) for parameter in scopes.actor},
        "critic": {id(parameter) for parameter in scopes.critic},
    }
    for name in ("actor", "critic"):
        actual = {id(parameter) for parameter in groups[name]["params"]}
        if actual != expected[name]:
            raise PPOV2Error(f"optimizer {name} group violates parameter scope")


def compute_smdp_gae(
    decisions: Sequence[Decision],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[GAETarget]:
    """Compute episode-local TD(lambda) targets on ST_MAIN macro-transitions.

    A transition discount is ``gamma ** transition_steps``.  Lambda is applied
    once per macro-transition, which is the standard variable-duration SMDP
    recursion.  The result is returned in the same order as ``decisions``.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not decisions:
        raise PPOV2Error("cannot compute GAE for an empty rollout")
    grouped: dict[int, list[tuple[int, Decision]]] = defaultdict(list)
    for position, row in enumerate(decisions):
        if (
            isinstance(row.episode_id, bool)
            or not isinstance(row.episode_id, int)
            or isinstance(row.decision_index, bool)
            or not isinstance(row.decision_index, int)
            or row.decision_index < 0
        ):
            raise PPOV2Error("decision has invalid episode/index identity")
        grouped[row.episode_id].append((position, row))

    output: list[GAETarget | None] = [None] * len(decisions)
    for episode_id, indexed in grouped.items():
        indexed.sort(key=lambda item: item[1].decision_index)
        if [row.decision_index for _, row in indexed] != list(range(len(indexed))):
            raise PPOV2Error(
                f"episode {episode_id} ST_MAIN indices are not consecutive",
            )
        if sum(bool(row.terminal) for _, row in indexed) != 1 \
                or not indexed[-1][1].terminal:
            raise PPOV2Error(
                f"episode {episode_id} must terminate exactly on its final row",
            )
        for _, row in indexed:
            if (
                isinstance(row.transition_steps, bool)
                or not isinstance(row.transition_steps, int)
                or row.transition_steps <= 0
            ):
                raise PPOV2Error("transition_steps must be a positive integer")
            if not all(math.isfinite(value) for value in (
                row.old_logp, row.old_value, row.reward,
            )):
                raise PPOV2Error("decision contains non-finite rollout values")
            if not row.terminal and row.reward != 0.0:
                raise PPOV2Error("terminal-only reward appeared before game end")
            if row.terminal and row.reward not in (-1.0, 0.0, 1.0):
                raise PPOV2Error("terminal reward must be win/draw/loss")

        next_advantage = 0.0
        for reverse_index in range(len(indexed) - 1, -1, -1):
            position, row = indexed[reverse_index]
            discount = float(gamma ** row.transition_steps)
            next_value = (
                0.0 if row.terminal
                else float(indexed[reverse_index + 1][1].old_value)
            )
            td_error = (
                float(row.reward) + discount * next_value - float(row.old_value)
            )
            advantage = td_error
            if not row.terminal:
                advantage += discount * gae_lambda * next_advantage
            output[position] = GAETarget(
                advantage=advantage,
                value_target=float(row.old_value) + advantage,
                td_error=td_error,
                transition_discount=discount,
            )
            next_advantage = advantage
    if any(target is None for target in output):
        raise AssertionError("GAE output did not cover every decision")
    return [target for target in output if target is not None]


def _frozen_action(
    obs: dict,
    deck: Sequence[int],
    card_net: model.Net,
    qu_net: model.Net,
) -> list[int]:
    view = ObsView(obs)
    active = card_net if LAYERED.CARD.supports_view(view, deck) else qu_net
    sample = RUNTIME_QF.encode_public_observation(obs, deck)
    logits, _ = active.forward(sample)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def _finish_trajectory(
    rows: list[Decision],
    *,
    final_learner_selects: int,
    terminal_reward: float,
) -> None:
    """Attach macro durations and the sole terminal reward in place."""
    for index, row in enumerate(rows):
        next_select = (
            rows[index + 1].learner_select_index
            if index + 1 < len(rows) else int(final_learner_selects)
        )
        duration = next_select - row.learner_select_index
        if duration <= 0:
            raise PPOV2Error("learner select accounting produced a non-positive span")
        row.transition_steps = duration
        row.reward = float(terminal_reward) if index + 1 == len(rows) else 0.0
        row.terminal = index + 1 == len(rows)


def collect_games(
    net: QM.TorchQuV2A,
    frozen_main: model.Net,
    card_net: model.Net,
    qu_net: model.Net,
    deck: Sequence[int],
    games: int,
    seed: int,
    device: torch.device,
) -> tuple[list[Decision], dict[str, Any]]:
    """Collect valid, same-learner ST_MAIN trajectories versus frozen MD-v3."""
    opponent_controller = LAYERED.LayeredMirrorCardController(
        frozen_main, card_net, qu_net, "frozen-md-v3-opponent", deck,
    )
    opponents = [OpponentSpec(
        key="grimmsnarl/frozen-md-v3",
        deck=tuple(deck),
        move=opponent_controller.opponent_move,
        policy_id="frozen-md-v3",
        schedule_group="frozen-md-v3",
    )]
    schedule = build_paired_schedule(opponents, games, seed=seed)
    env = PTCGRLEnv(
        deck, opponents, observation_encoder=lambda obs: obs,
        fault_mode="truncate", max_selects=5000,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    collected: list[Decision] = []
    outcomes: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    invalid = 0
    try:
        for episode in schedule:
            trajectory: list[Decision] = []
            obs, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            reward = float(info.get("reward", 0.0))
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            while obs is not None:
                raw = env.raw_observation
                if raw is None:
                    raise PPOV2Error("environment lost its learner observation")
                view = ObsView(raw)
                learner_select_index = int(
                    info["seat_selects"][episode.learner_seat],
                )
                if view.select_type == ST_MAIN:
                    routes["st_main"] += 1
                    features = TRAIN_QF.encode_public_observation(raw, deck)
                    with torch.no_grad():
                        logits, values = net(QM.collate([features], device=device))
                        picks, logp, _ = V1.sample_selection(
                            logits[0], SelectionSpec.from_observation(raw), generator,
                        )
                    trajectory.append(Decision(
                        features=features,
                        picks=tuple(picks),
                        n_options=len(view.options),
                        min_count=view.min_count,
                        max_count=view.max_count,
                        old_logp=float(logp.cpu()),
                        old_value=float(values[0].cpu()),
                        episode_id=episode.episode_id,
                        decision_index=len(trajectory),
                        learner_select_index=learner_select_index,
                    ))
                    action = picks
                else:
                    routes["frozen_non_main"] += 1
                    action = _frozen_action(raw, deck, card_net, qu_net)
                obs, reward, terminated, truncated, info = env.step(action)
            clean_terminal = (
                terminated
                and not truncated
                and info.get("reason") == "engine_terminal"
                and info.get("agent_error") is None
                and info.get("engine_error") is None
                and info.get("result") in ("win", "draw", "loss")
            )
            if not clean_terminal:
                invalid += 1
                outcomes["invalid"] += 1
                invalid_reasons[str(info.get("reason") or "unknown")] += 1
                continue
            outcomes[str(info["result"])] += 1
            if trajectory:
                final_selects = int(
                    info["seat_selects"][episode.learner_seat],
                )
                _finish_trajectory(
                    trajectory,
                    final_learner_selects=final_selects,
                    terminal_reward=float(reward),
                )
                collected.extend(trajectory)
    finally:
        env.close()
    return collected, {
        "games": int(games),
        "outcomes": dict(outcomes),
        "invalid": invalid,
        "invalid_reasons": dict(invalid_reasons),
        "routes": dict(routes),
        "opponent": opponent_controller.diagnostics(),
    }


def _critic_values(
    net: QM.TorchQuV2A, batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run the private value head on a graph-detached frozen representation."""
    with torch.no_grad():
        state = net.state_vector(batch)
    state = state.detach()
    return torch.tanh(
        net.value2(F.relu(net.value1(state))),
    ).squeeze(-1)


def critic_step(
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    value_coefficient: float,
    max_grad_norm: float = 0.5,
) -> tuple[float, float]:
    """One critic-only step; actor parameters cannot receive a value gradient."""
    _validate_optimizer(optimizer, scopes)
    if value_coefficient < 0 or not math.isfinite(value_coefficient):
        raise ValueError("value_coefficient must be finite and non-negative")
    values = _critic_values(net, batch)
    value_loss = (values - targets).square().mean()
    optimizer.zero_grad(set_to_none=True)
    (value_coefficient * value_loss).backward()
    if any(parameter.grad is not None for parameter in scopes.actor):
        raise PPOV2Error("value loss reached an actor parameter")
    gradient = float(torch.nn.utils.clip_grad_norm_(
        scopes.critic, max_grad_norm,
    ))
    optimizer.step()
    return float(value_loss.detach().cpu()), gradient


def ppo_update(
    net: QM.TorchQuV2A,
    parent: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    decisions: Sequence[Decision],
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
) -> dict[str, float]:
    """Apply PPO with disjoint actor/critic graphs using a caller-owned optimizer."""
    if not decisions:
        raise PPOV2Error("rollout has no ST_MAIN decisions")
    if epochs <= 0 or minibatch_size <= 0:
        raise ValueError("epochs and minibatch_size must be positive")
    if not 0.0 <= clip < 1.0:
        raise ValueError("clip must be in [0, 1)")
    if any(
        not math.isfinite(value) or value < 0
        for value in (value_coefficient, entropy_coefficient, parent_kl_coefficient)
    ):
        raise ValueError("loss coefficients must be finite and non-negative")
    _validate_optimizer(optimizer, scopes)
    targets = compute_smdp_gae(
        decisions, gamma=gamma, gae_lambda=gae_lambda,
    )
    raw_advantages = torch.tensor(
        [target.advantage for target in targets], dtype=torch.float32,
    )
    advantages = (
        (raw_advantages - raw_advantages.mean())
        / raw_advantages.std(unbiased=False).clamp(min=1e-6)
    )
    value_targets = torch.tensor(
        [target.value_target for target in targets], dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    totals: Counter[str] = Counter()
    batches = 0
    net.train()
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    for _ in range(epochs):
        for indices in V1.minibatches(len(decisions), minibatch_size, generator):
            rows = [decisions[int(index)] for index in indices]
            batch = QM.collate([row.features for row in rows], device=device)

            # Actor pass. The critic output is deliberately discarded and is
            # not part of the loss graph.
            logits, _ = net(batch)
            with torch.no_grad():
                parent_logits, _ = parent(batch)
            terms = [
                V1.sequence_statistics(
                    logits[row_index, :row.n_options + 1],
                    row.picks,
                    row.spec,
                    parent_logits=parent_logits[
                        row_index, :row.n_options + 1
                    ],
                )
                for row_index, row in enumerate(rows)
            ]
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
                raise PPOV2Error("actor loss reached a critic parameter")
            actor_gradient = float(torch.nn.utils.clip_grad_norm_(
                scopes.actor, 0.5,
            ))
            optimizer.step()

            value_loss, critic_gradient = critic_step(
                net,
                optimizer,
                scopes,
                batch,
                value_targets[indices].to(device),
                value_coefficient=value_coefficient,
            )
            metrics = {
                "loss": float(actor_loss.detach().cpu())
                    + value_coefficient * value_loss,
                "actor_loss": float(actor_loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": value_loss,
                "entropy": float(entropy.mean().detach().cpu()),
                "parent_kl": float(parent_kl.mean().detach().cpu()),
                "ratio": float(ratio.mean().detach().cpu()),
                "actor_gradient_norm": actor_gradient,
                "critic_gradient_norm": critic_gradient,
            }
            totals.update(metrics)
            batches += 1
    result = {name: float(value / batches) for name, value in totals.items()}
    result.update({
        "advantage_mean_raw": float(raw_advantages.mean()),
        "advantage_std_raw": float(raw_advantages.std(unbiased=False)),
        "value_target_mean": float(value_targets.mean()),
    })
    if not all(math.isfinite(value) for value in result.values()):
        raise PPOV2Error("PPO update produced non-finite metrics")
    return result


def parameter_snapshot(
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in parameters)


def assert_parameters_unchanged(
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
    *,
    label: str,
) -> None:
    if len(parameters) != len(before) or any(
        not torch.equal(parameter.detach().cpu(), frozen)
        for parameter, frozen in zip(parameters, before)
    ):
        raise PPOV2Error(f"{label} parameters changed")


def write_candidate_checkpoint(
    output_dir: Path,
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    scopes: ParameterScopes,
    *,
    completed_updates: int,
    parent_checkpoint_sha256: str,
    provenance: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write only resumable candidate artifacts, never a runtime package."""
    output = output_dir.expanduser().resolve()
    for protected in (ROOT / "agent", ROOT / "decks"):
        try:
            output.relative_to(protected.resolve())
        except ValueError:
            continue
        raise PPOV2Error(f"candidate output cannot be written below {protected}")
    if output.exists():
        raise PPOV2Error(f"refusing to overwrite candidate directory {output}")
    output.mkdir(parents=True)
    weights_path = output / "candidate-qu-v2a-weights.npz"
    checkpoint_path = output / "candidate-qu-v2a-ppo-v2-checkpoint.pt"
    V1.atomic_npz(weights_path, QM.export_numpy_weights(net))
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in net.state_dict().items()
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "completed_updates": int(completed_updates),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "architecture": list(net.architecture),
        "state_dict": state_dict,
        "state_dict_sha256": BC._state_dict_sha256(state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainable_parameter_names": {
            "actor": list(scopes.actor_names),
            "critic": list(scopes.critic_names),
        },
        "frozen_parameter_names": list(scopes.frozen_names),
        "provenance": dict(provenance),
    }
    V1.atomic_torch(checkpoint_path, checkpoint)
    return weights_path, checkpoint_path
