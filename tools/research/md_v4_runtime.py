"""Research-only layered runtime for the prospectively fixed MD-v4 candidate.

This module is deliberately outside :mod:`agent`.  It is used only by the
locked local gameplay evaluator.  The eventual submission runtime, if every
gameplay and temporal gate passes, must be separately vendored and audited as a
NumPy-only, self-contained production implementation.

Routing is intentionally narrow:

* exact target deck plus ``ST_MAIN`` uses the strict MD-v4 NumPy candidate;
* the already frozen MD-v3 ``ST_CARD`` specialist remains unchanged;
* every other ordinary route uses frozen Qu-v2B; and
* any MD-v4 scope, feature, inference, or decoding failure falls through to
  the complete frozen MD-v3 ``ST_MAIN`` path and is counted.  Gameplay gates
  require that candidate-fallback count to remain exactly zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np

from agent import md_v2_card as CARD
from agent import model, policy, qu_v2_features as QF, safety
from agent.obsview import ObsView, ST_CARD, ST_MAIN
from tools import index_corpus
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import qu_v2a_model as QM


class MDV4RuntimeError(RuntimeError):
    """A bound model artifact or research runtime contract is invalid."""


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if not archive.files:
                raise MDV4RuntimeError(f"{label} is empty")
            return {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except MDV4RuntimeError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise MDV4RuntimeError(
            f"cannot load {label} {resolved}: {error}"
        ) from error


def load_candidate_with_exact_parent(
    candidate_path: Path,
    frozen_parent_path: Path,
) -> MM.NumpyMDV4:
    """Strict-load MD-v4 and prove its embedded base equals frozen MD-v3.

    MD-v4 artifacts self-hash their embedded base, but that alone only proves
    internal consistency.  The direct gate additionally requires every
    embedded base key, dtype, shape, and value to equal the separately bound
    deployed frozen-main NPZ.
    """

    candidate_arrays = _load_npz(candidate_path, "MD-v4 candidate weights")
    parent_arrays = _load_npz(
        frozen_parent_path, "frozen MD-v3 main weights"
    )
    try:
        candidate = MM.NumpyMDV4(candidate_arrays)
        QM.NumpyQuV2A(parent_arrays)
    except (MM.MDV4ModelError, TypeError, ValueError) as error:
        raise MDV4RuntimeError(
            f"strict candidate/parent load failed: {error}"
        ) from error
    embedded = {
        name.removeprefix("base_"): value
        for name, value in candidate_arrays.items()
        if name.startswith("base_") and name != "base_weights_sha256"
    }
    if set(embedded) != set(parent_arrays):
        raise MDV4RuntimeError(
            "MD-v4 embedded-parent key set differs from frozen MD-v3"
        )
    mismatches = [
        name
        for name in sorted(parent_arrays)
        if (
            embedded[name].dtype != parent_arrays[name].dtype
            or embedded[name].shape != parent_arrays[name].shape
            or not np.array_equal(embedded[name], parent_arrays[name])
        )
    ]
    if mismatches:
        raise MDV4RuntimeError(
            "MD-v4 embedded parent differs from frozen MD-v3: "
            + ", ".join(mismatches[:8])
        )
    return candidate


def _same_action(left: Any, right: Any) -> bool:
    try:
        return list(left) == list(right)
    except (TypeError, ValueError):
        return left == right


class LayeredMDV4Controller:
    """MD-v4 exact-deck ``ST_MAIN`` over complete frozen MD-v3."""

    def __init__(
        self,
        candidate: MM.NumpyMDV4 | None,
        frozen_main: model.Net,
        frozen_card: model.Net,
        frozen_qu: model.Net,
        name: str,
        registered_deck: Sequence[int],
    ):
        if candidate is not None and not isinstance(candidate, MM.NumpyMDV4):
            raise TypeError("candidate must be None or a strict NumpyMDV4")
        for label, network in (
            ("frozen main", frozen_main),
            ("frozen card", frozen_card),
            ("frozen Qu-v2B", frozen_qu),
        ):
            if not getattr(network, "is_qu_v2", False):
                raise TypeError(f"{label} is not a Qu-v2 network")
        self.candidate = candidate
        self.frozen_main = frozen_main
        self.frozen_card = frozen_card
        self.frozen_qu = frozen_qu
        self.name = str(name)
        self.deck = tuple(int(card) for card in registered_deck)
        if len(self.deck) != 60:
            raise ValueError("registered deck must contain exactly 60 cards")

        self.calls = 0
        self.candidate_attempts = 0
        self.candidate_routes = 0
        self.candidate_fallbacks = 0
        self.parent_main_routes = 0
        self.card_routes = 0
        self.qu_routes = 0
        self.off_deck_main_routes = 0
        self.off_deck_card_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.candidate_fallback_reasons: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def _rules_failsoft(self, obs: dict, reason: str) -> list[int]:
        self.fallbacks += 1
        self.fallback_reasons[reason] += 1
        try:
            return policy.decide_rules(obs)
        except Exception as error:
            key = f"rules:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallback_reasons[key] += 1
            return safety._fallback(obs)

    @staticmethod
    def _decode_frozen(
        network: model.Net,
        sample: QF.PublicFeatures,
        view: ObsView,
    ) -> list[int]:
        logits, _ = network.forward(sample)
        return model.decode_qu_v2(
            logits,
            len(view.options),
            view.min_count,
            view.max_count,
        )

    def _frozen_route(
        self,
        obs: dict,
        view: ObsView,
        registration: tuple[int, ...],
        *,
        exact_deck: bool,
        force_parent_main: bool,
    ) -> list[int]:
        sample = QF.encode_public_observation(obs, registration)
        if force_parent_main:
            self.parent_main_routes += 1
            return self._decode_frozen(self.frozen_main, sample, view)
        if CARD.supports_view(view, registration):
            self.card_routes += 1
            return self._decode_frozen(self.frozen_card, sample, view)
        self.qu_routes += 1
        if view.select_type == ST_MAIN and not exact_deck:
            self.off_deck_main_routes += 1
        if (
            view.select_type == ST_CARD
            and not exact_deck
            and CARD.opponent_has_public_grim_signature(view)
        ):
            self.off_deck_card_routes += 1
        return self._decode_frozen(self.frozen_qu, sample, view)

    def act(
        self,
        obs: dict,
        registered_deck: Sequence[int] | None = None,
    ) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        try:
            registration = (
                self.deck
                if registered_deck is None
                else tuple(int(card) for card in registered_deck)
            )
            if safety._out_of_time(obs):
                self.fallbacks += 1
                self.fallback_reasons["panic_reserve"] += 1
                action = safety._fallback(obs)
            else:
                view = ObsView(obs)
                self.select_types[str(view.select_type)] += 1
                if not view.options:
                    action = self._rules_failsoft(
                        obs, "empty_option_menu"
                    )
                else:
                    exact_deck = MF.supports_deck(registration)
                    candidate_scope = (
                        exact_deck and view.select_type == ST_MAIN
                    )
                    if candidate_scope and self.candidate is not None:
                        self.candidate_attempts += 1
                        try:
                            features = MF.encode_runtime_observation(
                                obs, registration
                            )
                            logits, _ = self.candidate.forward(features)
                            action = QM.decode_sequential(
                                logits,
                                len(view.options),
                                view.min_count,
                                view.max_count,
                            )
                            self.candidate_routes += 1
                        except Exception as error:
                            key = type(error).__name__
                            self.candidate_fallbacks += 1
                            self.candidate_fallback_reasons[key] += 1
                            self.exceptions[f"candidate:{key}"] += 1
                            action = self._frozen_route(
                                obs,
                                view,
                                registration,
                                exact_deck=exact_deck,
                                force_parent_main=True,
                            )
                    elif candidate_scope:
                        action = self._frozen_route(
                            obs,
                            view,
                            registration,
                            exact_deck=exact_deck,
                            force_parent_main=True,
                        )
                    else:
                        action = self._frozen_route(
                            obs,
                            view,
                            registration,
                            exact_deck=exact_deck,
                            force_parent_main=False,
                        )
        except Exception as error:
            key = type(error).__name__
            self.exceptions[key] += 1
            action = self._rules_failsoft(obs, f"controller:{key}")
        try:
            repaired = safety._repair(action, obs)
            if not _same_action(repaired, action):
                self.repairs += 1
            action = repaired
        except Exception as error:
            key = f"repair:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallbacks += 1
            self.fallback_reasons[key] += 1
            action = safety._fallback(obs)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def opponent_move(self, obs: dict, rng: Any) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "overlay_enabled": self.candidate is not None,
            "calls": self.calls,
            "candidate_attempts": self.candidate_attempts,
            "candidate_routes": self.candidate_routes,
            "candidate_fallbacks": self.candidate_fallbacks,
            "candidate_fallback_reasons": dict(
                self.candidate_fallback_reasons
            ),
            "parent_main_routes": self.parent_main_routes,
            "main_routes": (
                self.candidate_routes + self.parent_main_routes
            ),
            "card_routes": self.card_routes,
            "qu_routes": self.qu_routes,
            "off_deck_main_routes": self.off_deck_main_routes,
            "off_deck_card_routes": self.off_deck_card_routes,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "exceptions": dict(self.exceptions),
            "repairs": self.repairs,
            "select_types": dict(self.select_types),
            "registered_deck_sha256": index_corpus.deck_sha256(self.deck),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": (
                    float(np.percentile(latency, 50))
                    if latency.size else 0.0
                ),
                "p95": (
                    float(np.percentile(latency, 95))
                    if latency.size else 0.0
                ),
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


__all__ = [
    "LayeredMDV4Controller",
    "MDV4RuntimeError",
    "load_candidate_with_exact_parent",
]
