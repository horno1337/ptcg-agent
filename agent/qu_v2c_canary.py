"""Torch-free conservative Qu-v2C canary layered over frozen Qu-v2B."""

from __future__ import annotations

import os

import numpy as np


SCHEMA = "ptcg.qu-v2c.public-critic-canary.v1"
MEMBERS = 3
MIN_MEMBER_ADVANTAGE = np.float32(0.10)
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qu_v2c_canary_weights.npz",
)
_ZONE_SPECS = (
    ("my_deck", 60, True),
    ("my_prize", 6, False),
    ("opponent_deck", 60, True),
    ("opponent_prize", 6, False),
    ("opponent_hand", 60, False),
    ("opponent_active", 1, False),
)
_CACHE = None


def _relu(value):
    return np.maximum(value, np.float32(0.0))


def _linear(value, arrays, member, name):
    return (
        value @ arrays[f"m{member}_{name}_weight"].T
        + arrays[f"m{member}_{name}_bias"]
    )


def _masked_mean(values, mask):
    weights = mask.astype(np.float32)[:, None]
    return (
        (values * weights).sum(axis=0)
        / np.maximum(weights.sum(), np.float32(1.0))
    )


def _masked_max(values, mask):
    if not np.any(mask):
        return np.zeros(values.shape[-1], dtype=np.float32)
    return values[mask].max(axis=0)


def _pack(source, capacity):
    values = np.asarray(source, dtype=np.int64).reshape(-1)
    values = values[(values > 0) & (values < 1300)][:capacity]
    ids = np.zeros(capacity, dtype=np.int64)
    mask = np.zeros(capacity, dtype=np.bool_)
    ids[:len(values)] = values
    mask[:len(values)] = True
    return ids, mask


def _public_zones(sample):
    sources = {
        "my_deck": sample.registered_deck_ids,
        "my_prize": sample.hand_ids,
        "opponent_deck": sample.opponent_discard_ids,
        "opponent_prize": sample.board_ids,
        "opponent_hand": sample.opponent_discard_ids,
        "opponent_active": sample.board_ids,
    }
    return {
        name: _pack(sources[name], capacity)
        for name, capacity, _ in _ZONE_SPECS
    }


def _public_context(sample, backbone):
    state = backbone._state_vector(sample)
    option_input = np.concatenate([
        sample.option_features,
        backbone.embedding[sample.option_ids],
        backbone.embedding[sample.option_target_ids],
        np.broadcast_to(state, (len(sample.option_ids), len(state))),
    ], axis=-1)
    option = _relu(backbone._linear(option_input, "option1"))
    mask = sample.option_mask.astype(np.bool_)
    option_mean = _masked_mean(option, mask)
    option_max = _masked_max(option, mask)
    contextual = np.concatenate([
        option,
        np.broadcast_to(option_mean, option.shape),
        np.broadcast_to(option_max, option.shape),
        np.broadcast_to(state, (len(option), len(state))),
    ], axis=-1)
    context = _relu(backbone._linear(contextual, "context1"))
    logits = backbone._linear(context, "policy").reshape(-1)
    logits = np.where(mask, logits, np.float32(-1e9)).astype(np.float32)
    value_hidden = _relu(backbone._linear(state, "value1"))
    value = np.float32(np.tanh(
        backbone._linear(value_hidden, "value2")[0]))
    return context, logits, value, mask


def _hidden_vector(sample, backbone, arrays, member):
    summaries = []
    ordered = []
    zones = _public_zones(sample)
    for zone_index, (name, _, has_order) in enumerate(_ZONE_SPECS):
        ids, mask = zones[name]
        cards = _relu(
            backbone.embedding[ids]
            @ arrays[f"m{member}_hidden_card_weight"].T
            + arrays[f"m{member}_hidden_card_bias"]
            + arrays[f"m{member}_zone_embedding"][zone_index]
        )
        summaries.extend([
            _masked_mean(cards, mask),
            _masked_max(cards, mask),
        ])
        if has_order:
            positions = arrays[
                f"m{member}_deck_position_embedding"][:len(ids)]
            ordered_cards = _relu(_linear(
                np.concatenate([cards, positions], axis=-1),
                arrays,
                member,
                "deck_order",
            ))
            ordered.append(_masked_mean(ordered_cards, mask))
    hidden = np.concatenate([*summaries, *ordered])
    return _relu(_linear(
        _relu(_linear(hidden, arrays, member, "hidden1")),
        arrays,
        member,
        "hidden2",
    ))


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with np.load(_PATH, allow_pickle=False) as raw:
            arrays = {name: np.array(raw[name], copy=True) for name in raw.files}
        schema = np.asarray(arrays.pop("schema"))
        threshold = np.asarray(arrays.pop("minimum_member_advantage"))
        members = np.asarray(arrays.pop("members"))
        if (
            schema.shape != ()
            or str(schema.item()) != SCHEMA
            or threshold.shape != ()
            or np.float32(threshold) != MIN_MEMBER_ADVANTAGE
            or members.shape != ()
            or int(members) != MEMBERS
            or any(
                value.dtype != np.float32
                or not np.isfinite(value).all()
                for value in arrays.values()
            )
        ):
            return None
        _CACHE = arrays
        return arrays
    except Exception:
        return None


def score_members(sample, backbone):
    """Return [member, action] public-only Q values, or ``None``."""
    arrays = _load()
    if arrays is None:
        return None
    try:
        context, logits, value, mask = _public_context(sample, backbone)
        scores = []
        for member in range(MEMBERS):
            hidden = _hidden_vector(sample, backbone, arrays, member)
            interaction = context * np.tanh(_linear(
                hidden, arrays, member, "hidden_to_context"))
            q_input = np.concatenate([
                context,
                np.broadcast_to(hidden, (len(context), len(hidden))),
                interaction,
                np.tanh(logits)[:, None],
                np.full((len(context), 1), value, dtype=np.float32),
            ], axis=-1)
            q = np.tanh(_linear(
                _relu(_linear(
                    _relu(_linear(q_input, arrays, member, "q1")),
                    arrays,
                    member,
                    "q2",
                )),
                arrays,
                member,
                "q_out",
            )).reshape(-1)
            scores.append(np.where(mask, q, np.float32(0.0)))
        result = np.stack(scores).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            return None
        return result
    except Exception:
        return None


def decide(sample, backbone, base_index, n_options):
    """Return a unanimous high-margin alternative or ``None``."""
    if (
        not isinstance(base_index, int)
        or base_index < 0
        or base_index >= n_options
        or n_options < 2
    ):
        return None
    scores = score_members(sample, backbone)
    if scores is None or scores.shape[1] < n_options:
        return None
    legal = scores[:, :n_options]
    choices = np.argmax(legal, axis=1)
    if not np.all(choices == choices[0]):
        return None
    candidate = int(choices[0])
    if candidate == base_index:
        return None
    advantages = legal[:, candidate] - legal[:, base_index]
    if np.any(advantages < MIN_MEMBER_ADVANTAGE):
        return None
    return [candidate]
