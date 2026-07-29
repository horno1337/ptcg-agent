import json
from pathlib import Path

import numpy as np

from tools.research import md_v3_ppo_v2_population as POP


def _weights(path: Path):
    # Population construction is integration-tested against real artifacts;
    # focused unit tests cover the independent field/mass contract.
    path.write_bytes(b"placeholder")


def test_field_rows_excludes_grim_and_preserves_positive_rows(tmp_path):
    path = tmp_path / "field.json"
    path.write_text(json.dumps({
        "schema": "field-v1",
        "field": [
            {"archetype": "Grimmsnarl", "deck": [1] * 60,
             "field_weight": 0.6},
            {"archetype": "Alakazam", "deck": [2] * 60,
             "field_weight": 0.25},
            {"archetype": "Mewtwo", "deck": [3] * 60,
             "field_weight": 0.15},
        ],
    }))
    payload, rows = POP._field_rows(path)
    assert payload["schema"] == "field-v1"
    assert [row["archetype"] for row in rows] == ["Alakazam", "Mewtwo"]
    assert np.isclose(sum(row["field_weight"] for row in rows), 0.4)


def test_pilot_mass_is_normalized_and_split_evenly_by_scope():
    assert np.isclose(sum(POP.PILOT_MASS.values()), 1.0)
    mirror = sum(
        weight for key, weight in POP.PILOT_MASS.items()
        if key.startswith("mirror_")
    )
    field = sum(
        weight for key, weight in POP.PILOT_MASS.items()
        if key.startswith("field_")
    )
    assert np.isclose(mirror, 0.5)
    assert np.isclose(field, 0.5)
