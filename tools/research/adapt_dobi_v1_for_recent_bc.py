"""Re-containerize frozen Dobi-v1 for the strict conservative BC loader."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import train_qu_v2a as TRAIN  # noqa: E402


SOURCE = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt"
)
OUTPUT = ROOT / "tools/checkpoints/dobi-v1-bc-recent-v1/dobi-v1-bc-adapter.pt"


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    state = source.get("state_dict")
    if (
        source.get("completed_updates") != 130
        or not isinstance(state, dict)
        or source.get("state_dict_sha256") != TRAIN._state_dict_sha256(state)
    ):
        raise SystemExit("Dobi-v1 terminal checkpoint contract failed")
    payload = {
        "schema": TRAIN.TRAINING_SCHEMA,
        "candidate_only": True,
        "feature_schema": TRAIN.QF.SCHEMA,
        "feature_dependency_fingerprint": TRAIN._feature_contract_fingerprint(),
        "model_schema": TRAIN.QM.MODEL_SCHEMA,
        "model_implementation_sha256": TRAIN._model_implementation_sha256(),
        "architecture": tuple(source["architecture"]),
        "state_dict": state,
        "state_dict_sha256": source["state_dict_sha256"],
        "adapter_provenance": {
            "source_path": str(SOURCE.relative_to(ROOT)),
            "source_sha256": TRAIN._sha256_file(SOURCE),
            "source_schema": source["schema"],
            "source_completed_updates": source["completed_updates"],
            "state_modified": False,
        },
    }
    TRAIN._atomic_torch_save(payload, OUTPUT)
    verify = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    if (
        verify["state_dict_sha256"] != source["state_dict_sha256"]
        or any(
            not torch.equal(state[name], verify["state_dict"][name])
            for name in state
        )
    ):
        raise SystemExit("BC adapter changed the Dobi-v1 terminal state")
    print(json.dumps({
        "output": str(OUTPUT),
        "sha256": TRAIN._sha256_file(OUTPUT),
        "state_dict_sha256": verify["state_dict_sha256"],
        "state_modified": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
