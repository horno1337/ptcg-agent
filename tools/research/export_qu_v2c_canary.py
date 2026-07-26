"""Export the frozen three-member public critic ensemble to NumPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import qu_v2c_canary as CANARY  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402


PRIVATE_NAMES = (
    "zone_embedding",
    "deck_position_embedding",
    "hidden_card.weight",
    "hidden_card.bias",
    "deck_order.weight",
    "deck_order.bias",
    "hidden1.weight",
    "hidden1.bias",
    "hidden2.weight",
    "hidden2.bias",
    "hidden_to_context.weight",
    "hidden_to_context.bias",
    "q1.weight",
    "q1.bias",
    "q2.weight",
    "q2.bias",
    "q_out.weight",
    "q_out.bias",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report_path = Path(args.training_report).expanduser().resolve()
    report = json.loads(report_path.read_text())
    recorded = report.pop("report_sha256", None)
    if (
        recorded != HASH.value_sha256(report)
        or report.get("schema") != TRAIN.SCHEMA
    ):
        parser.error("training report hash/schema mismatch")
    checkpoints = report.get("final_ensemble", {}).get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != CANARY.MEMBERS:
        parser.error("training report has no three-member ensemble")
    arrays = {
        "schema": np.asarray(CANARY.SCHEMA),
        "minimum_member_advantage":
            np.asarray(CANARY.MIN_MEMBER_ADVANTAGE, dtype=np.float32),
        "members": np.asarray(CANARY.MEMBERS, dtype=np.int32),
    }
    for member, record in enumerate(checkpoints):
        path = Path(record["path"])
        if HASH.file_sha256(path) != record["sha256"]:
            parser.error(f"checkpoint hash mismatch: {path}")
        value = torch.load(path, map_location="cpu", weights_only=True)
        state = value.get("state_dict")
        if not isinstance(state, dict):
            parser.error(f"checkpoint state is malformed: {path}")
        if any(name not in state for name in PRIVATE_NAMES):
            parser.error(f"checkpoint private state is incomplete: {path}")
        for name in PRIVATE_NAMES:
            key = f"m{member}_{name.replace('.', '_')}"
            arrays[key] = (
                state[name].detach().cpu().numpy().astype(
                    np.float32, copy=True)
            )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"output exists: {output}")
    np.savez_compressed(output, **arrays)
    print(f"Exported Qu-v2C canary: {output}")
    print(f"SHA256: {HASH.file_sha256(output)}")


if __name__ == "__main__":
    main()
