"""One-shot sealed-test behaviour readout for a Grim BC head.

Opened exactly once per head, after validation has already selected the
checkpoint.  It decides only whether the candidate REPLACES its Dobi-v2 parent
in the gameplay stack; it is not a strength claim and grants no packaging or
upload authority.

Pass requires, on the sealed split, all of:
  * lower weighted NLL than the parent,
  * exact-action agreement not below the parent,
  * on the subset where candidate and parent choose DIFFERENT actions, the
    candidate matches the logged expert strictly more often than the parent.
The third criterion is the one that distinguishes a real behaviour change from
a rounding-level tie: it is scored only where the two policies actually differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research.build_grim_bc_dataset import load_parent  # noqa: E402
from tools.research.train_grim_current_meta_bc import Data, head_logits  # noqa: E402


def decode_all(net, data: Data, batch: int) -> list[list[int]]:
    net.eval()
    out: list[list[int]] = []
    with torch.no_grad():
        for s in range(0, data.n, batch):
            idx = np.arange(s, min(s + batch, data.n))
            oid, otid, feat, mask = data.menu(idx)
            lg = head_logits(net, data.state[idx], oid, otid, feat, mask)
            for r, i in enumerate(idx):
                n_eng = int(data.n_options[i])
                out.append(QM.decode_sequential(
                    lg[r, :n_eng + 1].detach().numpy().astype(np.float64),
                    n_eng, int(data.min_count[i]), int(data.max_count[i])))
    return out


def weighted_nll(net, data: Data, batch: int) -> float:
    net.eval()
    tot = w_tot = 0.0
    with torch.no_grad():
        for s in range(0, data.n, batch):
            idx = np.arange(s, min(s + batch, data.n))
            oid, otid, feat, mask = data.menu(idx)
            lg = head_logits(net, data.state[idx], oid, otid, feat, mask)
            take, rows = data.steps_for(idx)
            if take is None:
                continue
            t = torch.from_numpy(take)
            m = data.step_mask[t]
            lp = torch.log_softmax(
                lg[torch.from_numpy(rows)].masked_fill(~m, -1e9), dim=-1)
            nll = -lp.gather(1, data.step_tgt[t].unsqueeze(1)).squeeze(1)
            ww = data.step_weight[t]
            tot += float((nll * ww).sum()); w_tot += float(ww.sum())
    return tot / max(w_tot, 1e-12)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--head", choices=("main", "card"), required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(8)

    data = Data(args.dataset / f"{args.head}-test.npz")
    meta = json.loads((args.dataset / f"{args.head}-test-meta.json").read_text())
    parent = load_parent(args.parent)
    cand = load_parent(args.candidate)

    p_act, c_act = decode_all(parent, data, args.batch), decode_all(cand, data, args.batch)
    p_nll, c_nll = weighted_nll(parent, data, args.batch), weighted_nll(cand, data, args.batch)

    names = meta["archetypes"]
    w = data.weight
    exp = [data.act_flat[data.act_off[i]:data.act_off[i + 1]].tolist()
           for i in range(data.n)]
    p_hit = np.array([a == b for a, b in zip(p_act, exp)], dtype=float)
    c_hit = np.array([a == b for a, b in zip(c_act, exp)], dtype=float)
    differ = np.array([a != b for a, b in zip(p_act, c_act)], dtype=bool)

    def wavg(mask, hit):
        ww = w[mask]
        return float((hit[mask] * ww).sum() / ww.sum()) if ww.sum() else 0.0

    allm = np.ones(data.n, dtype=bool)
    per = {}
    for k, name in enumerate(names):
        m = data.archetype_index == k
        if m.sum() == 0:
            continue
        per[name] = {"decisions": int(m.sum()),
                     "parent": wavg(m, p_hit), "candidate": wavg(m, c_hit),
                     "delta_pp": (wavg(m, c_hit) - wavg(m, p_hit)) * 100}

    n_dis = int(differ.sum())
    res = {
        "schema": "ptcg.grim-current-meta-bc.sealed-test.v1",
        "head": args.head,
        "dataset": str(args.dataset / f"{args.head}-test.npz"),
        "dataset_sha256": meta["dataset_sha256"],
        "decisions": data.n, "steps": int(len(data.step_dec)),
        "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "weighted_nll": {"parent": p_nll, "candidate": c_nll,
                         "delta": c_nll - p_nll},
        "exact_agreement": {"parent": wavg(allm, p_hit),
                            "candidate": wavg(allm, c_hit),
                            "delta_pp": (wavg(allm, c_hit) - wavg(allm, p_hit)) * 100},
        "disagreement_subset": {
            "decisions": n_dis,
            "share": n_dis / max(data.n, 1),
            "parent": wavg(differ, p_hit) if n_dis else 0.0,
            "candidate": wavg(differ, c_hit) if n_dis else 0.0,
        },
        "per_archetype": per,
    }
    res["checks"] = {
        "nll_improved": res["weighted_nll"]["delta"] < 0,
        "agreement_not_worse": res["exact_agreement"]["delta_pp"] >= 0,
        "wins_disagreement_subset": (n_dis > 0 and
            res["disagreement_subset"]["candidate"] >
            res["disagreement_subset"]["parent"]),
    }
    res["passed"] = all(res["checks"].values())
    body = json.dumps(res, indent=1, sort_keys=True)
    args.out.write_text(body + "\n")

    print(f"===== SEALED TEST -- {args.head.upper()} head =====")
    print(f"decisions {data.n:,}  ({meta['seats_encoded']} seats)")
    print(f"weighted NLL     parent {p_nll:.5f}  candidate {c_nll:.5f}  "
          f"delta {c_nll - p_nll:+.5f}")
    print(f"exact agreement  parent {res['exact_agreement']['parent']:.4%}  "
          f"candidate {res['exact_agreement']['candidate']:.4%}  "
          f"delta {res['exact_agreement']['delta_pp']:+.2f} pp")
    d = res["disagreement_subset"]
    print(f"disagreement subset ({d['decisions']:,} decisions, {d['share']:.1%}): "
          f"parent {d['parent']:.2%} vs candidate {d['candidate']:.2%}")
    print(f"\n{'archetype':<24}{'parent':>9}{'cand':>9}{'delta':>9}")
    for k in sorted(per, key=lambda x: -per[x]["decisions"]):
        v = per[k]
        print(f"{k:<24}{v['parent']:>9.2%}{v['candidate']:>9.2%}"
              f"{v['delta_pp']:>8.2f}p")
    print("\nchecks:")
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("SEALED BEHAVIOUR PASSED" if res["passed"] else "SEALED BEHAVIOUR FAILED")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
