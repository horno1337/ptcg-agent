"""Train one head-isolated Grimmsnarl BC challenger from frozen Dobi-v2.

The parent is the exact packaged Dobi-v2 head.  Only ``option1``, ``context1``
and ``policy`` are trainable; the embedding, board/state trunk and value heads
stay frozen and byte-identical, which is why the dataset ships precomputed
state vectors.

Supervision is the runtime's own sequential no-replacement decode, expanded
into steps: a k-pick action becomes k masked softmax steps (plus a STOP step
when STOP is legal).  That is mathematically the same objective as scoring the
whole sequence, but it batches, so a 250k-decision epoch is minutes rather
than hours.

The trust region is KL(parent || candidate) at the same states, so the
candidate cannot drift far from a field-proven policy on states it will
actually face.  Selection uses validation only; the sealed test split is
opened once, afterwards, by the separate behaviour readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research.build_grim_bc_dataset import load_parent  # noqa: E402

TRAINABLE = ("option1", "context1", "policy")


class Data:
    """Dataset plus its precomputed sequential-decode step expansion."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.state = torch.from_numpy(z["state"])
        self.opt_off = z["option_offsets"]
        self.option_ids = z["option_ids"]
        self.option_target_ids = z["option_target_ids"]
        self.option_features = z["option_features"]
        self.n_options = z["n_options"]
        self.weight = z["weight"]
        self.act_off = z["action_offsets"]
        self.act_flat = z["action_flat"]
        self.min_count = z["min_count"]
        self.max_count = z["max_count"]
        self.archetype_index = z["archetype_index"]
        self.episode_id = z["episode_id"]
        self.n = len(self.n_options)
        self.max_menu = int(max(self.opt_off[i + 1] - self.opt_off[i]
                                for i in range(self.n)))
        self._build_steps()

    def _build_steps(self) -> None:
        dec, tgt, masks = [], [], []
        for i in range(self.n):
            n_eng = int(self.n_options[i])
            stop = n_eng
            picks = self.act_flat[self.act_off[i]:self.act_off[i + 1]].tolist()
            eff_min = min(int(self.min_count[i]), n_eng)
            mx = int(self.max_count[i])
            eff_max = min(mx, n_eng) if mx > 0 else n_eng
            avail = np.ones(n_eng + 1, dtype=bool)
            path = list(picks) + ([stop] if len(picks) < eff_max else [])
            for step, action in enumerate(path):
                legal = avail.copy()
                if step >= eff_max:
                    legal[:n_eng] = False
                legal[stop] = step >= eff_min
                if not legal.any() or not legal[action]:
                    break
                # A forced step carries no information and no gradient.
                if legal.sum() > 1:
                    row = np.zeros(self.max_menu, dtype=bool)
                    row[:n_eng + 1] = legal
                    masks.append(row); dec.append(i); tgt.append(action)
                if action == stop:
                    break
                avail[action] = False
        dec_arr = np.asarray(dec, dtype=np.int64)
        # Steps are emitted in increasing decision order, so a CSR pointer
        # turns per-batch step selection from O(all steps) into O(batch steps).
        counts = np.bincount(dec_arr, minlength=self.n)
        self.step_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.step_dec = torch.from_numpy(dec_arr)
        self.step_tgt = torch.from_numpy(np.asarray(tgt, dtype=np.int64))
        self.step_mask = torch.from_numpy(np.stack(masks)) if masks else \
            torch.zeros((0, self.max_menu), dtype=torch.bool)
        w = torch.from_numpy(self.weight.astype(np.float32))
        self.step_weight = w[self.step_dec] if len(dec_arr) else w[:0]

    def steps_for(self, idx: np.ndarray):
        """(step indices, row-in-batch) for a batch of decision indices."""
        counts = self.step_ptr[idx + 1] - self.step_ptr[idx]
        if counts.sum() == 0:
            return None, None
        take = np.concatenate([np.arange(self.step_ptr[i], self.step_ptr[i + 1])
                               for i in idx if self.step_ptr[i + 1] > self.step_ptr[i]])
        rows = np.repeat(np.arange(len(idx)), counts)
        return take, rows

    def menu(self, idx: np.ndarray):
        """Padded option tensors for a batch of decision indices."""
        b = len(idx)
        oid = np.zeros((b, self.max_menu), dtype=np.int64)
        otid = np.zeros((b, self.max_menu), dtype=np.int64)
        feat = np.zeros((b, self.max_menu, self.option_features.shape[1]),
                        dtype=np.float32)
        mask = np.zeros((b, self.max_menu), dtype=bool)
        for r, i in enumerate(idx):
            a, z = self.opt_off[i], self.opt_off[i + 1]
            k = z - a
            oid[r, :k] = self.option_ids[a:z]
            otid[r, :k] = self.option_target_ids[a:z]
            feat[r, :k] = self.option_features[a:z]
            mask[r, :k] = True
        return (torch.from_numpy(oid), torch.from_numpy(otid),
                torch.from_numpy(feat), torch.from_numpy(mask))


def head_logits(net: QM.TorchQuV2A, state, oid, otid, feat, mask):
    """Trainable path only: the frozen trunk already produced `state`."""
    option_input = torch.cat(
        [feat, net.embedding(oid), net.embedding(otid),
         state.unsqueeze(1).expand(-1, feat.shape[1], -1)], dim=-1)
    option = F.relu(net.option1(option_input))
    w = mask.unsqueeze(-1).to(option.dtype)
    mean = (option * w).sum(1) / w.sum(1).clamp(min=1.0)
    mx = option.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
    mx = torch.nan_to_num(mx, neginf=0.0)
    ctx = torch.cat([option, mean.unsqueeze(1).expand_as(option),
                     mx.unsqueeze(1).expand_as(option),
                     state.unsqueeze(1).expand(-1, option.shape[1], -1)], dim=-1)
    logits = net.policy(F.relu(net.context1(ctx))).squeeze(-1)
    return logits.masked_fill(~mask, -1e9)


def evaluate(net, data: Data, batch: int) -> dict:
    """Weighted NLL and exact-action agreement under the runtime decode."""
    net.eval()
    tot_w = tot_nll = 0.0
    agree_w = agree_tot = 0.0
    per_arche: dict[int, list[float]] = {}
    with torch.no_grad():
        for s in range(0, data.n, batch):
            idx = np.arange(s, min(s + batch, data.n))
            oid, otid, feat, mask = data.menu(idx)
            lg = head_logits(net, data.state[idx], oid, otid, feat, mask)
            take, rows = data.steps_for(idx)
            if take is not None:
                take_t = torch.from_numpy(take)
                m = data.step_mask[take_t]
                lp = torch.log_softmax(
                    lg[torch.from_numpy(rows)].masked_fill(~m, -1e9), dim=-1)
                nll = -lp.gather(1, data.step_tgt[take_t].unsqueeze(1)).squeeze(1)
                w = data.step_weight[take_t]
                tot_nll += float((nll * w).sum()); tot_w += float(w.sum())
            # exact greedy action agreement, decision level
            for r, i in enumerate(idx):
                n_eng = int(data.n_options[i])
                got = QM.decode_sequential(
                    lg[r, :n_eng + 1].detach().numpy().astype(np.float64),
                    n_eng, int(data.min_count[i]), int(data.max_count[i]))
                want = data.act_flat[data.act_off[i]:data.act_off[i + 1]].tolist()
                ww = float(data.weight[i])
                hit = ww if got == want else 0.0
                agree_w += hit; agree_tot += ww
                per_arche.setdefault(int(data.archetype_index[i]), [0.0, 0.0])
                per_arche[int(data.archetype_index[i])][0] += hit
                per_arche[int(data.archetype_index[i])][1] += ww
    return {"weighted_nll": tot_nll / max(tot_w, 1e-12),
            "exact_agreement": agree_w / max(agree_tot, 1e-12),
            "per_archetype": {k: v[0] / max(v[1], 1e-12)
                              for k, v in per_arche.items()},
            "steps": int(len(data.step_dec)), "decisions": data.n}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--head", choices=("main", "card"), required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    train = Data(args.dataset / f"{args.head}-train.npz")
    valid = Data(args.dataset / f"{args.head}-validation.npz")
    print(f"train {train.n:,} decisions / {len(train.step_dec):,} steps   "
          f"valid {valid.n:,} / {len(valid.step_dec):,}", flush=True)

    net = load_parent(args.parent)
    frozen = load_parent(args.parent)
    for q in frozen.parameters():
        q.requires_grad_(False)
    trainable = []
    for name, param in net.named_parameters():
        if name.split(".")[0] in TRAINABLE:
            param.requires_grad_(True); trainable.append(name)
        else:
            param.requires_grad_(False)
    print("trainable:", trainable, flush=True)
    frozen_snapshot = {k: v.detach().clone() for k, v in net.state_dict().items()
                       if k.split(".")[0] not in TRAINABLE}

    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad],
                           lr=args.lr)
    base = evaluate(frozen, valid, args.batch)
    print(f"parent  valid NLL {base['weighted_nll']:.5f}  "
          f"agreement {base['exact_agreement']:.4%}", flush=True)

    history, best, best_state = [], None, None
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(train.n)
        t0, run_nll, run_kl, seen = time.time(), 0.0, 0.0, 0
        for s in range(0, train.n, args.batch):
            idx = np.sort(order[s:s + args.batch])
            oid, otid, feat, mask = train.menu(idx)
            state = train.state[idx]
            lg = head_logits(net, state, oid, otid, feat, mask)
            with torch.no_grad():
                pl = head_logits(frozen, state, oid, otid, feat, mask)
            take, rows = train.steps_for(idx)
            if take is None:
                continue
            take_t = torch.from_numpy(take)
            d = torch.from_numpy(rows)
            m = train.step_mask[take_t]
            w = train.step_weight[take_t]
            cand = torch.log_softmax(lg[d].masked_fill(~m, -1e9), dim=-1)
            par = torch.log_softmax(pl[d].masked_fill(~m, -1e9), dim=-1)
            nll = -cand.gather(1, train.step_tgt[take_t].unsqueeze(1)).squeeze(1)
            kl = (par.exp() * (par - cand)).masked_fill(~m, 0.0).sum(-1)
            wn = w / w.sum().clamp(min=1e-12)
            loss = (nll * wn).sum() + args.kl * (kl * wn).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [q for q in net.parameters() if q.requires_grad], 5.0)
            opt.step()
            run_nll += float((nll * wn).sum()); run_kl += float((kl * wn).sum())
            seen += 1
            if seen % 100 == 0:
                print(f"  e{epoch} {s+len(idx)}/{train.n} "
                      f"nll {run_nll/seen:.4f} kl {run_kl/seen:.5f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        # frozen tensors must be bit-identical after every epoch
        for k, v in frozen_snapshot.items():
            if not torch.equal(net.state_dict()[k], v):
                raise SystemExit(f"frozen tensor {k} changed during training")
        ev = evaluate(net, valid, args.batch)
        row = {"epoch": epoch, "train_nll": run_nll / max(seen, 1),
               "train_kl": run_kl / max(seen, 1), "seconds": time.time() - t0,
               **{k: v for k, v in ev.items() if k != "per_archetype"}}
        history.append(row)
        print(f"epoch {epoch}: valid NLL {ev['weighted_nll']:.5f} "
              f"(parent {base['weighted_nll']:.5f})  agreement "
              f"{ev['exact_agreement']:.4%} (parent {base['exact_agreement']:.4%})  "
              f"{row['seconds']:.0f}s", flush=True)
        if best is None or ev["weighted_nll"] < best["weighted_nll"]:
            best = {**ev, "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    arrays = QM.export_numpy_weights(net.eval())
    out_npz = args.out / f"candidate-{args.head}-weights.npz"
    np.savez(out_npz, **arrays)
    sha = hashlib.sha256(out_npz.read_bytes()).hexdigest()
    result = {
        "schema": "ptcg.grim-current-meta-bc.training-result.v1",
        "head": args.head, "seed": args.seed, "lr": args.lr, "kl": args.kl,
        "epochs": args.epochs, "batch": args.batch,
        "trainable_parameters": trainable,
        "parent_npz": str(args.parent),
        "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        "dataset_dir": str(args.dataset),
        "parent_validation": {k: v for k, v in base.items() if k != "per_archetype"},
        "parent_validation_per_archetype": base["per_archetype"],
        "selected_epoch": best["epoch"],
        "candidate_validation": {k: v for k, v in best.items()
                                 if k != "per_archetype"},
        "candidate_validation_per_archetype": best["per_archetype"],
        "delta_nll": best["weighted_nll"] - base["weighted_nll"],
        "delta_agreement": best["exact_agreement"] - base["exact_agreement"],
        "history": history,
        "candidate_weights": str(out_npz), "candidate_sha256": sha,
    }
    (args.out / f"training-result-{args.head}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"\nselected epoch {best['epoch']}  "
          f"NLL {base['weighted_nll']:.5f} -> {best['weighted_nll']:.5f} "
          f"({result['delta_nll']:+.5f})   agreement "
          f"{base['exact_agreement']:.4%} -> {best['exact_agreement']:.4%} "
          f"({result['delta_agreement']*100:+.2f} pp)")
    print(f"candidate {out_npz}  sha256 {sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
