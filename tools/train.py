"""PPO self-play training for the option-scoring policy net.

Usage (inside the RL venv):
  ~/.venvs/ptcg-rl/bin/python tools/train.py --iters 50 --games 128

Collects mirror self-play games on the local engine (lockstep across N
parallel battles, batched GPU forward), updates with PPO, exports numpy
weights to agent/weights.npz after every iteration, and evals vs the
rule-based policy. The torch net here MUST mirror agent/model.py exactly;
test_roundtrip() asserts that on every export.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import Battle  # noqa: E402
from agent import cards, policy  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model as NPM  # noqa: E402
from agent.obsview import ObsView  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB = NPM.EMB
STATE_IN = FE.STATE_ID_SLOTS * EMB + 3 * EMB + FE.STATE_SCALARS
OPT_IN = FE.OPT_FEATS + EMB + 128


class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(FE.N_CARD_IDS, EMB)
        self.s1 = nn.Linear(STATE_IN, 256)
        self.s2 = nn.Linear(256, 128)
        self.v1 = nn.Linear(128, 64)
        self.v2 = nn.Linear(64, 1)
        self.o1 = nn.Linear(OPT_IN, 128)
        self.o2 = nn.Linear(128, 64)
        self.o3 = nn.Linear(64, 1)
        nn.init.normal_(self.emb.weight, std=0.05)

    def state_vec(self, ids, hand, mdisc, odisc, scalars):
        """ids [B,13] hand [B,48] mdisc/odisc [B,60] scalars [B,S]"""
        def pool(x):
            e = self.emb(x)
            m = (x > 0).float().unsqueeze(-1)
            return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

        x = torch.cat([
            self.emb(ids).flatten(1),
            pool(hand), pool(mdisc), pool(odisc), scalars,
        ], dim=1)
        h = F.relu(self.s1(x))
        return F.relu(self.s2(h))

    def value(self, sv):
        return torch.tanh(self.v2(F.relu(self.v1(sv)))).squeeze(-1)

    def logits(self, sv, opt_ids, opt_feats, mask):
        """sv [B,128] opt_ids [B,M] opt_feats [B,M,F] mask [B,M] -> [B,M]"""
        B, M, _ = opt_feats.shape
        x = torch.cat([
            opt_feats,
            self.emb(opt_ids),
            sv.unsqueeze(1).expand(B, M, sv.shape[-1]),
        ], dim=-1)
        h = F.relu(self.o1(x))
        h = F.relu(self.o2(h))
        lg = self.o3(h).squeeze(-1)
        return lg.masked_fill(~mask, -1e9)


def export_npz(net: TorchNet, path: str):
    w = {
        "emb": net.emb.weight, "s1w": net.s1.weight.T, "s1b": net.s1.bias,
        "s2w": net.s2.weight.T, "s2b": net.s2.bias,
        "v1w": net.v1.weight.T, "v1b": net.v1.bias,
        "v2w": net.v2.weight.T, "v2b": net.v2.bias,
        "o1w": net.o1.weight.T, "o1b": net.o1.bias,
        "o2w": net.o2.weight.T, "o2b": net.o2.bias,
        "o3w": net.o3.weight.T, "o3b": net.o3.bias,
    }
    np.savez(path, feat_version=FE.FEAT_VERSION,
             **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in w.items()})


# ---------------------------------------------------------------------------
# Sequential pick-with-STOP: sampling + log-prob (must match model.select_indices
# semantics at greedy temperature)
# ---------------------------------------------------------------------------

def sample_picks(logits: torch.Tensor, n_opts: int, n_min: int, n_max: int,
                 greedy: bool = False):
    """logits [n_opts+1] (last = STOP). Returns (picks, logprob, entropy)."""
    eff_max = min(n_max, n_opts) if n_max > 0 else n_opts
    picks, logp = [], torch.zeros((), device=logits.device)
    avail = torch.ones(n_opts + 1, dtype=torch.bool, device=logits.device)
    ent = None
    while len(picks) < max(eff_max, 1):
        can_stop = len(picks) >= n_min
        m = avail.clone()
        m[n_opts] = can_stop
        lg = logits.masked_fill(~m, -1e9)
        dist = torch.distributions.Categorical(logits=lg)
        if ent is None:
            ent = dist.entropy()
        a = int(lg.argmax()) if greedy else int(dist.sample())
        logp = logp + dist.log_prob(torch.tensor(a, device=logits.device))
        if a == n_opts:
            break
        picks.append(a)
        avail[a] = False
        if len(picks) >= eff_max:
            break
        if n_opts - (~avail[:n_opts]).sum().item() == 0:
            break
    return picks, logp, (ent if ent is not None else torch.zeros((), device=logits.device))


def picks_logprob(logits: torch.Tensor, picks: list[int], n_opts: int,
                  n_min: int, n_max: int):
    """Recompute logprob+entropy of a stored pick sequence under new logits."""
    eff_max = min(n_max, n_opts) if n_max > 0 else n_opts
    logp = torch.zeros((), device=logits.device)
    avail = torch.ones(n_opts + 1, dtype=torch.bool, device=logits.device)
    ent = None
    seq = list(picks)
    if len(seq) < eff_max:
        seq = seq + [n_opts]  # explicit STOP was taken
    for step, a in enumerate(seq):
        can_stop = step >= n_min
        m = avail.clone()
        m[n_opts] = can_stop
        lg = logits.masked_fill(~m, -1e9)
        dist = torch.distributions.Categorical(logits=lg)
        if ent is None:
            ent = dist.entropy()
        logp = logp + dist.log_prob(torch.tensor(a, device=logits.device))
        if a == n_opts:
            break
        avail[a] = False
    return logp, (ent if ent is not None else torch.zeros((), device=logits.device))


# ---------------------------------------------------------------------------
# Self-play collection (lockstep over parallel battles)
# ---------------------------------------------------------------------------

class Decision:
    __slots__ = ("state", "opt_ids", "opt_feats", "picks", "logp", "value",
                 "n_opts", "n_min", "n_max", "adv", "ret")

    def __init__(self, state, opt_ids, opt_feats, picks, logp, value,
                 n_opts, n_min, n_max):
        self.state, self.opt_ids, self.opt_feats = state, opt_ids, opt_feats
        self.picks, self.logp, self.value = picks, logp, value
        self.n_opts, self.n_min, self.n_max = n_opts, n_min, n_max


@torch.no_grad()
def collect(net: TorchNet, n_games: int, deck: list[int], max_selects=1200,
            temperature_greedy=False):
    """Mirror self-play; returns (decisions, stats). Rewards via GAE per
    player trajectory with gamma=1 (single terminal reward)."""
    alive = [Battle(deck, deck) for _ in range(n_games)]
    trajs = [([], []) for _ in range(n_games)]  # per game: (p0 decisions, p1)
    results = []
    steps = [0] * n_games
    done = [False] * n_games

    while not all(done):
        batch_idx, views = [], []
        for gi, b in enumerate(alive):
            if done[gi]:
                continue
            obs, sp = b.obs()
            if obs["current"]["result"] != -1 or steps[gi] >= max_selects:
                res = obs["current"]["result"] if obs["current"]["result"] != -1 else 2
                results.append(res)
                for p in (0, 1):
                    r = 0.0 if res == 2 else (1.0 if res == p else -1.0)
                    ds = trajs[gi][p]
                    for d in ds:
                        d.ret = r
                        d.adv = r - d.value
                b.close()
                done[gi] = True
                continue
            v = ObsView(obs)
            views.append((gi, sp, b, v))
        if not views:
            break

        # encode + batch
        st_list = [FE.encode_state(v) for _, _, _, v in views]
        opt_list = [FE.encode_options(v) for _, _, _, v in views]
        B = len(views)
        M = max(o[1].shape[0] for o in opt_list)
        ids = torch.from_numpy(np.stack([s["ids"] for s in st_list])).long().to(DEV)
        hand = torch.from_numpy(np.stack([s["hand_ids"] for s in st_list])).long().to(DEV)
        mdisc = torch.from_numpy(np.stack([s["my_disc"] for s in st_list])).long().to(DEV)
        odisc = torch.from_numpy(np.stack([s["opp_disc"] for s in st_list])).long().to(DEV)
        scal = torch.from_numpy(np.stack([s["scalars"] for s in st_list])).to(DEV)
        oi = torch.zeros(B, M, dtype=torch.long, device=DEV)
        of = torch.zeros(B, M, FE.OPT_FEATS, device=DEV)
        mask = torch.zeros(B, M, dtype=torch.bool, device=DEV)
        for k, (cids, feats) in enumerate(opt_list):
            m = feats.shape[0]
            oi[k, :m] = torch.from_numpy(cids.astype(np.int64))
            of[k, :m] = torch.from_numpy(feats)
            mask[k, :m] = True

        sv = net.state_vec(ids, hand, mdisc, odisc, scal)
        values = net.value(sv)
        logits = net.logits(sv, oi, of, mask)

        for k, (gi, sp, b, v) in enumerate(views):
            n_opts = opt_list[k][1].shape[0] - 1
            picks, logp, _ = sample_picks(logits[k, :n_opts + 1], n_opts,
                                          v.min_count, v.max_count,
                                          greedy=temperature_greedy)
            action = picks if picks else list(range(min(max(v.min_count, 1), n_opts)))
            err = b.select(action)
            steps[gi] += 1
            if err:
                # net produced something the engine rejects: hard loss for sp
                results.append(1 - sp)
                for p in (0, 1):
                    r = 1.0 if (1 - sp) == p else -1.0
                    for d in trajs[gi][p]:
                        d.ret, d.adv = r, r - d.value
                b.close()
                done[gi] = True
                continue
            trajs[gi][sp].append(Decision(
                st_list[k], opt_list[k][0], opt_list[k][1], picks,
                float(logp), float(values[k]), n_opts, v.min_count, v.max_count))

    decisions = [d for t in trajs for side in t for d in side]
    return decisions, results


# ---------------------------------------------------------------------------
# Behavior cloning from leaderboard episodes (tools/il_dataset.py)
# ---------------------------------------------------------------------------

def load_bc_samples(episode_dir: str) -> list["Decision"]:
    import il_dataset
    out = []
    for obs, act, reward in il_dataset.iter_dir(episode_dir):
        v = ObsView(obs)
        st = FE.encode_state(v)
        cids, feats = FE.encode_options(v)
        d = Decision(st, cids, feats, act, 0.0, 0.0,
                     feats.shape[0] - 1, v.min_count, v.max_count)
        d.ret, d.adv = reward, 0.0
        out.append(d)
    return out


def _stack_minibatch(mb):
    B = len(mb)
    M = max(d.opt_feats.shape[0] for d in mb)
    ids = torch.from_numpy(np.stack([d.state["ids"] for d in mb])).long().to(DEV)
    hand = torch.from_numpy(np.stack([d.state["hand_ids"] for d in mb])).long().to(DEV)
    mdisc = torch.from_numpy(np.stack([d.state["my_disc"] for d in mb])).long().to(DEV)
    odisc = torch.from_numpy(np.stack([d.state["opp_disc"] for d in mb])).long().to(DEV)
    scal = torch.from_numpy(np.stack([d.state["scalars"] for d in mb])).to(DEV)
    oi = torch.zeros(B, M, dtype=torch.long, device=DEV)
    of = torch.zeros(B, M, FE.OPT_FEATS, device=DEV)
    mask = torch.zeros(B, M, dtype=torch.bool, device=DEV)
    for k, d in enumerate(mb):
        m = d.opt_feats.shape[0]
        oi[k, :m] = torch.from_numpy(d.opt_ids.astype(np.int64))
        of[k, :m] = torch.from_numpy(d.opt_feats)
        mask[k, :m] = True
    return ids, hand, mdisc, odisc, scal, oi, of, mask


def bc_train(net, samples, epochs, lr, out_path, eval_every=25, eval_games=40,
             deck=None, mb_size=256, vcoef=0.5):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    idx = np.arange(len(samples))
    best = -1.0
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        tot, n = 0.0, 0
        for s in range(0, len(idx), mb_size):
            mb = [samples[i] for i in idx[s:s + mb_size]]
            ids, hand, mdisc, odisc, scal, oi, of, mask = _stack_minibatch(mb)
            sv = net.state_vec(ids, hand, mdisc, odisc, scal)
            values = net.value(sv)
            logits = net.logits(sv, oi, of, mask)
            logps = [picks_logprob(logits[k, :d.n_opts + 1], d.picks,
                                   d.n_opts, d.n_min, d.n_max)[0]
                     for k, d in enumerate(mb)]
            nll = -torch.stack(logps).mean()
            rett = torch.tensor([d.ret for d in mb], device=DEV)
            loss = nll + vcoef * F.mse_loss(values, rett)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += nll.item() * len(mb)
            n += len(mb)
        if ep % eval_every == 0 or ep == epochs:
            export_npz(net, out_path)
            wr = eval_vs_rules(out_path, eval_games, deck)
            print(f"bc epoch {ep}: nll={tot / max(n, 1):.3f} "
                  f"eval vs rules {wr:.1%}", flush=True)
            if wr >= best:
                best = wr
                torch.save(net.state_dict(),
                           os.path.join(os.path.dirname(out_path), "bc_best.pt"))
    return best


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(net, opt, decisions, epochs=2, mb_size=1024,
               clip=0.2, vcoef=0.5, ecoef=0.01):
    idx = np.arange(len(decisions))
    adv = np.array([d.adv for d in decisions], dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    for d, a in zip(decisions, adv):
        d.adv = float(a)
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), mb_size):
            mb = [decisions[i] for i in idx[s:s + mb_size]]
            B = len(mb)
            M = max(d.opt_feats.shape[0] for d in mb)
            ids = torch.from_numpy(np.stack([d.state["ids"] for d in mb])).long().to(DEV)
            hand = torch.from_numpy(np.stack([d.state["hand_ids"] for d in mb])).long().to(DEV)
            mdisc = torch.from_numpy(np.stack([d.state["my_disc"] for d in mb])).long().to(DEV)
            odisc = torch.from_numpy(np.stack([d.state["opp_disc"] for d in mb])).long().to(DEV)
            scal = torch.from_numpy(np.stack([d.state["scalars"] for d in mb])).to(DEV)
            oi = torch.zeros(B, M, dtype=torch.long, device=DEV)
            of = torch.zeros(B, M, FE.OPT_FEATS, device=DEV)
            mask = torch.zeros(B, M, dtype=torch.bool, device=DEV)
            for k, d in enumerate(mb):
                m = d.opt_feats.shape[0]
                oi[k, :m] = torch.from_numpy(d.opt_ids.astype(np.int64))
                of[k, :m] = torch.from_numpy(d.opt_feats)
                mask[k, :m] = True

            sv = net.state_vec(ids, hand, mdisc, odisc, scal)
            values = net.value(sv)
            logits = net.logits(sv, oi, of, mask)

            logps, ents = [], []
            for k, d in enumerate(mb):
                lp, en = picks_logprob(logits[k, :d.n_opts + 1], d.picks,
                                       d.n_opts, d.n_min, d.n_max)
                logps.append(lp)
                ents.append(en)
            logp = torch.stack(logps)
            ent = torch.stack(ents).mean()
            old = torch.tensor([d.logp for d in mb], device=DEV)
            advt = torch.tensor([d.adv for d in mb], device=DEV)
            rett = torch.tensor([d.ret for d in mb], device=DEV)

            ratio = (logp - old).exp()
            pl = -torch.min(ratio * advt,
                            ratio.clamp(1 - clip, 1 + clip) * advt).mean()
            vl = F.mse_loss(values, rett)
            loss = pl + vcoef * vl - ecoef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["pi"] += pl.item() * B
            stats["v"] += vl.item() * B
            stats["ent"] += ent.item() * B
            stats["n"] += B
    n = max(stats["n"], 1)
    return {k: v / n for k, v in stats.items() if k != "n"}


# ---------------------------------------------------------------------------
# Eval vs rule-based policy (numpy path, exactly like the submission runs)
# ---------------------------------------------------------------------------

def eval_vs_rules(weights_path: str, n_games: int, deck: list[int]) -> float:
    import importlib
    from agent import model as npm
    npm._cached = None
    net = npm.load(weights_path)
    assert net is not None, "exported weights failed to load"

    def net_agent(obs):
        if obs.get("select") is None:
            return list(deck)
        v = ObsView(obs)
        st = FE.encode_state(v)
        cids, feats = FE.encode_options(v)
        logits, _ = net.forward(st, cids, feats)
        picks = npm.select_indices(logits, feats.shape[0] - 1, v.min_count, v.max_count)
        return picks if picks else [0]

    def rule_agent(obs):
        if obs.get("select") is None:
            return list(deck)
        return policy.decide(obs)

    from cabt import run_battle
    wins = 0
    for g in range(n_games):
        a, b = (net_agent, rule_agent) if g % 2 == 0 else (rule_agent, net_agent)
        me = g % 2 and 1 or 0
        r = run_battle(a, b)
        if r["result"] == (0 if g % 2 == 0 else 1):
            wins += 1
    return wins / max(n_games, 1)


def test_roundtrip(net: TorchNet, deck: list[int]):
    """Torch and numpy nets must agree on a real obs."""
    path = "/tmp/rt_weights.npz"
    export_npz(net, path)
    from agent import model as npm
    npm._cached = None
    nnp = npm.load(path)
    b = Battle(deck, deck)
    obs, _ = b.obs()
    v = ObsView(obs)
    st = FE.encode_state(v)
    cids, feats = FE.encode_options(v)
    np_logits, np_val = nnp.forward(st, cids, feats)
    with torch.no_grad():
        ids = torch.from_numpy(st["ids"][None]).long().to(DEV)
        hand = torch.from_numpy(st["hand_ids"][None]).long().to(DEV)
        md = torch.from_numpy(st["my_disc"][None]).long().to(DEV)
        od = torch.from_numpy(st["opp_disc"][None]).long().to(DEV)
        sc = torch.from_numpy(st["scalars"][None]).to(DEV)
        oi = torch.from_numpy(cids[None].astype(np.int64)).to(DEV)
        of = torch.from_numpy(feats[None]).to(DEV)
        mask = torch.ones(1, feats.shape[0], dtype=torch.bool, device=DEV)
        sv = net.state_vec(ids, hand, md, od, sc)
        t_logits = net.logits(sv, oi, of, mask)[0].cpu().numpy()
        t_val = float(net.value(sv)[0])
    b.close()
    assert np.allclose(np_logits, t_logits, atol=1e-4), \
        f"logit mismatch {np.abs(np_logits - t_logits).max()}"
    assert abs(np_val - t_val) < 1e-4, "value mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "weights.npz"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--bc", default=None, help="dir of episode JSONs: behavior-clone first")
    ap.add_argument("--bc-epochs", type=int, default=150)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "tools", "checkpoints"))
    args = ap.parse_args()

    deck = policy.load_deck()
    net = TorchNet().to(DEV)
    if args.resume and os.path.exists(args.resume):
        net.load_state_dict(torch.load(args.resume, map_location=DEV))
        print(f"resumed from {args.resume}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    test_roundtrip(net, deck)
    print(f"roundtrip ok. device={DEV}, {sum(p.numel() for p in net.parameters())} params")

    ckpt = args.ckpt_dir
    os.makedirs(ckpt, exist_ok=True)

    if args.bc:
        samples = load_bc_samples(args.bc)
        print(f"bc: {len(samples)} expert decisions from {args.bc}", flush=True)
        best = bc_train(net, samples, args.bc_epochs, args.bc_lr, args.out,
                        deck=deck, eval_games=args.eval_games)
        print(f"bc done, best eval {best:.1%}", flush=True)
    for it in range(1, args.iters + 1):
        t0 = time.time()
        decisions, results = collect(net, args.games, deck)
        t1 = time.time()
        wl = {r: results.count(r) for r in set(results)}
        stats = ppo_update(net, opt, decisions)
        t2 = time.time()
        print(f"iter {it}: {len(decisions)} decisions from {len(results)} games "
              f"{wl} | pi={stats['pi']:.3f} v={stats['v']:.3f} ent={stats['ent']:.2f} "
              f"| collect {t1-t0:.0f}s update {t2-t1:.0f}s", flush=True)
        torch.save(net.state_dict(), os.path.join(ckpt, "latest.pt"))
        if it % args.eval_every == 0 or it == args.iters:
            export_npz(net, args.out)
            wr = eval_vs_rules(args.out, args.eval_games, deck)
            print(f"  eval vs rules: {wr:.1%} ({args.eval_games} games)", flush=True)
            torch.save(net.state_dict(), os.path.join(ckpt, f"iter{it:04d}.pt"))


if __name__ == "__main__":
    main()
