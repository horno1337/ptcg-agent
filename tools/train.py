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
import json
import os
import subprocess
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


class TorchNet(nn.Module):
    """Sizes are free hyperparameters; the numpy side derives every shape
    from the exported npz, so old and new weights coexist."""

    def __init__(self, emb=16, s1=256, s2=128, o1=128, o2=64):
        super().__init__()
        state_in = FE.STATE_ID_SLOTS * emb + 3 * emb + FE.STATE_SCALARS
        opt_in = FE.OPT_FEATS + emb + s2
        self.emb = nn.Embedding(FE.N_CARD_IDS, emb)
        self.s1 = nn.Linear(state_in, s1)
        self.s2 = nn.Linear(s1, s2)
        self.v1 = nn.Linear(s2, s2 // 2)
        self.v2 = nn.Linear(s2 // 2, 1)
        self.o1 = nn.Linear(opt_in, o1)
        self.o2 = nn.Linear(o1, o2)
        self.o3 = nn.Linear(o2, 1)
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
                 "n_opts", "n_min", "n_max", "adv", "ret", "weight")

    def __init__(self, state, opt_ids, opt_feats, picks, logp, value,
                 n_opts, n_min, n_max, weight=1.0):
        self.state, self.opt_ids, self.opt_feats = state, opt_ids, opt_feats
        self.picks, self.logp, self.value = picks, logp, value
        self.n_opts, self.n_min, self.n_max = n_opts, n_min, n_max
        self.weight = weight


@torch.no_grad()
def collect(net: TorchNet, n_games: int, deck: list[int], max_selects=1200,
            temperature_greedy=False, league_rules=0.0, league_random=0.0,
            deck_pool=None):
    """Self-play; returns (decisions, stats). Rewards via GAE per player
    trajectory with gamma=1 (single terminal reward). A `league_rules`
    fraction of battles seats the rule-based policy as one opponent (its
    decisions are not recorded), so the net can't overfit the mirror.

    `deck_pool` (list of decklists) turns on MULTI-DECK self-play: each game
    draws both seats from the pool, so the net learns to pilot every deck AND
    (crucially) faces a net-piloted — i.e. competent — version of the decks
    that beat us, which the rules bot never provides. None = mirror on `deck`."""
    import random as _random
    if deck_pool:
        alive = [Battle(_random.choice(deck_pool), _random.choice(deck_pool))
                 for _ in range(n_games)]
    else:
        alive = [Battle(deck, deck) for _ in range(n_games)]
    # scripted[gi] = player index piloted by a league opponent, or None.
    # First league_rules fraction gets the rule policy, next league_random
    # fraction gets random legal play (keeps the net punishing bad boards —
    # a corpus of only strong games regressed 97%->86% vs random, cycle 3).
    n_rules = int(n_games * league_rules)
    n_rand = int(n_games * league_random)
    scripted = [(gi % 2) if gi < n_rules + n_rand else None
                for gi in range(n_games)]
    from cabt import random_agent as _rand_move
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
            if scripted[gi] == sp:
                try:
                    act = (policy.decide_rules(obs) if gi < n_rules
                           else _rand_move(obs))
                except Exception:
                    act = [0]
                err = b.select(list(act))
                steps[gi] += 1
                if err:
                    results.append(1 - sp)
                    for p in (0, 1):
                        r = 1.0 if (1 - sp) == p else -1.0
                        for d in trajs[gi][p]:
                            d.ret, d.adv = r, r - d.value
                    b.close()
                    done[gi] = True
                continue
            v = ObsView(obs)
            views.append((gi, sp, b, v))
        if not views:
            continue

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
            if not picks:
                # STOP-at-zero gets substituted, never executed: re-label the
                # decision as the substitute so the update credits the action
                # that actually ran (recorded [] + executed [0] poisons PPO)
                picks = list(range(min(max(v.min_count, 1), n_opts)))
                logp, _ = picks_logprob(logits[k, :n_opts + 1], picks,
                                        n_opts, v.min_count, v.max_count)
            action = picks
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

def load_bc_samples(spec: str, w_win=1.0, w_draw=0.3,
                    w_loss=0.1) -> list["Decision"]:
    """Outcome-weighted imitation: the winning seat of every episode is the
    demonstration that matters (in our own losses, that's the opponent who
    beat us). Zero-weight samples are dropped entirely.

    `spec` is "dir[:mult][,dir[:mult]...]" — the per-dir multiplier scales
    every sample weight from that source, so band-representative data can
    be the base and scouting data seasoning (pure top-play BC = cycle3d)."""
    import il_dataset
    out = []
    for part in spec.split(","):
        d_dir, _, m = part.partition(":")
        mult = float(m) if m else 1.0
        n0 = len(out)
        for obs, act, reward in il_dataset.iter_dir(os.path.expanduser(d_dir)):
            w = (w_win if reward > 0 else (w_loss if reward < 0 else w_draw)) * mult
            if w <= 0:
                continue
            v = ObsView(obs)
            st = FE.encode_state(v)
            cids, feats = FE.encode_options(v)
            d = Decision(st, cids, feats, act, 0.0, 0.0,
                         feats.shape[0] - 1, v.min_count, v.max_count, weight=w)
            d.ret, d.adv = reward, 0.0
            out.append(d)
        print(f"  bc source {d_dir} x{mult}: {len(out) - n0} samples", flush=True)
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
            wts = torch.tensor([d.weight for d in mb], device=DEV)
            nll = -(torch.stack(logps) * wts).sum() / wts.sum().clamp(min=1e-6)
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
               clip=0.2, vcoef=0.5, ecoef=0.01,
               anchor=None, anchor_coef=0.0):
    """anchor: list of BC Decision samples; each minibatch mixes in their
    NLL so self-play can't erode the cloned expert behavior."""
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
            if anchor and anchor_coef > 0:
                amb = [anchor[i] for i in
                       np.random.randint(0, len(anchor), min(256, len(anchor)))]
                aids, ahand, amd, aod, asc, aoi, aof, amask = _stack_minibatch(amb)
                asv = net.state_vec(aids, ahand, amd, aod, asc)
                alogits = net.logits(asv, aoi, aof, amask)
                albs = [picks_logprob(alogits[k, :d.n_opts + 1], d.picks,
                                      d.n_opts, d.n_min, d.n_max)[0]
                        for k, d in enumerate(amb)]
                awts = torch.tensor([d.weight for d in amb], device=DEV)
                anll = (torch.stack(albs) * awts).sum() / awts.sum().clamp(min=1e-6)
                loss = loss - anchor_coef * anll
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
        return policy.decide_rules(obs)

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
    ap.add_argument("--bc", default=None,
                    help="episode dirs 'dir[:mult],dir2[:mult]': behavior-clone first")
    ap.add_argument("--bc-epochs", type=int, default=150)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--w-win", type=float, default=1.0)
    ap.add_argument("--w-draw", type=float, default=0.3)
    ap.add_argument("--w-loss", type=float, default=0.1)
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "tools", "checkpoints"))
    ap.add_argument("--league-rules", type=float, default=0.0,
                    help="fraction of collect battles vs the rule policy")
    ap.add_argument("--league-random", type=float, default=0.0,
                    help="fraction of collect battles vs random legal play")
    ap.add_argument("--deck-pool", default=None,
                    help="multi-deck self-play: comma-sep of 'self' (shipped "
                         "deck) or 'meta:<i>'; repeat a token to weight it. "
                         "e.g. self,self,meta:2,meta:4 = our deck + Lucario + "
                         "Grimmsnarl. Default: mirror self-play on the deck.")
    ap.add_argument("--bc-anchor", default=None,
                    help="episode dir: mix expert NLL into every PPO update")
    ap.add_argument("--bc-anchor-coef", type=float, default=0.3)
    ap.add_argument("--gate-games", type=int, default=0,
                    help="if >0: every --eval-every iters, gate the net vs the "
                         "FROZEN start-net on eval_ab pool:8 -- the real progress "
                         "signal (self-play winrate is meaningless). 0 = off.")
    ap.add_argument("--arch", default="16,256,128,128,64",
                    help="emb,s1,s2,o1,o2 layer sizes")
    args = ap.parse_args()

    deck = policy.load_deck()
    deck_pool = None
    if args.deck_pool:
        _meta = None
        deck_pool = []
        for tok in args.deck_pool.split(","):
            tok = tok.strip()
            if tok == "self":
                deck_pool.append(deck)
            elif tok.startswith("meta:"):
                if _meta is None:
                    with open(os.path.join(ROOT, "agent", "meta_decks.json")) as _f:
                        _meta = json.load(_f)
                deck_pool.append(_meta[int(tok.split(":")[1])]["deck"])
        deck_pool = deck_pool or None
        print(f"deck pool: {len(deck_pool or [])} decks from '{args.deck_pool}'", flush=True)
    net = TorchNet(*[int(x) for x in args.arch.split(",")]).to(DEV)
    if args.resume and os.path.exists(args.resume):
        net.load_state_dict(torch.load(args.resume, map_location=DEV))
        print(f"resumed from {args.resume}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    test_roundtrip(net, deck)
    print(f"roundtrip ok. device={DEV}, {sum(p.numel() for p in net.parameters())} params")

    ckpt = args.ckpt_dir
    os.makedirs(ckpt, exist_ok=True)

    if args.bc:
        samples = load_bc_samples(args.bc, args.w_win, args.w_draw, args.w_loss)
        print(f"bc: {len(samples)} expert decisions from {args.bc}", flush=True)
        best = bc_train(net, samples, args.bc_epochs, args.bc_lr, args.out,
                        deck=deck, eval_games=args.eval_games)
        print(f"bc done, best eval {best:.1%}", flush=True)
    anchor = (load_bc_samples(args.bc_anchor, args.w_win, args.w_draw, args.w_loss)
              if args.bc_anchor else None)
    if anchor:
        print(f"bc anchor: {len(anchor)} samples, coef {args.bc_anchor_coef}", flush=True)

    # Freeze the PPO start-net as the progress reference. The self-play win
    # split is always ~50% (you play yourself), so it says nothing about
    # improvement; gating vs this frozen snapshot on pool:8 is the real signal.
    gate_ref = None
    if args.gate_games > 0:
        gate_ref = os.path.join(ckpt, "ref_start.npz")
        export_npz(net, gate_ref)
        print(f"progress gate: net vs frozen start on pool:8, "
              f"{args.gate_games} games every {args.eval_every} iters", flush=True)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        decisions, results = collect(net, args.games, deck,
                                     league_rules=args.league_rules,
                                     league_random=args.league_random,
                                     deck_pool=deck_pool)
        t1 = time.time()
        wl = {r: results.count(r) for r in set(results)}
        stats = ppo_update(net, opt, decisions,
                           anchor=anchor, anchor_coef=args.bc_anchor_coef)
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
            if gate_ref:
                try:
                    r = subprocess.run(
                        [sys.executable, os.path.join(ROOT, "tools", "eval_ab.py"),
                         str(args.gate_games), args.out, "--opp", "pool:8",
                         "--base", gate_ref],
                        capture_output=True, text=True, timeout=3600)
                    line = next((l for l in r.stdout.splitlines()
                                 if l.startswith("POOL")), None)
                    print(f"  [progress vs start] "
                          f"{line or ('gate failed: ' + (r.stderr or '')[-200:])}",
                          flush=True)
                except Exception as e:
                    print(f"  [progress gate error] {e}", flush=True)


if __name__ == "__main__":
    main()
