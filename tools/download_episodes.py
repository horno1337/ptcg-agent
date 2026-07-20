"""Bulk-download replay episodes from the Kaggle leaderboard.

Scrapes the public leaderboard for the top-N agents, lists each scoring
submission's episodes, and downloads the full replay JSON for each one --
the same {episodeId}.json format the browser "download replay" produces and
the same format ~/Desktop/ptcg_episodes/ already holds. Meant to be re-run
daily: it dedupes by episode id and skips anything already on disk (in --out
or any --skip-dir), so a rerun only fetches what's new.

Kaggle contract (reverse-engineered 2026-07-17, all under www.kaggle.com):
  leaderboard : POST /api/i/competitions.LeaderboardService/GetLeaderboard
                     {"competitionId": <id>}  -> publicLeaderboard[]
  episodes    : POST /api/i/competitions.EpisodeService/ListEpisodes
                     {"submissionId": <id>}   -> episodes[] (<=1000, newest first)
  replay      : GET  /competitions/episodes/<episodeId>/replay.json  (public)
The two POST endpoints need a session cookie + matching x-xsrf-token, seeded
by one GET of the competition page. The replay GET needs no auth.

Usage:
  python tools/download_episodes.py                       # top 50, 100 eps each -> ~/Downloads
  python tools/download_episodes.py --top 50 --per-sub 100 \
      --out ~/Desktop/ptcg_corpus_top --skip-dir ~/Desktop/ptcg_episodes
  # mid-range band: 50 agents scoring 600-800, into a separate dir
  python tools/download_episodes.py --min-score 600 --max-score 800 --top 50 \
      --out ~/Desktop/ptcg_corpus_mid --skip-dir ~/Desktop/ptcg_episodes
  python tools/download_episodes.py --dry-run             # plan only, no replays
"""

import argparse
import concurrent.futures as futures
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.kaggle.com"
LEADERBOARD = BASE + "/api/i/competitions.LeaderboardService/GetLeaderboard"
LIST_EPISODES = BASE + "/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY = BASE + "/competitions/episodes/{}/replay.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) episode-scraper/1.0"

# Stable per-competition. pokemon-tcg-ai-battle == 116727 (from the teams
# block of any ListEpisodes response); override with --competition-id.
DEFAULT_COMPETITION_ID = 116727


class Kaggle:
    """Cookie-bearing client for the internal JSON endpoints."""

    def __init__(self, competition_slug):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        # Seed session + XSRF-TOKEN cookie from the competition page.
        self._get(BASE + "/competitions/" + competition_slug)
        self.xsrf = next(
            (c.value for c in self.jar if c.name == "XSRF-TOKEN"), None
        )
        if not self.xsrf:
            sys.exit("could not obtain XSRF token (Kaggle page layout changed?)")

    def _get(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return self.opener.open(req, timeout=60).read()

    def post(self, url, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": UA,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-requested-with": "XMLHttpRequest",
                "x-xsrf-token": self.xsrf,
            },
        )
        return json.loads(self.opener.open(req, timeout=90).read())

    def leaderboard(self, competition_id):
        rows = self.post(LEADERBOARD, {"competitionId": competition_id})
        return rows.get("publicLeaderboard", [])

    def episodes_for(self, submission_id, retries=2):
        # ListEpisodes is a slow-refill token bucket (~20 calls, then HTTP 429).
        # One short retry rides out a transient blip; a genuinely drained bucket
        # fails fast so the caller can abort the attempt and let it refill during
        # a long rest, rather than burning ~2 min of backoff on every agent.
        for attempt in range(retries):
            try:
                r = self.post(LIST_EPISODES, {"submissionId": submission_id})
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(8)
                    continue
                raise
        eps = r.get("episodes", [])
        eps.sort(key=lambda e: e.get("endTime", ""), reverse=True)  # newest first
        return eps


def existing_ids(dirs):
    """Episode ids already present as <id>.json across the given dirs."""
    have = set()
    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".json"):
                stem = name[:-5]
                if stem.isdigit():
                    have.add(int(stem))
    return have


def download_replay(episode_id, out_dir, retries=3):
    """Fetch one replay to <out_dir>/<id>.json atomically. Returns bytes or None."""
    url = REPLAY.format(episode_id)
    dest = os.path.join(out_dir, "%d.json" % episode_id)
    tmp = dest + ".tmp"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(b"{"):
                raise ValueError("not JSON (episode not finished yet?)")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)  # atomic: a partial file never looks done
            return len(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # unfinished / no replay -> don't retry
                return None
            if e.code == 429:  # rate limited -> back off harder
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=50,
                    help="cap on agents to sample; the N highest-scoring that pass "
                         "the score filter (default 50)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="only agents with leaderboard score >= this (e.g. 600)")
    ap.add_argument("--max-score", type=float, default=None,
                    help="only agents with leaderboard score <= this (e.g. 800)")
    ap.add_argument("--spread", action="store_true",
                    help="sample --top agents evenly across the score band instead of "
                         "taking the highest; use for a representative band foundation")
    ap.add_argument("--per-sub", type=int, default=100,
                    help="newest episodes to take per submission (default 100)")
    ap.add_argument("--out", default="~/Downloads",
                    help="output dir for <id>.json (default ~/Downloads)")
    ap.add_argument("--skip-dir", action="append", default=[],
                    help="also treat episodes already in this dir as done (repeatable)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent replay downloads (default 4; keep it polite)")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="seconds between leaderboard/list API calls (default 0.4)")
    ap.add_argument("--competition", default="pokemon-tcg-ai-battle",
                    help="competition slug")
    ap.add_argument("--competition-id", type=int, default=DEFAULT_COMPETITION_ID)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the .done_subs.json progress file and re-list every agent")
    ap.add_argument("--abort-after", type=int, default=4,
                    help="stop the run after this many consecutive 429 list failures "
                         "(exit 3) so a supervisor can cool down and resume (default 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the plan and print counts, but download nothing")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    kg = Kaggle(args.competition)
    board = kg.leaderboard(args.competition_id)
    if not board:
        sys.exit("empty leaderboard (bad competition-id or endpoint changed?)")
    board.sort(key=lambda r: r.get("rank", 1 << 30))  # rank 1 (highest score) first

    def score_of(r):
        try:
            return float(r.get("displayScore"))
        except (TypeError, ValueError):
            return None

    sel = board
    if args.min_score is not None or args.max_score is not None:
        lo = args.min_score if args.min_score is not None else float("-inf")
        hi = args.max_score if args.max_score is not None else float("inf")
        sel = [r for r in board
               if score_of(r) is not None and lo <= score_of(r) <= hi]
        band = "%s..%s" % (args.min_score, args.max_score)
        print("leaderboard: %d teams; %d in score band %s (ranks %s..%s)"
              % (len(board), len(sel), band,
                 sel[0]["rank"] if sel else "-", sel[-1]["rank"] if sel else "-"))
    if args.spread and len(sel) > args.top:
        step = len(sel) / args.top  # evenly spaced across the band by rank
        top = [sel[int(i * step)] for i in range(args.top)]
        print("spreading %d agents evenly across the band" % len(top))
    else:
        top = sel[: args.top]
        print("taking top %d of those (highest scoring first)" % len(top))

    # Interleave listing and downloading per agent: each agent's replays hit
    # disk right after it is listed, so a crash or a 429 wall loses at most the
    # current agent and a rerun resumes for free (dedup skips what is present).
    # `have` starts as everything already on disk and doubles as the running
    # dedup set, so an episode shared by two sampled agents is fetched once.
    have = existing_ids([out_dir] + args.skip_dir)
    print("already on disk (out + skip-dirs): %d episodes" % len(have))

    # Per-agent progress state: a submission id lands here once we've listed it
    # and downloaded all its new replays with zero failures. On a rerun we skip
    # those agents' LIST calls entirely -- so a run throttled part-way (HTTP 429)
    # converges when re-invoked instead of re-hammering the rate-limited endpoint.
    state_path = os.path.join(out_dir, ".done_subs.json")
    done = set()
    if not args.refresh and os.path.exists(state_path):
        try:
            done = set(json.load(open(state_path)))
        except Exception:
            done = set()

    def mark_done(sub):
        done.add(sub)
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(done), f)
        os.replace(tmp, state_path)

    targets = [r["submissionId"] for r in top if r.get("submissionId")]

    listed = skipped = got = failed = 0
    total_bytes = 0
    consec_fail = 0
    ex = futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        for row in top:
            sub = row.get("submissionId")
            rank = row.get("rank")
            score = row.get("displayScore", "?")
            if not sub:
                continue
            if sub in done and not args.dry_run:
                skipped += 1
                continue  # already satisfied on a prior run -- no LIST call
            try:
                eps = kg.episodes_for(sub)
                consec_fail = 0
            except Exception as e:
                consec_fail += 1
                print("  rank %-4s sub %-10s  LIST FAILED: %s" % (rank, sub, e))
                if consec_fail >= args.abort_after:
                    print("  aborting attempt: %d consecutive list failures "
                          "(bucket drained) -- resume after cooldown" % consec_fail)
                    break
                continue
            listed += 1
            take = eps[: args.per_sub]
            new = [e["id"] for e in take if e["id"] not in have]
            for eid in new:
                have.add(eid)  # claim before fetching so a shared ep isn't dup'd

            sizes = ([] if args.dry_run else
                     list(ex.map(lambda eid: download_replay(eid, out_dir), new)))
            ok = sum(1 for s in sizes if s)
            got += ok
            fail_here = len(sizes) - ok
            failed += fail_here
            total_bytes += sum(s for s in sizes if s)
            if not args.dry_run and fail_here == 0:
                mark_done(sub)  # durable: survives a crash mid-run
            print("  rank %-4s sub %-10s score %-8s  %4d listed, %3d new, %3d saved"
                  "  (run total: %d eps, %.1f GB)"
                  % (rank, sub, score, len(eps), len(new), ok, got, total_bytes / 1e9))
            time.sleep(args.delay)
    finally:
        ex.shutdown(wait=True)

    remaining = [s for s in targets if s not in done]
    verb = "would download" if args.dry_run else "downloaded"
    print("\nfinished: listed %d, skipped %d done; %s %d episodes (%d failed), "
          "%.2f GB -> %s" % (listed, skipped, verb, got, failed,
                             total_bytes / 1e9, out_dir))
    if not args.dry_run and remaining:
        # signal "incomplete" so a supervisor loop knows to retry after cooldown
        print("INCOMPLETE: %d/%d agents still pending (likely 429-throttled)"
              % (len(remaining), len(targets)))
        sys.exit(3)
    if not args.dry_run:
        print("COMPLETE: all %d target agents satisfied" % len(targets))


if __name__ == "__main__":
    main()
