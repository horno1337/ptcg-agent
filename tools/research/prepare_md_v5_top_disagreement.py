"""Stream the locked July 29 winner/MD-v3-disagreement preference cohort."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model, qu_v2_features as QF  # noqa: E402
from agent.obsview import ST_MAIN  # noqa: E402
from tools import analyze_ladder_replays as LADDER, il_dataset  # noqa: E402

RUN = ROOT / "tools/checkpoints/md-v5-top-disagreement-v1"
PREREG = RUN / "preregistration.json"
ARCHIVE = Path("/home/horn/Desktop/ptcg_official_2026-07-29.zip")
PARENT = ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
DATA = RUN / "preferences.jsonl.gz"
RESULT = RUN / "cohort-result.json"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def split(episode_id):
    value=hashlib.sha256(f"ptcg.md-v5.top-disagreement.v1\0{episode_id}".encode()).digest()
    return "validation" if int.from_bytes(value[:8],"big")%10==9 else "train"

def main():
    lock=json.loads(PREREG.read_text())
    if sha(ARCHIVE)!=lock["training_source"]["archive_sha256"] or sha(PARENT)!=lock["parent"]["weights_sha256"]:
        raise SystemExit("locked source or parent drifted")
    target=tuple(sorted(int(x) for x in DECK.read_text().splitlines() if x.strip()))
    parent=model.load(str(PARENT))
    if not isinstance(parent,model.QuV2Net): raise SystemExit("parent did not load")
    counts={"games_scanned":0,"eligible_games":0,"eligible_seats":0,"st_main":0,"preferences":0,"train":0,"validation":0}
    DATA.parent.mkdir(parents=True,exist_ok=True)
    if DATA.exists() or RESULT.exists(): raise SystemExit("refusing to overwrite cohort artifacts")
    digest=hashlib.sha256()
    with zipfile.ZipFile(ARCHIVE) as zf, gzip.open(DATA,"wt",encoding="utf-8") as out:
        manifest={int(r["episode_id"]):r for r in csv.DictReader(io.TextIOWrapper(zf.open("manifest.csv"),encoding="utf-8"))}
        names=sorted((n for n in zf.namelist() if n.endswith('.json') and Path(n).stem.isdigit()),key=lambda n:int(Path(n).stem))
        for ordinal,name in enumerate(names,1):
            eid=int(Path(name).stem); counts["games_scanned"]+=1
            if float(manifest[eid]["min_score"])<1050: continue
            replay=json.loads(zf.read(name)); decks=il_dataset.decks_from_document(replay)
            seats=[s for s,d in decks.items() if tuple(sorted(d))==target and float(replay["rewards"][s])>0]
            if not seats: continue
            counts["eligible_games"]+=1
            for seat in seats:
                counts["eligible_seats"]+=1
                opponent=LADDER.archetype(decks.get(1-seat,[])); part=split(eid)
                for view,logged in LADDER.action_rows(replay,seat):
                    if view.select_type!=ST_MAIN: continue
                    counts["st_main"]+=1
                    sample=QF.encode_public_observation(view.obs,target)
                    logits,_=parent.forward(sample)
                    select=view.obs["select"]
                    rejected=model.decode_qu_v2(logits,len(view.options),view.min_count,view.max_count)
                    if list(rejected)==list(logged): continue
                    row={"episode_id":eid,"seat":seat,"split":part,"min_score":float(manifest[eid]["min_score"]),"opponent":opponent,"observation":view.obs,"preferred":list(logged),"rejected":list(rejected)}
                    raw=(json.dumps(row,separators=(",",":"),ensure_ascii=False)+"\n").encode()
                    out.write(raw.decode()); digest.update(raw)
                    counts["preferences"]+=1; counts[part]+=1
            if ordinal%250==0: print(f"scanned {ordinal}/{len(names)}",file=sys.stderr,flush=True)
    payload={"schema":"ptcg.md-v5.top-disagreement-cohort.v1","preregistration_sha256":sha(PREREG),"archive_sha256":sha(ARCHIVE),"parent_weights_sha256":sha(PARENT),"counts":counts,"uncompressed_jsonl_sha256":digest.hexdigest(),"compressed_file_sha256":sha(DATA),"training_authority":False,"upload_authority":False}
    payload["result_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    RESULT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
