"""Apply the locked July 30 qualification rule to all three fixed arms."""
from __future__ import annotations
from collections import Counter
import csv,hashlib,io,json
from pathlib import Path
import sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from agent import model,qu_v2_features as QF  # noqa:E402
from agent.obsview import ST_MAIN  # noqa:E402
from tools import analyze_ladder_replays as LADDER,il_dataset  # noqa:E402
RUN=ROOT/'tools/checkpoints/md-v5-top-disagreement-v1';LOCK=RUN/'preregistration.json';TRAINING=RUN/'training-result.json'
ARCHIVE=Path('/home/horn/Desktop/ptcg_official_2026-07-30.zip');PARENT=ROOT/'tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz';DECK=tuple(sorted(int(x) for x in (ROOT/'decks/md_v1_grimmsnarl.csv').read_text().splitlines() if x.strip()))
OUT=RUN/'july30-qualification.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def act(net,sample,select):
 logits,_=net.forward(sample);return model.decode_qu_v2(logits,len(select['option']),int(select.get('minCount',0)),int(select.get('maxCount',0)))
def add(c,logged,parent,candidate):
 c['disagreements']+=1
 if list(logged)==list(candidate):c['logged_candidate']+=1
 elif list(logged)==list(parent):c['logged_parent']+=1
 else:c['logged_other']+=1
def main():
 lock=json.loads(LOCK.read_text());training=json.loads(TRAINING.read_text());parent=model.load(str(PARENT));arms=[]
 for row in training['arms']:
  name=row['arm']['name'];path=RUN/name/'candidate-qu-v2a-weights.npz'
  if sha(path)!=row['weights_sha256']:raise SystemExit('arm drift')
  arms.append((name,model.load(str(path)),Counter(),Counter(),Counter()))
 with zipfile.ZipFile(ARCHIVE) as zf:
  manifest={int(r['episode_id']):r for r in csv.DictReader(io.TextIOWrapper(zf.open('manifest.csv'),encoding='utf-8'))}
  names=sorted((n for n in zf.namelist() if n.endswith('.json') and Path(n).stem.isdigit()),key=lambda n:int(Path(n).stem))
  for ordinal,name in enumerate(names,1):
   eid=int(Path(name).stem);replay=json.loads(zf.read(name));decks=il_dataset.decks_from_document(replay);seats=[s for s,d in decks.items() if tuple(sorted(d))==DECK]
   if not seats:continue
   high=float(manifest[eid]['min_score'])>=1100
   for seat in seats:
    win=float(replay['rewards'][seat])>0;mirror=(1-seat) in seats
    for view,logged in LADDER.action_rows(replay,seat):
     if view.select_type!=ST_MAIN:continue
     select=view.obs['select'];sample=QF.encode_public_observation(view.obs,DECK);pa=act(parent,sample,select)
     for _,net,total,primary,mirrorwin in arms:
      total['main_prompts']+=1;ca=act(net,sample,select)
      if ca==pa:continue
      add(total,logged,pa,ca)
      if win and high:add(primary,logged,pa,ca)
      if win and mirror:add(mirrorwin,logged,pa,ca)
   if ordinal%250==0:print(f'scanned {ordinal}/{len(names)}',file=sys.stderr,flush=True)
 rule=lock['selection_on_untouched_july30'];results=[]
 for name,_,total,primary,mirrorwin in arms:
  change=total['disagreements']/total['main_prompts'];pd=primary['logged_candidate']+primary['logged_parent'];md=mirrorwin['logged_candidate']+mirrorwin['logged_parent'];pc=primary['logged_candidate']/pd if pd else 0;mc=mirrorwin['logged_candidate']/md if md else 0
  passed=rule['minimum_change_rate']<=change<=rule['maximum_change_rate'] and primary['disagreements']>=rule['minimum_primary_disagreements'] and pc>=rule['minimum_primary_conditional_logged_agreement'] and mc>=rule['minimum_mirror_winner_conditional_logged_agreement']
  results.append({'arm':name,'total':dict(total),'primary_high_rating_winners':dict(primary),'mirror_winners':dict(mirrorwin),'change_rate':change,'primary_conditional_logged_agreement':pc,'mirror_winner_conditional_logged_agreement':mc,'passed':passed})
 qualified=[r for r in results if r['passed']];kl={a['name']:a['kl_coefficient'] for a in lock['training']['arms']};qualified.sort(key=lambda r:(-r['primary_conditional_logged_agreement'],-kl[r['arm']]))
 result={'schema':'ptcg.md-v5.top-disagreement-july30-qualification.v1','training_result_sha256':training['result_sha256'],'july30_archive_sha256':sha(ARCHIVE),'arms':results,'selected_arm':qualified[0]['arm'] if qualified else None,'direct_gameplay_permitted':bool(qualified),'promotion_authority':False,'upload_authority':False}
 result['result_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
