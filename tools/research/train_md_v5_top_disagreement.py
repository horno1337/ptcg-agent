"""Train the three locked July 29 MD-v3-disagreement preference arms."""
from __future__ import annotations
from dataclasses import dataclass
import copy,gzip,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research import qu_v2a_features as QF, qu_v2a_model as QM  # noqa:E402
from tools.research import train_qu_v2a as TRAIN  # noqa:E402

RUN=ROOT/'tools/checkpoints/md-v5-top-disagreement-v1'
LOCK=RUN/'preregistration.json'; COHORT=RUN/'cohort-result.json'; DATA=RUN/'preferences.jsonl.gz'
PARENT_CKPT=ROOT/'tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-checkpoint.pt'
PARENT_NPZ=ROOT/'tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz'
DECK=tuple(sorted(int(x) for x in (ROOT/'decks/md_v1_grimmsnarl.csv').read_text().splitlines() if x.strip()))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

@dataclass
class Pair:
 features: object; preferred:tuple[int,...]; rejected:tuple[int,...]; n_opts:int; n_min:int; n_max:int; parent_logits:np.ndarray; split:str

def seq(logits,picks,n_opts,n_min,n_max,parent=None):
 eff=min(n_max,n_opts) if n_max>0 else n_opts; sequence=list(picks)
 if len(sequence)<eff:sequence.append(n_opts)
 available=torch.ones(n_opts+1,dtype=torch.bool,device=logits.device); lp=logits.new_zeros(()); kl=logits.new_zeros(())
 pt=None if parent is None else torch.as_tensor(parent,dtype=logits.dtype,device=logits.device)
 for step,a in enumerate(sequence):
  legal=available.clone();legal[n_opts]=step>=n_min
  sl=F.log_softmax(logits[:n_opts+1].masked_fill(~legal,-1e9),0);lp=lp+sl[a]
  if pt is not None:
   pl=F.log_softmax(pt.masked_fill(~legal,-1e9),0);kl=kl+(pl.exp()*(pl-sl)).sum()
  if a==n_opts:break
  available[a]=False
 return lp,kl

def load_parent():
 payload=torch.load(PARENT_CKPT,map_location='cpu',weights_only=True);arch=tuple(payload['architecture'])
 net=QM.TorchQuV2A(*arch);net.load_state_dict(payload['state_dict'],strict=True);net.eval()
 with np.load(PARENT_NPZ,allow_pickle=False) as z: numpy=QM.NumpyQuV2A({k:np.array(z[k],copy=True) for k in z.files})
 return net,numpy,arch

def load_pairs(numpy):
 rows=[]
 with gzip.open(DATA,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);obs=r['observation'];s=QF.encode_public_observation(obs,DECK);pl,_=numpy.forward(s);sel=obs['select']
   rows.append(Pair(s,tuple(r['preferred']),tuple(r['rejected']),len(sel['option']),int(sel.get('minCount',0)),int(sel.get('maxCount',0)),np.asarray(pl,dtype=np.float32),r['split']))
 return rows

def metrics(net,pairs,device,batch_size=128):
 net.eval();pref=rej=other=0;losses=[];kls=[]
 with torch.no_grad():
  for start in range(0,len(pairs),batch_size):
   group=pairs[start:start+batch_size];logits,_=net(QM.collate([p.features for p in group],device=device))
   for i,p in enumerate(group):
    lp,k=seq(logits[i],p.preferred,p.n_opts,p.n_min,p.n_max,p.parent_logits);lr,_=seq(logits[i],p.rejected,p.n_opts,p.n_min,p.n_max)
    pp,_=seq(torch.as_tensor(p.parent_logits),p.preferred,p.n_opts,p.n_min,p.n_max);pr,_=seq(torch.as_tensor(p.parent_logits),p.rejected,p.n_opts,p.n_min,p.n_max)
    losses.append(float(F.softplus(-((lp-lr)-(pp-pr)))));kls.append(float(k))
    chosen=tuple(QM.decode_sequential(logits[i,:p.n_opts+1].cpu().numpy(),p.n_opts,p.n_min,p.n_max))
    if chosen==p.preferred:pref+=1
    elif chosen==p.rejected:rej+=1
    else:other+=1
 return {'examples':len(pairs),'preferred':pref,'rejected':rej,'other':other,'conditional_preferred':pref/(pref+rej) if pref+rej else 0.0,'pair_loss':float(np.mean(losses)),'parent_kl':float(np.mean(kls))}

def main():
 lock=json.loads(LOCK.read_text());cohort=json.loads(COHORT.read_text())
 if sha(DATA)!=cohort['compressed_file_sha256'] or sha(PARENT_CKPT)!=lock['parent']['checkpoint_sha256']:raise SystemExit('locked artifact drift')
 parent,numpy,arch=load_parent();pairs=load_pairs(numpy);train=[p for p in pairs if p.split=='train'];val=[p for p in pairs if p.split=='validation']
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');parent=parent.to(device)
 outputs=[]
 for arm_index,arm in enumerate(lock['training']['arms']):
  net=copy.deepcopy(parent);net.train()
  for p in list(net.value1.parameters())+list(net.value2.parameters()):p.requires_grad=False
  params=[p for p in net.parameters() if p.requires_grad]
  opt=torch.optim.AdamW(params,lr=lock['training']['learning_rate'],weight_decay=lock['training']['weight_decay'])
  history=[];rng=np.random.default_rng(lock['training']['seed']+arm_index)
  for epoch in range(1,lock['training']['epochs']+1):
   order=rng.permutation(len(train));s_loss=s_kl=0.0;seen=0;net.train()
   for begin in range(0,len(order),lock['training']['batch_size']):
    group=[train[int(i)] for i in order[begin:begin+lock['training']['batch_size']]]
    logits,_=net(QM.collate([p.features for p in group],device=device));terms=[];kterms=[]
    for i,p in enumerate(group):
     lp,k=seq(logits[i],p.preferred,p.n_opts,p.n_min,p.n_max,p.parent_logits);lr,_=seq(logits[i],p.rejected,p.n_opts,p.n_min,p.n_max)
     pp,_=seq(torch.as_tensor(p.parent_logits,device=device),p.preferred,p.n_opts,p.n_min,p.n_max);pr,_=seq(torch.as_tensor(p.parent_logits,device=device),p.rejected,p.n_opts,p.n_min,p.n_max)
     terms.append(F.softplus(-((lp-lr)-(pp-pr))));kterms.append(k)
    dpo=torch.stack(terms).mean();kl=torch.stack(kterms).mean();loss=dpo+float(arm['kl_coefficient'])*kl
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,lock['training']['gradient_clip']);opt.step()
    s_loss+=float(dpo.detach())*len(group);s_kl+=float(kl.detach())*len(group);seen+=len(group)
   vm=metrics(net,val,device);row={'epoch':epoch,'train_pair_loss':s_loss/seen,'train_parent_kl':s_kl/seen,'validation':vm};history.append(row);print(json.dumps({'arm':arm['name'],**row}),flush=True)
  arrays=QM.export_numpy_weights(net.cpu());out=RUN/arm['name'];out.mkdir(exist_ok=True);np.savez_compressed(out/'candidate-qu-v2a-weights.npz',**arrays)
  torch.save({'schema':'ptcg.md-v5.top-disagreement-arm.v1','arm':arm,'architecture':arch,'state_dict':net.state_dict(),'history':history},out/'checkpoint.pt')
  outputs.append({'arm':arm,'weights_sha256':sha(out/'candidate-qu-v2a-weights.npz'),'history':history})
 result={'schema':'ptcg.md-v5.top-disagreement-training-result.v1','device':str(device),'cohort_result_sha256':cohort['result_sha256'],'arms':outputs,'promotion_authority':False,'upload_authority':False}
 result['result_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();(RUN/'training-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'result_sha256':result['result_sha256'],'arms':[x['weights_sha256'] for x in outputs]},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
