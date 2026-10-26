"""Unique Frozen384 supervision activation; zero updates. Full run is user-executed."""
import time
from pathlib import Path
import numpy as np,torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,write,artifact,require,sha
from .spec import qualify_bindings
from .objective import forward

def batch_stats(model,samples,geo):
 z=forward(model,samples,geo,'query_minco_pg1_task_complete_v1');params=list(model.parameters())
 near=[np.linalg.norm(s.state[6:9].numpy().astype(float))<=3.4 for s in samples]
 groups={'all':list(range(len(samples)))}
 for name,ids in [('near',[i for i,v in enumerate(near) if v]),('far',[i for i,v in enumerate(near) if not v])]+[('M%d'%m,[i for i,s in enumerate(samples) if len(s.teacher_coeffs)==m]) for m in (1,4)]:
  if ids:groups[name]=ids
 output={}
 for name,ids in groups.items():
  terms={k:v[ids].mean() for k,v in z['per_query'].items()}
  terms['imitation']=torch.stack([z['costs'][i][z['matches'][i]['queries'],z['matches'][i]['teachers']].mean() for i in ids]).mean()
  pg={k:torch.cat([x.detach().flatten() for x in torch.autograd.grad(v,params,retain_graph=True)]) for k,v in terms.items()}
  phys=pg['obstacle']+pg['goal']+.1*pg['dynamic']+pg['domain'];p=float(phys.norm());im=float(pg['imitation'].norm())
  output[name]=dict(samples=len(ids),queries=len(ids)*6,
   losses={k:float(v.detach()) for k,v in terms.items()},
   activation={k:dict(exact_nonzero=int((v[ids].detach()>0).sum()),above_1e_minus12=int((v[ids].detach()>1e-12).sum())) for k,v in z['per_query'].items()},
   parameter_gradient_norms={k:float(v.norm()) for k,v in pg.items()},
   physical_imitation_ratio=p/im if im else None,
   gradient_cosine=float((phys*pg['imitation']).sum())/(p*im) if p and im else None,
   gradient_conflict=bool((phys*pg['imitation']).sum()<0) if p and im else None)
 require(all(torch.isfinite(g).all() for g in pg.values()),'ACTIVATION_FINITE_GRAD')
 return output

def run(binding,out,max_batches=48,wall_s=3600):
 require(type(max_batches) is int and 1<=max_batches<=48,'ACTIVATION_BATCH_BOUND')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();allseeds={}
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for seed in (42,43,44):
  ds,geo,init,q=qualify_bindings(binding,seed);m=QueryMincoPlanner()
  m.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'],strict=True);m.train();before=state_sha(m)
  ids=sorted(ds.rows);seen=[];rows=[]
  for b in range(max_batches):
   require(time.monotonic()-start<wall_s,'ACTIVATION_WALL_LIMIT')
   selected=ids[b*8:(b+1)*8];stats=batch_stats(m,ds.load(selected),geo);seen+=selected
   write(out/('A%d_batch%02d.json'%(seed,b)),dict(seed=seed,ids=selected,stats=stats))
   rows.append(stats);print('ACTIVATION',seed,b+1,'/',max_batches,flush=True)
  require(state_sha(m)==before==init['weights_sha256'],'ACTIVATION_UPDATE_FORBIDDEN')
  aggregate={}
  for group in ('all','near','far','M1','M4'):
   values=[r[group] for r in rows if group in r];qcount=sum(v['queries'] for v in values)
   aggregate[group]=dict(samples=sum(v['samples'] for v in values),queries=qcount,
    activation={k:dict(exact_nonzero=sum(v['activation'][k]['exact_nonzero'] for v in values),
      fraction=sum(v['activation'][k]['exact_nonzero'] for v in values)/qcount if qcount else None,
      above_1e_minus12=sum(v['activation'][k]['above_1e_minus12'] for v in values),fraction_above_1e_minus12=sum(v['activation'][k]['above_1e_minus12'] for v in values)/qcount if qcount else None) for k in ('goal','dynamic','domain','obstacle')},
    batch_gradient_norms={k:[v['parameter_gradient_norms'][k] for v in values] for k in ('imitation','goal','dynamic','domain','obstacle')},
    batch_ratios=[v['physical_imitation_ratio'] for v in values],batch_conflicts=[v['gradient_conflict'] for v in values])
  allseeds[str(seed)]=dict(covered_ids=seen,unique_tasks=len(set(seen)),strata=aggregate,weights_sha256=before)
  write(out/('A%d_SUMMARY.json'%seed),allseeds[str(seed)])
 complete=max_batches==48 and all(s['unique_tasks']==384 for s in allseeds.values())
 result=dict(schema='pg1_activation384_v1',complete=complete,status='COMPLETE' if complete else 'INCOMPLETE_SMOKE',
  bindings=binding,code=read(artifact(binding))['code'],seeds=allseeds,optimizer_steps=0,
  sampling='sorted unique384 train IDs, each once per seed; DOES NOT change training FrozenOrder',
  low_new_signal_risk=any(all((s['strata']['all']['activation'][k]['fraction_above_1e_minus12'] or 0)<.01 for k in ('goal','dynamic','domain')) for s in allseeds.values()),
  diagnostic_activation_threshold=1e-12,low_signal_warning_fraction=.01,note='Gradient norms are per group/batch, not norm of a dataset-mean gradient; cosine/conflicts report denominators. No formula or weights adjusted.',
  wall_s=time.monotonic()-start,files={p.name:sha(p) for p in out.glob('*.json')})
 write(out/'ACTIVATION_RESULT.json',result);return result
