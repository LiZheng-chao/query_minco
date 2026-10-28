"""Frozen network VJPs from saved loss raw-gradients; no new loss formula."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import torch
from experiments.pg1_v2_r11.audit import param_grads,budget,norm,cos
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.io import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.output
start=time.monotonic();torch.set_num_threads(1)
result=read(r/'summary/COMBINED_CPU0_RESULT.json')
weights=read(r/'summary/TRAINING_ONLY_WEIGHT_SCALES.json')
ds,geo,_,_=qualify_bindings(dict(path=str(r/'_preparation/CPU0_ONLY_BINDINGS.json'),sha256=sha(r/'_preparation/CPU0_ONLY_BINDINGS.json')),42)
parent=Path(read(r/'_preparation/CONTINUATION_INPUTS.json')['parent_run']) if (r/'_preparation/CONTINUATION_INPUTS.json').exists() else None
tasks=read(r/'runs/UNIQUE_TASKS.json')['records']+ (read(parent/'runs/UNIQUE_TASKS.json')['records'] if parent else [])
taskmap={(t['model'],t['sample_id']):t for t in tasks}
joint=[];scenegrads=[];checks=[]
for mm in result['models']:
 name=mm['model'];src=parent if name=='C1_step0' and parent else r
 batches=read(src/'_preparation/COVERAGE_CLARIFICATION.json')['ordered_batches']
 model=QueryMincoPlanner();model.load_state_dict(torch.load(mm['checkpoint']['path'],map_location='cpu',weights_only=True)['state_dict']);model.eval();before=state_sha(model);assert before==mm['weights_before']
 pars=list(model.parameters())
 for bi,ids in enumerate(batches):
  budget(start,r);assert time.monotonic()-start<600,'POSTAUDIT_CPU_CAP600'
  samples=ds.load(ids);raw=model(depth_input=torch.stack([s.depth_input for s in samples]),state=torch.stack([s.state for s in samples]))
  with np.load(src/'runs'/('%s_batch%03d_RAW.npz'%(name,bi)),allow_pickle=False) as z:
   assert np.array_equal(raw.detach().numpy(),z['raw']),'RAW_FORWARD_DIFFERENT'
   gg={k[2:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('g_')}
  gv={k:param_grads(raw,gg[k],pars) for k in ('imitation','obstacle')}
  base=gg['imitation']+gg['obstacle']+gg['goal']+.1*gg['limit']+gg['domain']
  vb=param_grads(raw,base,pars)
  for safety in ('lse','tail'):
   keys=['QA','QJ','QT',safety]
   lam={k:.025/weights[k]['worst_model_p95_ratio'] for k in keys}
   gr=sum(lam[k]*gg[k] for k in keys);vn=param_grads(raw,gr,pars)
   joint.append(dict(model=name,batch=bi,terms=keys,weights=lam,weight_source='training-only worst-model P95; four equal .025 budget shares; NOT_recipe',
      base_norm=norm(vb),new_norm=norm(vn),total_norm=norm(vb+vn),
      total_physics_to_imitation=norm(vb+vn-gv['imitation'])/norm(gv['imitation']) if norm(gv['imitation']) else None,
      new_cos_obstacle=cos(vn,gv['obstacle']),new_to_imitation=norm(vn)/norm(gv['imitation']) if norm(gv['imitation']) else None))
  mins=[]
  for sid in ids:
   cc=[x for x in taskmap[name,sid]['surface_clearances_m'] if x is not None]
   mins.append(min(cc) if cc else float('inf'))
  for group,select in [('scene_bottleneck',[x<=.08 for x in mins]),('scene_wide',[x>.15 for x in mins])]:
   mask=torch.tensor(select,dtype=torch.bool)[:,None,None].expand_as(raw)
   if not mask.any():continue
   pp={k:param_grads(raw,gg[k]*mask,pars) for k in ('imitation','obstacle','goal','limit','domain','QA','QJ','QT','lse','tail')}
   for k,v in pp.items():
    scenegrads.append(dict(model=name,batch=bi,group=group,term=k,tasks=int(sum(select)),query_count=int(sum(select)*6),
       parameter_norm=norm(v),cos_imitation=cos(v,pp['imitation']),cos_obstacle=cos(v,pp['obstacle']),
       ratio_full_batch_imitation=norm(v)/norm(gv['imitation']) if norm(gv['imitation']) else None,
       reduction='full original Batch8 denominator, scene selection via min over six candidate clearances'))
  if bi%24==0:print('JOINT_AND_SCENE',name,bi,flush=True)
 assert state_sha(model)==before
 checks.append(dict(model=name,before=before,after=state_sha(model)))
write(r/'runs/POSTAUDIT_JOINT_BUDGET.json',dict(records=joint,models=checks,wall_s=time.monotonic()-start,optimizer_steps=0,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
write(r/'runs/SCENE_PARAMETER_GRADIENTS.json',dict(records=scenegrads))
print('POSTAUDIT_COMPLETE',len(joint),len(scenegrads),round(time.monotonic()-start,2))
