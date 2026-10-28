"""Frozen checkpoint export, no optimizer; dev8 is a diagnostic set."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
from query_minco.training.io import read,write,sha,artifact,require,digest
from query_minco.training.pg1.state_io import inspect_commit
from query_minco.training.pg1_v2_r12.spec import validate_checkpoint_blob
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg1_v2_r12.evaluation import evaluate
from query_minco.training.pg1_v2_r12.objective import tail_from_samples
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.training.pilot_evaluation import load_rows,_certified_empty
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss
from query_minco.inference.identity import state_sha
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bindings',type=Path);p.add_argument('--bindings-sha256');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
s=read(a.run/'RUN_SPEC.json');binding=s['bindings']
if a.bindings:
 require(a.bindings_sha256 is not None,'R12_EVAL_BINDING_SHA')
 binding=dict(path=str(a.bindings),sha256=a.bindings_sha256)
 old=read(artifact(s['bindings']));new=read(artifact(binding))
 require({k:v for k,v in old.items() if k!='code'}=={k:v for k,v in new.items() if k!='code'},'R12_EVAL_SAME_ASSETS')
if (a.run/'RUN_SEAL.json').exists():
 require(all(sha(a.run/n)==h for n,h in read(a.run/'RUN_SEAL.json')['files'].items()),'R12_EVAL_RUN_SEAL')
ds,geo,_,_=qualify_bindings(binding,42);dev=read(artifact(binding))['development']
root_spec=read(a.run/'ROOT_SPEC.json')
require(read(a.run/'RESULT.json')['completed_optimizer_updates']==32,'R12_EVAL_FULL_BUDGET')
require(sorted(p.name for p in (a.run/'commits').glob('step[0-9]*'))==['step%04d'%i for i in range(33)],'R12_EVAL_COMMIT_SEQUENCE')
previous=None
for i in range(33):
 commit=a.run/'commits'/('step%04d'%i);info=inspect_commit(commit,root_spec)
 require(info['step']==i and info['parent_hash']==previous,'R12_EVAL_COMMIT_CHAIN')
 previous=sha(commit/'COMMIT.json')
require(sorted(p.name for p in (a.run/'checkpoints').glob('*.pt'))==['checkpoint_step%04d.pt'%i for i in [0,8,16,32]],'R12_EVAL_CHECKPOINT_SET')
with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('R12_EVAL_NO_OPTIMIZER')):
 result=[]
 for step in [0,8,16,32]:
  cp=a.run/'checkpoints'/('checkpoint_step%04d.pt'%step);rc=read(str(cp)+'.json');require(sha(cp)==rc['sha256'],'R12_EVAL_CHECKPOINT_SHA')
  blob=torch.load(cp,map_location='cpu',weights_only=True)
  validate_checkpoint_blob(blob,root_spec,step,rc['weights_sha256'])
  require(sha(cp)==sha(a.run/'commits'/('step%04d'%step)/cp.name),'R12_EVAL_CHECKPOINT_COMMIT_ALIAS')
  model=QueryMincoPlanner();model.load_state_dict(blob['state_dict'],strict=True);model.eval();before=state_sha(model)
  require(before==rc['weights_sha256'],'R12_EVAL_WEIGHT_ID')
  ev=evaluate(model,dev)
  for detail in ev['r12_candidate_details']:
   detail['minimum_safety_margin_m']=detail['clearance_m']-.03 if detail['clearance_m'] is not None else None
  gr=[]
  for sample,meta in load_rows(dev):
   raw=model(depth_input=sample.depth_input[None],state=sample.state[None]);out=TrajectoryPostprocessor()(raw,sample.state[None]);cost,_=cost_matrix(out,sample,0);im,qq,tt=match(cost,'B_set')
   boxes=[raw.new_tensor([[b['min'],b['max']] for b in meta['scene']['obstacles']]).reshape(-1,2,3)]
   p,v,w=segment_samples(out.trajectory,64)
   obs,_=physical_loss(p,v,w,boxes,geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
   tail=tail_from_samples(p,w,boxes,[_certified_empty(meta['scene'])])
   pp=list(model.parameters());gv={}
   for name,value in [('imitation',im),('obstacle',obs.mean()),('tail',tail.mean())]:
    gg=torch.autograd.grad(value,pp,retain_graph=True,allow_unused=True)
    g=torch.cat([(torch.zeros_like(p) if x is None else x).detach().flatten() for x,p in zip(gg,pp)])
    gv[name]=g
   gr.append(dict(id=sample.sample_id,parameter_norms={k:float(x.norm()) for k,x in gv.items()},
    cosine_tail_imitation=float(gv['tail'].dot(gv['imitation'])/(gv['tail'].norm()*gv['imitation'].norm())) if gv['tail'].norm()*gv['imitation'].norm()>0 else None,
    cosine_tail_obstacle=float(gv['tail'].dot(gv['obstacle'])/(gv['tail'].norm()*gv['obstacle'].norm())) if gv['tail'].norm()*gv['obstacle'].norm()>0 else None))
  require(state_sha(model)==before,'R12_EVAL_WEIGHT_MUTATION')
  write(a.output/('step%04d.json'%step),dict(checkpoint_sha256=sha(cp),weights_sha256=before,evaluation=ev,gradients=gr,optimizer_steps=0,independent_final_test=False))
  result.append(dict(step=step,physics_rate=ev['physics_rate'],oracle_rate=ev['oracle_rate'],goal_rate=ev['goal_rate']))
 write(a.output/'SUMMARY.json',dict(run=str(a.run),records=result,method=s['method'],initialization='frozen C1step2880',bindings=binding,optimizer_steps=0,scope='READ_ONLY_MECHANISM_DIAGNOSTICS'))
 print('R12_FROZEN_EXPORT_COMPLETE')
