"""Two synthetic tasks, one backward, no optimizer step or saved model."""
import json,contextlib
from pathlib import Path
import numpy as np,torch
from types import SimpleNamespace
def execute(side,run):
 R=Path('/home/lizheng/reach_planner');N=Path(__file__).resolve().parents[1]
 cp=R/'query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt'
 if side=='old':
  from query_minco.planner import QueryMincoPlanner
  from query_minco.experiments.b2c1 import execution_revision as rev
  from query_minco.p2c2a import objective
  from query_minco.experiments.b2c1.common import state_sha
  from query_minco.p2c2c1a import teachers
  from query_minco.experiments.b2c1.teacher_exec import teacher_evaluate
  post=rev.TrajectoryPostprocessor();scope=rev.enabled
 else:
  from query_minco.models import QueryMincoPlanner
  from query_minco.trajectory import TrajectoryPostprocessor
  from query_minco.training import objective
  from query_minco.inference.identity import state_sha
  from query_minco.training.teacher import c1a as teachers
  from query_minco.training.teacher.sampling import teacher_evaluate
  post=TrajectoryPostprocessor();scope=contextlib.nullcontext
 model=QueryMincoPlanner()
 loaded=model.load_state_dict(torch.load(str(cp),map_location='cpu',weights_only=True)['state_dict'],strict=True)
 assert not loaded.missing_keys and not loaded.unexpected_keys
 model.train();initial=state_sha(model);batch=[]
 raw_teacher=torch.sin(torch.arange(126,dtype=torch.float32).reshape(1,6,21)*.13)*.04
 for name,M in [('near',1),('far',4)]:
  with np.load(N/'research_outputs/migration_v1/fixtures'/(name+'.npz'),allow_pickle=False) as z:
   d=torch.from_numpy(z['depth_input'][0].copy());s=torch.from_numpy(z['state'][0].copy())
  with torch.no_grad():o=post(raw_teacher,s[None],num_samples=32)
  batch.append(SimpleNamespace(sample_id='synthetic_'+name,depth_input=d,state=s,
   teacher_coeffs=o.trajectory.coeffs[0,:M].clone(),teacher_durations=o.parameters.durations[0,:M].clone()))
 with scope():
  loss,records,raw,out,costs=objective.forward(model,batch,'B_set','cpu')
  retained={'durations':out.parameters.durations,'inner_pts':out.parameters.inner_pts,
   'head_pva':out.parameters.head_pva,'tail_pva':out.parameters.tail_pva,'coefficients':out.trajectory.coeffs}
  for t in retained.values():
   if t.requires_grad:t.retain_grad()
  loss.backward()
 assert torch.isfinite(loss) and raw.grad is not None and torch.isfinite(raw.grad).all()
 assert torch.count_nonzero(raw.grad[:,:,19:21])==0
 for b,rec in enumerate(records):
  q=set(rec['queries']);assert len(q)==len(batch[b].teacher_coeffs)
  assert set(torch.nonzero(raw.grad[b].norm(dim=-1)>0).flatten().tolist())==q
  assert not torch.count_nonzero(raw.grad[b,[i for i in range(6) if i not in q]])
 assert state_sha(model)==initial
 arrays=dict(loss=loss.detach().numpy(),raw=raw.detach().numpy(),raw_grad=raw.grad.numpy())
 arrays.update({k:t.detach().numpy() for k,t in retained.items()})
 arrays.update({'grad_'+k:t.grad.numpy() for k,t in retained.items() if t.grad is not None})
 for i,c in enumerate(costs):
  arrays['cost_'+str(i)]=c.detach().numpy()
  arrays['matched_q_'+str(i)]=np.asarray(records[i]['queries'])
  arrays['matched_t_'+str(i)]=np.asarray(records[i]['teachers'])
  arrays['teacher_coeff_'+str(i)]=batch[i].teacher_coeffs.numpy()
  arrays['teacher_durations_'+str(i)]=batch[i].teacher_durations.numpy()
  cst=c.detach()
  v,q,t=objective.match(cst,'B_set');assert v.item()==float(cst[q,t].mean())
 for n,p in model.named_parameters():
  assert p.grad is not None and torch.isfinite(p.grad).all()
  arrays['parameter_grad_'+n]=p.grad.numpy()
 for b,s in enumerate(batch):
  d=s.teacher_durations[0];t=torch.linspace(0,1,32)[None,:]*d.sum()
  for order in range(4):arrays['teacher_sample_%d_%d'%(b,order)]=teacher_evaluate(s.teacher_coeffs[0],d,t,order).numpy()
 negative=[]
 for name,c in [('M0',torch.empty(6,0)),('M7',torch.zeros(6,7)),('nonfinite',torch.full((6,1),float('nan')))]:
  try:objective.match(c,'B_set')
  except ValueError:negative.append(name)
  else:raise AssertionError('invalid matching accepted')
 with (run/'training_arrays.npz').open('xb') as f:np.savez(f,**arrays)
 # C1A positive seed/canonical roundtrip: no search optimizer, no network.
 state=np.array([.8,0,0,0,0,0,2.8,0,0,0,0,-1],np.float32)
 scene=dict(kind='free',crossing_x_m=1.4,
  obstacles=[dict(name='remote_box',min=[8.,-2.,-2.],max=[8.2,2.,2.])],
  flight_center_domain_m=[[0,-1.65,-1.2],[7.,1.65,1.2]])
 inner,dt=teachers.seeds(state,scene,'direct')
 tail=teachers.expert_tail(state)
 teacher,receipt=teachers.canonical_teacher(state,scene,inner[0],tail,dt[0],'direct')
 if teacher is None:raise RuntimeError('synthetic canonical teacher not certified')
 with (run/'teacher_arrays.npz').open('xb') as f:
  np.savez(f,**{k:v for k,v in teacher.items() if isinstance(v,np.ndarray)},seed_inner=inner,seed_durations=dt)
 with (run/'TEACHER_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
 return dict(passed=True,side=side,M=[1,4],batch=2,loss=float(loss.detach()),matches=records,
  optimizer_updates=0,model_sha_initial=initial,model_sha_final=state_sha(model),
  score_feas_direct_gradient_zero=True,unmatched_direct_gradient_zero=True,
  negative_cases=negative,canonical_C1A_positive_fixture=True,
  R2_full_search_not_executed=True,full_cache_replay_not_executed=True)
