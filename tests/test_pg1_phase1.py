"""G1 synthetic CPU only. G2 intentionally NOT_QUALIFIED by user instruction."""
import copy,json,math,os,subprocess,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from test_certified_empty_p0 import Fixtures,RawModel,fixture_raw,ROOT,binding,dump,code_binding
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.trajectory.types import MincoTrajectory
from query_minco.training.pilot_worker import batch as e2_batch
from query_minco.training.pilot_spec import recipe as e2_recipe,IDS
from query_minco.training.io import digest
from query_minco.training.datasets.geometry_sidecar import geometry_digest
from query_minco.training.pg1.losses import *
from query_minco.training.pg1.geometry import authenticated_task_gates
from query_minco.training.pg1.objective import cpu_zero_update,forward
from query_minco.training.pg1.audit import audit_cpu,numpy_samples,numpy_terms,strict_diagnostics
from query_minco.training.pg1.spec import *
PG1='query_minco_pg1_task_complete_v1'
OUT=Path(os.environ.get('PG1_TEST_OUTPUT') or tempfile.mkdtemp(prefix='pg1_cpu_tests_'))
(OUT/'audit').mkdir(parents=True,exist_ok=True)

class PG1Phase1(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.n=0
  self.guard=patch('torch.optim.AdamW.step',side_effect=AssertionError('PG1 optimizer.step prohibited'));self.guard.start()
 def tearDown(self):self.guard.stop();self.tmp.cleanup()
 def fixture(self,kinds=('free','obstacle'),**kwargs):
  self.n+=1;return Fixtures(self.root/str(self.n),list(kinds),**kwargs)
 def refresh(self,f,index,scene=None):
  mp=f.metadata[index];meta=json.loads(mp.read_text());r=f.rows[index]
  if scene is not None:
   meta['scene']=scene;r.update(obstacles=scene['obstacles'],flight_center_domain_m=scene['flight_center_domain_m'],scene_id=geometry_digest(scene),empty_certified=not scene['obstacles'])
   cp=Path(r['contract_source']['path']);contract=json.loads(cp.read_text());contract['scene']=scene;dump(cp,contract);r['contract_source']=binding(cp)
  dump(mp,meta);r['scene_source']=binding(mp);f.original[index]['metadata_sha256']=binding(mp)['sha256']
  dump(f.anchor,dict(rows=f.original))
  for row in f.rows:row['anchor']['artifact']=binding(f.anchor)
  dump(f.manifest,dict(scope='SYNTHETIC_ROW_AUTH_ONLY',rows=f.rows))
  # Authenticate actual sample tensors again, including M0 synthetic fixtures.
  from test_certified_empty_p0 import RowSidecar
  f.geo=RowSidecar(f.manifest,f.samples,f.rows)
 def blocked(self,M=1,covers=True,goal=1.):
  f=self.fixture(('obstacle',),goal=goal)
  scene=json.loads(f.metadata[0].read_text())['scene'];scene['kind']='blocked'
  scene['obstacles']=[dict(name='separating_slab',min=[.35,-3.,-3.],max=[.65,3. if covers else .1,3.])]
  if M==0:
   f.samples[0].teacher_coeffs=f.samples[0].teacher_coeffs[:0];f.samples[0].teacher_durations=f.samples[0].teacher_durations[:0]
  elif M==4:
   s=f.samples[0];s.teacher_coeffs=s.teacher_coeffs.repeat(4,1,1,1);s.teacher_durations=s.teacher_durations.repeat(4,1)
  self.refresh(f,0,scene)
  return f
 def check(self,result,f):
  json.dumps(result['record'],allow_nan=False)
  audit=audit_cpu(result['record'],result['arrays'],f.samples,f.geo);json.dumps(audit,allow_nan=False)
  self.assertTrue(audit['passed'])
  return audit
 def test_analytic_pvaj_matches_numpy_and_original_sampler(self):
  from query_minco.training.losses.set_physical_risk import segment_samples
  f=self.fixture();m=RawModel();out=TrajectoryPostprocessor()(m.raw[None].expand(2,-1,-1),torch.stack([s.state for s in f.samples]))
  v=analytic_samples(out.trajectory);p,vel,w=segment_samples(out.trajectory,64)
  for key,x in [('p',p),('v',vel),('weights',w)]:self.assertTrue(torch.equal(v[key],x))
  n=numpy_samples(out.trajectory.coeffs.detach().numpy(),out.parameters.durations.detach().numpy())
  for key in n:self.assertTrue(np.allclose(v[key].detach().numpy(),n[key],rtol=5e-4,atol=5e-4),key)
 def test_goal_near_far_boundaries_match_original_metrics(self):
  from query_minco.evaluation.goal import task_metrics
  for dtype in [torch.float32,torch.float64]:
   for distance in [3.39999,3.4,3.40001,5.8]:
    goal=torch.tensor([[distance,0,0]],dtype=dtype);end=goal[:,None].expand(1,6,3).clone();v=torch.zeros_like(end)
    _,diag=goal_task_loss(end,v,goal)
    hard=task_metrics(end[0,0].numpy(),v[0,0].numpy(),goal[0].numpy())
    self.assertEqual(bool(diag['near'][0]),hard['mode']=='near_arrival')
 def test_goal_numpy_and_zero_goal_rejection(self):
  f=self.fixture(goal=5.8);m=RawModel();r=cpu_zero_update(m,f.samples,f.geo,PG1);self.check(r,f)
  with self.assertRaises(ValueError):goal_task_loss(torch.ones(1,6,3),torch.zeros(1,6,3),torch.zeros(1,3))
 def test_full_methods_free_obstacle_mixed_M1_M4_and_strict_json(self):
  for kinds in [('free','free'),('obstacle','obstacle'),('free','obstacle')]:
   f=self.fixture(kinds)
   for method in METHODS:
    with self.subTest(kinds=kinds,method=method):self.check(cpu_zero_update(RawModel(),f.samples,f.geo,method),f)
 def test_paired_e2_exact_scalar_gradients(self):
  f=self.fixture();m=RawModel();cfg=e2_recipe('E2')
  spec=dict(code=code_binding(),device='cpu',recipe=cfg,recipe_sha256=digest(cfg),training_method_id=IDS['E2'])
  old,a=e2_batch(m,f.samples,f.geo,spec)
  r=cpu_zero_update(RawModel(),f.samples,f.geo,'query_minco_pg1_e2_paired_v1')
  self.assertEqual(old['loss'],r['record']['total'])
  for key in ['raw','raw_grad','coeffs','durations','duration_grad','positions','velocities','time_weights']:
   self.assertTrue(np.array_equal(a[key],r['arrays'][key]),key)
 def test_trusted_near_far_infeasible_only_goal_masked_M0_M1_M4(self):
  for M in [0,1,4]:
   for goal in [1.,5.8]:
    f=self.blocked(M=M,goal=goal);m=RawModel()
    with torch.no_grad():m.raw[:,6]-=.2
    r=cpu_zero_update(m,f.samples,f.geo,PG1)
    self.assertEqual(r['record']['goal_loss_masked_count'],1);self.assertEqual(r['record']['goal_query_masked_count'],6)
    self.assertFalse(np.count_nonzero(r['arrays']['grad_goal']))
    self.assertGreater(r['record']['losses']['obstacle'],0)
    self.assertGreater(np.linalg.norm(r['arrays']['grad_obstacle']),0)
    self.assertGreater(r['record']['naive_goal_offline_only'],0)
    self.check(r,f)
 def test_unknown_or_partial_slab_retains_goal_and_M0_rejects(self):
  f=self.blocked(covers=False)
  _,g=authenticated_task_gates(f.samples,f.geo);self.assertTrue(g[0]['goal_enabled']);self.assertEqual(g[0]['status'],'UNKNOWN_NOT_PROVEN_INFEASIBLE')
  f.samples[0].teacher_coeffs=f.samples[0].teacher_coeffs[:0];f.samples[0].teacher_durations=f.samples[0].teacher_durations[:0]
  self.refresh(f,0)
  with self.assertRaises(ValueError):cpu_zero_update(RawModel(),f.samples,f.geo,PG1)
 def test_forged_proof_score_and_source_do_not_mask_goal(self):
  f=self.fixture(('free',));scene=json.loads(f.metadata[0].read_text())['scene'];scene['proven_infeasible']=True;self.refresh(f,0,scene)
  m=RawModel()
  with torch.no_grad():m.raw[:,19:21]=1000
  r=cpu_zero_update(m,f.samples,f.geo,PG1);self.assertEqual(r['record']['feasibility_mask'],[True]);self.assertFalse(np.count_nonzero(r['arrays']['raw_grad'][...,19:21]))
  f.geo.rows[f.samples[0].sample_id]['scene_source']['sha256']='0'*64
  with self.assertRaises(ValueError):cpu_zero_update(RawModel(),f.samples,f.geo,PG1)
 def test_wrong_frame_empty_flag_and_missing_slab_rejected(self):
  for what in ['frame','empty','slab']:
   f=self.fixture(('free',))
   if what=='frame':f.geo.rows['synthetic_0']['frame']='world'
   elif what=='empty':f.geo.rows['synthetic_0']['empty_certified']=False
   else:
    scene=json.loads(f.metadata[0].read_text())['scene'];scene['kind']='blocked';self.refresh(f,0,scene)
   with self.subTest(what=what),self.assertRaises(ValueError):cpu_zero_update(RawModel(),f.samples,f.geo,PG1)
 def test_free_all_query_goal_dynamic_domain_nonzero_gradients(self):
  f=self.fixture(('free',))
  m=RawModel()
  with torch.no_grad():m.raw[:,0:9]=.4;m.raw[:,9]=1.
  # Tight domain is source-bound, not an invented obstacle.
  scene=json.loads(f.metadata[0].read_text())['scene'];scene['flight_center_domain_m']=[[-.01,-.01,-.01],[.1,.1,.1]];self.refresh(f,0,scene)
  r=cpu_zero_update(m,f.samples,f.geo,PG1)
  for key in ['goal','dynamic','domain']:
   self.assertTrue((np.linalg.norm(r['arrays']['grad_'+key],axis=-1)>0).all(),key)
  self.assertFalse(np.count_nonzero(r['arrays']['grad_obstacle']));self.check(r,f)
 def test_proven_infeasible_keeps_all_other_component_gradients(self):
  f=self.blocked();m=RawModel()
  with torch.no_grad():m.raw[:,0:9]=.4;m.raw[:,9]=1.
  r=cpu_zero_update(m,f.samples,f.geo,PG1)
  self.assertFalse(np.count_nonzero(r['arrays']['grad_goal']))
  for k in ['obstacle','dynamic','domain']:self.assertTrue((np.linalg.norm(r['arrays']['grad_'+k],axis=-1)>0).all(),k)
 def test_raw_finite_difference_full_objective_and_time_channels(self):
  f=self.fixture(('free',),goal=2.4);m=RawModel()
  with torch.no_grad():m.raw[0,0]=-.5;m.raw[0,15]=.3;m.raw[0,16]=.2
  r=cpu_zero_update(m,f.samples,f.geo,PG1)
  records=[]
  for index in [0,6,9,15,16,19]:
   # Small duration/velocity derivatives are audited component-wise to avoid
   # cancellation against the much larger goal scalar. Still retain full-total FD.
   component='dynamic' if index in [9,15,16] else 'total'
   analytic=float(r['arrays']['grad_dynamic' if component=='dynamic' else 'raw_grad'][0,0,index])
   values=[];totals=[]
   for h in [1e-2,3e-3,1e-3]:
    initial=float(m.raw[0,index])
    with torch.no_grad():m.raw[0,index]=initial+h
    z=forward(m,f.samples,f.geo,PG1);plus=float((z['terms']['dynamic'] if component=='dynamic' else z['loss']).detach());tp=float(z['loss'].detach())
    with torch.no_grad():m.raw[0,index]=initial-h
    z=forward(m,f.samples,f.geo,PG1);minus=float((z['terms']['dynamic'] if component=='dynamic' else z['loss']).detach());tm=float(z['loss'].detach())
    with torch.no_grad():m.raw[0,index]=initial
    values.append((plus-minus)/(2*h));totals.append((tp-tm)/(2*h))
   records.append(dict(raw_channel=index,component=component,analytic=analytic,steps=[.01,.003,.001],
    finite_differences=values,total_finite_differences=totals,total_analytic=float(r['arrays']['raw_grad'][0,0,index]),
    note='FP32 total cancellation retained; FP64 polynomial/duration gradcheck is separate'))
   if index==19:self.assertEqual(analytic,0);self.assertEqual(values,[0.,0.,0.])
   else:self.assertTrue(all(abs(n-analytic)<=max(.05*abs(analytic),.001) for n in values),(index,analytic,values))
  (OUT/'audit/FINITE_DIFFERENCES.json').write_text(json.dumps(records,indent=2))
 def test_float64_pure_derivative_duration_gradcheck(self):
  c=torch.randn(1,6,3,3,6,dtype=torch.float64,requires_grad=True)*.01
  dt=torch.full((1,6,3),.5,dtype=torch.float64,requires_grad=True)
  def func(c,t):
   tr=MincoTrajectory(c,t,torch.ones(1,6,dtype=torch.bool),{})
   x=analytic_samples(tr,5);return x['jerk'].square().mean()+x['a'].square().mean()
  self.assertTrue(torch.autograd.gradcheck(func,(c,dt),eps=1e-6,atol=1e-5,rtol=1e-3))
 def test_illegal_shapes_times_and_invalid_solve(self):
  f=self.fixture();out=TrajectoryPostprocessor()(fixture_raw()[None].expand(2,-1,-1),torch.stack([s.state for s in f.samples]))
  for case in ['solve','time','nan']:
   c=out.trajectory.coeffs.clone();d=out.parameters.durations.clone();valid=out.trajectory.solve_valid.clone()
   if case=='solve':valid[0,0]=False
   elif case=='time':d[0,0,0]=.01
   else:c[0,0,0,0,0]=float('nan')
   with self.assertRaises(ValueError):analytic_samples(MincoTrajectory(c,d,valid,{}))
 def test_audit_rejects_tampering_proof_identity_losses_and_gradient(self):
  f=self.fixture();r=cpu_zero_update(RawModel(),f.samples,f.geo,PG1)
  for key in ['proof','total','code','method','gradient','strata','array']:
   rec=copy.deepcopy(r['record']);a=copy.deepcopy(r['arrays'])
   if key=='proof':rec['gates'][0]['goal_enabled']=False
   elif key=='total':rec['total']+=1
   elif key=='code':rec['source_files']={}
   elif key=='method':rec['recipe']['goal_scale']=999
   elif key=='gradient':a['raw_grad'][0,0,19]=1
   elif key=='strata':rec['teacher_count_strata']['M1']['samples']+=1
   else:a['analytic_jerk'][0,0,0,0]+=2
   with self.subTest(key=key),self.assertRaises(ValueError):audit_cpu(rec,a,f.samples,f.geo)
 def test_empty_certificate_export_strict_finite(self):
  d=dict(dense_surface_clearance_m=float('inf'),certificates=[dict(surface_clearance_lower_bound_m=float('inf'))])
  normalized=strict_diagnostics(d,True);json.dumps(normalized,allow_nan=False)
  self.assertIsNone(normalized['dense_surface_clearance_m'])
  for bad in [dict(loss=float('inf')),dict(dense_surface_clearance_m=float('nan'))]:
   with self.assertRaises(ValueError):strict_diagnostics(bad,True)
 def test_profile_budget_authorization_and_schedule_negative(self):
  self.assertEqual(profile('pg1_formal2880_v1')['updates'],2880);self.assertEqual(profile('pg1_pilot32_v1')['checkpoints'],[0,8,16,32])
  self.assertAlmostEqual(lr_at('pg1_formal2880_v1',0),2e-5);self.assertAlmostEqual(lr_at('pg1_formal2880_v1',2879),1e-6)
  s=draft(PG1,'pg1_pilot32_v1',42,code_binding());json.dumps(s,allow_nan=False)
  with self.assertRaisesRegex(ValueError,'NOT_QUALIFIED'):validate_spec(s)
  a=authorization_template(s);self.assertIs(a['approved'],False)
  with self.assertRaises(ValueError):authorize(s,a)
  for key,value in [('additional_updates',960),('seed',43),('training_method_id','query_minco_rsps_pilot32_e2_v1')]:
   bad=copy.deepcopy(s);bad[key]=value
   with self.assertRaises(ValueError):validate_spec(bad)
  updates=[dict(completed_optimizer_updates=i+1,order_cursor=i+1,lr=lr_at(s['profile'],i),spec_sha256=digest(s)) for i in range(32)]
  self.assertTrue(audit_schedule(s,updates,[0,8,16,32])['passed'])
  updates[1]['completed_optimizer_updates']=1
  with self.assertRaises(ValueError):audit_schedule(s,updates,[0,8,16,32])
 def test_real_query_model_CPU_benchmark_and_record(self):
  retained=Path(tempfile.mkdtemp(prefix='synthetic_b8_',dir=str(OUT/'audit')))
  f=Fixtures(retained/'source_rows',['free','obstacle']*4,goal=2.4);torch.manual_seed(42);model=QueryMincoPlanner().train()
  rows=[]
  for method in METHODS:
   result=cpu_zero_update(model,f.samples,f.geo,method);audit=self.check(result,f)
   rows.append(dict(method=method,wall_ms=result['record']['wall_ms'],thread_cpu_ms=result['record']['thread_cpu_ms'],
                    array_bytes=sum(v.nbytes for v in result['arrays'].values()),parameter_count=sum(p.numel() for p in model.parameters())))
   if method==PG1:
    (OUT/'audit/SYNTHETIC_RECORD.json').write_text(json.dumps(result['record'],indent=2,allow_nan=False))
    (OUT/'audit/SYNTHETIC_AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    with (OUT/'audit/SYNTHETIC_ARRAYS.npz').open('wb') as file:np.savez(file,**result['arrays'])
  (OUT/'CPU_BENCHMARK.json').write_text(json.dumps(dict(scope='single randomly initialized QueryModel syntheticB8 CPU forward/backward and gradient diagnostics perarm; not frozenC1/G2/GPU/optimizer/eval timing',source_rows=str(retained/'source_rows'),samples=rows),indent=2))
if __name__=='__main__':unittest.main(verbosity=2)
