"""R12 math/contracts tests; NEVER constructs optimizer."""
import copy,json,unittest,tempfile,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_pg1_v2_r1 import tr,raw_fixture
from query_minco.training.pg1_v2_r12.objective import tail_from_samples,compose
from query_minco.training.pg1_v2_r12.spec import recipe,R0,RTAIL,make_spec,authorization,validate
from query_minco.training.pg1_v2_r12.diagnostics import numpy_tail,ratio_guard
from query_minco.training.pg1.spec import recipe as old_recipe,METHODS as OLD_METHODS
from query_minco.training.pg1_v2_r1.losses import bottleneck_safety_loss,weighted_upper_tail
from query_minco.training.losses.set_physical_risk import segment_samples
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.io import sha,write
from query_minco.training.run_spec import ROOT
class R12(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def fixture(self,dtype=torch.float64):
  r,s=raw_fixture();t=TrajectoryPostprocessor()(r,s).trajectory
  p,v,w=segment_samples(t,64);b=torch.tensor([[[-2.,.3,-2.],[4.,.6,2.]]])
  return p.to(dtype),w.to(dtype),b.to(dtype)
 def test_tail_exact_R1_and_shape(self):
  p,w,b=self.fixture();a=tail_from_samples(p,w,[b],[False])
  z=bottleneck_safety_loss(p,w,[b],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)
  self.assertEqual(a.shape,(1,6));self.assertTrue(torch.equal(a,z['losses']['tail']))
 def test_independent_numpy(self):
  p,w,b=self.fixture();a=tail_from_samples(p,w,[b],[False])
  np.testing.assert_allclose(a,numpy_tail(p.numpy(),w.numpy(),[b.numpy()]),rtol=1e-12,atol=1e-12)
 def test_coeff_duration_gradient(self):
  tt=tr();c=tt.coeffs.requires_grad_();d=tt.durations.requires_grad_();b=c.new_tensor([[[.2,-.2,-.2],[.5,.2,.2]]])
  def fun(c,d):
   t=SimpleNamespace(coeffs=c,durations=d,solve_valid=tt.solve_valid)
   p,v,w=segment_samples(t,64);return tail_from_samples(p,w,[b],[False])
  self.assertTrue(torch.autograd.gradcheck(fun,(c,d),eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=True))
 def test_certified_empty_and_mixed(self):
  p,w,b=self.fixture();p=p.repeat(2,1,1,1).requires_grad_();w=w.repeat(2,1,1)
  a=tail_from_samples(p,w,[b.new_empty(0,2,3),b],[True,False])
  self.assertTrue(torch.equal(a[0],torch.zeros(6,dtype=a.dtype)))
  g=torch.autograd.grad(a.sum(),p)[0];self.assertFalse(torch.count_nonzero(g[0]));self.assertTrue(torch.isfinite(g).all())
  with self.assertRaises(ValueError):tail_from_samples(p[:1],w[:1],[b.new_empty(0,2,3)],[False])
 def test_invalid_geometry(self):
  p,w,b=self.fixture()
  for field,cert in [(b.flip(1),False),(b,True),(b.clone().requires_grad_(),False)]:
   with self.assertRaises(ValueError):tail_from_samples(p,w,[field],[cert])
 def test_raw_time_gradient_and_score_rows(self):
  r,s=raw_fixture()
  for piece,y in enumerate((.05,.12,.02)):r[:,:,piece*3+1]=math.atanh(y/4.)
  r.requires_grad_()
  out=TrajectoryPostprocessor()(r,s);p,v,w=segment_samples(out.trajectory,64);b=r.new_tensor([[[-2.,.3,-2.],[4.,.6,2.]]])
  g=torch.autograd.grad(tail_from_samples(p,w,[b],[False]).mean(),r)[0]
  self.assertTrue(torch.isfinite(g).all());self.assertFalse(torch.count_nonzero(g[...,19:21]));self.assertGreater(float(g[...,15:19].norm()),0)
 def test_zero_tail_is_exact_old_graph(self):
  x=torch.tensor([1.,2.],requires_grad=True);old=(x*x).sum();t=x.sum()
  self.assertIs(compose(old,t,0.),old)
  self.assertTrue(torch.equal(torch.autograd.grad(compose(old,t,0.),x,retain_graph=True)[0],torch.autograd.grad(old,x)[0]))
  with self.assertRaises(ValueError):compose(old,t,.003)
 def test_quantile_switch_is_nonsmooth(self):
  def get(x):
   xx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
   v=torch.stack([.3+xx,.3-xx,xx*0,xx*0,xx*0]).expand(1,6,5)
   a=weighted_upper_tail(v,torch.ones_like(v),.1).mean()
   return float(a),float(torch.autograd.grad(a,xx)[0])
  a,ga=get(-1e-5);b,gb=get(1e-5);c,gc=get(0)
  self.assertAlmostEqual(a,b);self.assertAlmostEqual(ga,-1);self.assertAlmostEqual(gb,1);self.assertTrue(np.isfinite(gc))
 def test_fixed_recipe_no_old_mutation(self):
  before=copy.deepcopy(OLD_METHODS);base=old_recipe('query_minco_pg1_task_complete_v1')
  a=recipe(R0);b=recipe(RTAIL)
  self.assertEqual(a['tail_scale'],0);self.assertEqual(b['tail_scale'],.0025);self.assertEqual(b['tail_fraction'],.1);self.assertFalse(b['tail_rho']);self.assertEqual(b['quality_training_terms'],[])
  for k in ['dynamic_limits','time_contract','radius','margin','goal_scale','dynamic_scale','domain_scale']:self.assertEqual(b[k],base[k])
  self.assertEqual(OLD_METHODS,before)
 def test_gradient_guard_negative_cases(self):
  self.assertEqual(ratio_guard(1.,11.,0),(11.,1))
  self.assertEqual(ratio_guard(1.,1.,1),(1.,0))
  with self.assertRaises(ValueError):ratio_guard(1.,11.,1)
  self.assertEqual(ratio_guard(0.,0.,0),(None,0))
  with self.assertRaises(ValueError):ratio_guard(float('nan'),1.,0)
 def test_plan_and_no_authorization(self):
  bp=ROOT/'research_outputs/pg1_v2_r11/audit003/_preparation/CPU0_ONLY_BINDINGS.json'
  s=make_spec(dict(path=str(bp),sha256=sha(bp)),RTAIL,'A42_test_R12')
  validate(s);self.assertFalse(authorization(s)['approved'])
  with self.assertRaises(ValueError):validate(s,training=True)
  bad=copy.deepcopy(s);bad['recipe']['tail_scale']=.004
  with self.assertRaises(ValueError):validate(bad)
  bad=copy.deepcopy(s);bad['device']='cuda'
  with self.assertRaises(ValueError):validate(bad)
 def test_R11_exclusive_journal_and_unique_count(self):
  from experiments.pg1_v2_r11.audit import save_model_snapshot
  from experiments.pg1_v2_r11.math_probes import mark_first_visits
  seen=set();self.assertEqual(mark_first_visits(['a','a','b'],seen),[True,False,True])
  parent=ROOT/'research_outputs/pg1_v2_r12/preflight001/_preparation'
  with tempfile.TemporaryDirectory(prefix='r11_io_',dir=str(parent)) as td:
   p=Path(td);(p/'runs').mkdir();(p/'audit').mkdir()
   for model in ['C1','E2','PG1']:save_model_snapshot(p,model,dict(CASES=[model]),[model],1.)
   with self.assertRaises(FileExistsError):save_model_snapshot(p,'E2',dict(CASES=['bad']),['E2'],1.)
   self.assertEqual(json.loads((p/'runs/E2_CASES.json').read_text())['records'],['E2'])

 def metadata_fixture(self):
  from query_minco.training.pg1_v2_r12.spec import cpu0_contract,runtime_identity,external_sources
  code=dict(git_commit='synthetic',files={},allow_uncommitted_sources=False)
  binding=dict(path='/synthetic/no_actual_data',sha256='a'*64)
  spec=dict(code=code,bindings=binding)
  pf=dict(schema='pg1_v2_r12_cpu0_seal_v2',passed=True,optimizer_created=False,optimizer_steps=0,
   contract=cpu0_contract(),runtime=runtime_identity(),code=code,bindings=binding,
   external_sources=external_sources(),model_weights_before='b'*64,model_weights_after='b'*64)
  return spec,pf
 def test_cpu0_metadata_positive_and_negative(self):
  from query_minco.training.pg1_v2_r12.spec import validate_cpu0_metadata
  s,pf=self.metadata_fixture();validate_cpu0_metadata(s,pf,'b'*64)
  cases=[]
  for key,val in [('schema','pg1_v2_r12_cpu0_seal_v1'),('optimizer_steps',1),('optimizer_created',True),('model_weights_after','c'*64)]:
   x=copy.deepcopy(pf);x[key]=val;cases.append(x)
  for key,val in [('tail_fraction',.2),('lambda_tail',.003),('tail_rho',True),('unique_tasks_per_method',383),('methods',[RTAIL])]:
   x=copy.deepcopy(pf);x['contract'][key]=val;cases.append(x)
  x=copy.deepcopy(pf);x['runtime']['torch']='wrong';cases.append(x)
  x=copy.deepcopy(pf);x['external_sources']={};cases.append(x)
  x=copy.deepcopy(pf);x['code']['git_commit']='wrong';cases.append(x)
  for x in cases:
   with self.assertRaises(ValueError):validate_cpu0_metadata(s,x,'b'*64)
 def sealed_fixture(self,root):
  from query_minco.training.pg1_v2_r12.spec import METHODS
  from query_minco.training.io import digest
  s,pf=self.metadata_fixture()
  (root/'audit').mkdir();(root/'_preparation').mkdir();(root/'runs').mkdir()
  batches=[['task_%03d'%((i*8+j)%384) for j in range(8)] for i in range(96)]
  cov=dict(batches=batches,covered_ids=sorted(set(sum(batches,[]))));cov['covered_ids_sha256']=digest(cov['covered_ids'])
  write(root/'_preparation/COVERAGE.json',cov)
  result=dict(pf,schema='pg1_v2_r12_cpu0_v2',source_bundle_clean=True,methods=list(METHODS),batches_per_method=96,unique_tasks_per_method=384,exposures_per_method=768)
  write(root/'audit/CPU0_RESULT.json',result)
  rows=[]
  for method,arm in zip(METHODS,['R0','Rtail']):
   for i in range(96):
    # Synthetic bytes test seal/hash coverage, not real-data qualification.
    (root/'runs'/('%s_batch%03d.npz'%(arm,i))).write_bytes(b'SYNTHETIC_UNIT_TEST_ONLY')
    rows.append(dict(method=method,batch=i,ids=batches[i],weights_before='b'*64,weights_after='b'*64,
     numpy_reconstruction=True,raw19_21_zero=True,score_rows_zero=True,original_loss_and_dynamic_regression=True,
     paired_forward_same_raw=True,all_original_numeric_guards=True,physical_imitation_gradient_ratio=1.,ratio_alarm_streak=0))
  (root/'runs/BATCHES.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
  pf['files']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
  write(root/'CPU0_SEAL.json',pf)
  return s,pf
 def test_cpu0_full_seal_and_missing_or_tampered_files(self):
  from query_minco.training.pg1_v2_r12.spec import validate_cpu0_seal
  parent=ROOT/'research_outputs/pg1_v2_r12/source_freeze_review001/_preparation'
  with tempfile.TemporaryDirectory(prefix='seal_test_',dir=str(parent)) as td:
   root=Path(td);s,pf=self.sealed_fixture(root);path=root/'CPU0_SEAL.json'
   validate_cpu0_seal(s,path,'b'*64)
   (root/'runs/R0_batch000.npz').write_bytes(b'TAMPERED')
   with self.assertRaises(ValueError):validate_cpu0_seal(s,path,'b'*64)
 def test_cpu0_wrong_coverage_and_traversal_rejected(self):
  from query_minco.training.pg1_v2_r12.spec import validate_cpu0_seal
  parent=ROOT/'research_outputs/pg1_v2_r12/source_freeze_review001/_preparation'
  with tempfile.TemporaryDirectory(prefix='seal_scope_',dir=str(parent)) as td:
   root=Path(td);s,pf=self.sealed_fixture(root);path=root/'CPU0_SEAL.json'
   pf['files']['../not_allowed']='0'*64;path.write_text(json.dumps(pf))
   with self.assertRaises(ValueError):validate_cpu0_seal(s,path,'b'*64)
   del pf['files']['../not_allowed']
   cov=read_for_test(root/'_preparation/COVERAGE.json');cov['batches'][0]=['same']*8
   (root/'_preparation/COVERAGE.json').write_text(json.dumps(cov));pf['files']['_preparation/COVERAGE.json']=sha(root/'_preparation/COVERAGE.json');path.write_text(json.dumps(pf))
   with self.assertRaises(ValueError):validate_cpu0_seal(s,path,'b'*64)
 def test_checkpoint_foreign_spec_or_cursor_rejected(self):
  from query_minco.training.pg1_v2_r12.spec import validate_checkpoint_blob
  from query_minco.training.io import digest
  spec=dict(method=RTAIL);b=dict(spec=spec,spec_sha256=digest(spec),completed_optimizer_updates=8,order_cursor=8,next_lr_index=8,weights_sha256='z')
  validate_checkpoint_blob(b,spec,8,'z')
  with self.assertRaises(ValueError):validate_checkpoint_blob(dict(b,completed_optimizer_updates=True),spec,8,'z')
  for k,v in [('spec',{}),('spec_sha256','bad'),('completed_optimizer_updates',16),('order_cursor',7),('next_lr_index',9),('weights_sha256','bad')]:
   with self.assertRaises(ValueError):validate_checkpoint_blob(dict(b,**{k:v}),spec,8,'z')
 def test_certified_empty_evaluation_uses_original_hard_validator(self):
  from unittest.mock import patch
  from query_minco.training.pg1_v2_r12.evaluation import candidate_metrics
  r,state=raw_fixture();sample=SimpleNamespace(state=state[0])
  scene=dict(kind='free',certified_empty=True,obstacles=[],flight_center_domain_m=[[0.,-4.,-2.],[10.,4.,2.]])
  with torch.no_grad(),patch('query_minco.training.pg1_v2_r12.evaluation.metrics',side_effect=AssertionError('nonempty path called')):
   z=candidate_metrics(r,sample,scene,0)
  self.assertIsNone(z['clearance_m']);self.assertIsNone(z['closest_time_s'])
  self.assertTrue(np.isfinite([z['vmax'],z['amax'],z['jmax'],z['time_s']]).all())
  self.assertEqual(z['per_query_terms']['tail'],0.)
  scene['certified_empty']=False
  with self.assertRaises(ValueError):candidate_metrics(r,sample,scene,0)

def read_for_test(p):return json.loads(p.read_text())

if __name__=='__main__':unittest.main(verbosity=2)
