"""Exact-math contract T01-T12: synthetic math plus evidence-bound integration."""
import copy,json,os,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import minimize
from query_minco.training.io import read,sha,digest
from query_minco.training.run_spec import ROOT
from query_minco.training.pg3_next_exact_v1.quality import quality,select
from query_minco.training.pg3_next_exact_v1.projection import project,search,ALPHAS
from query_minco.training.pg3_next_exact_v1.proxy import times_and_active,proxy,Parameters,semantic_sha
from query_minco.training.pg3_next_exact_v1.minimal import MinimalObjective,LAMBDA_MODE,WEIGHTS,ID
from query_minco.evaluation.validation import evaluate,validate_teacher
BASE='10e1d9e4a30b6330d194d0f35a18fd24a0fa25d7'
def evidence():
 p=os.environ.get('PG3_NEXT_EVIDENCE')
 if not p:raise unittest.SkipTest('Actual new-source diagnostic evidence required')
 return Path(p)
def candidate(q=0,**kwargs):return dict(query_id=q,solve_valid=True,hard_physics_pass=True,goal_pass=True,QA=1.,QJ=2.,QT=1.,duration_s=2.2,**kwargs)
class ExactContract(unittest.TestCase):
 def setUp(self):torch.manual_seed(42);torch.set_num_threads(1)
 def test_T01_SOURCE_IMMUTABLE(self):
  names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
  for n in names:self.assertEqual((ROOT/n).read_bytes(),subprocess.check_output(['git','show',BASE+':'+n],cwd=ROOT))
 def test_T02_LOSS_EFFECTIVE_WEIGHTS(self):
  import sys
  sys.path.insert(0,str(ROOT/'tests'))
  from test_pg3_v2_1_no_corridor import RawModel,base_forward
  from query_minco.inference.identity import state_sha
  from query_minco.training.pg2_guided_v2.spec import weights_for_method
  z,s,f=base_forward(RawModel());m=RawModel();z,s,f=base_forward(m)
  obj=MinimalObjective(m)
  with patch('query_minco.training.pg3_v2_no_corridor.objective.full_stopping_risk_v21',side_effect=AssertionError('Stop must not run')),patch('query_minco.training.pg3_v2_no_corridor.objective.geometry_query_v21',side_effect=AssertionError('Geom must not run')):
   r=obj.augment(z,[s],0)
  expected=z['loss']+LAMBDA_MODE*r['terms']['mode'];self.assertTrue(torch.equal(expected,r['loss']));self.assertIs(r['base_loss'],z['loss'])
  pp=list(m.parameters());a=torch.autograd.grad(r['loss'],pp,retain_graph=True);b=torch.autograd.grad(expected,pp,retain_graph=True);self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)));obj.assignment.rollback(0)
  w=weights_for_method(dict(z['weights']),'query_minco_pg2_guided_v2_safe_v1')
  for k in ('lateral','acceleration','jerk','time'):self.assertEqual(w[k],0.)
 def test_T03_QUALITY_SI(self):
  c=np.random.RandomState(42).normal(0,.02,(3,3,6));d=np.array([.4,.5,.6]);r=quality(c,d)
  for order,key in [(2,'E_a_SI'),(3,'E_j_SI')]:
   expected=sum(quad(lambda t:float(np.square(evaluate(cc,t,order)).sum()),0,float(T),epsabs=1e-12,epsrel=1e-12)[0] for cc,T in zip(c,d));self.assertAlmostEqual(float(r[key][0,0]),expected,places=10)
  self.assertAlmostEqual(float(r['QA'][0,0]),float(r['E_a_SI'][0,0])/(64*2.2),places=12);self.assertAlmostEqual(float(r['QJ'][0,0]),float(r['E_j_SI'][0,0])/(10000*2.2),places=12)
  line=np.zeros((3,3,6));line[:,0,1]=1.;z=quality(line,d);self.assertEqual(float(z['QA'][0,0]),0.);self.assertEqual(float(z['QJ'][0,0]),0.);self.assertAlmostEqual(float(z['QT'][0,0]),1.5/2.2)
  cc=c/(1.2**np.arange(6))[None,None];s=quality(cc,d*1.2);self.assertAlmostEqual(float(s['E_a_SI'][0,0]),float(r['E_a_SI'][0,0])/1.2**3,places=10);self.assertAlmostEqual(float(s['E_j_SI'][0,0]),float(r['E_j_SI'][0,0])/1.2**5,places=10)
 def test_T04_QUALITY_SELECTOR(self):
  a=candidate(3);b=candidate(1);self.assertEqual(select([a,b])[0],1)
  bad=dict(candidate(0),QA=-100.,hard_physics_pass=False);self.assertEqual(select([bad,a])[0],3);self.assertEqual(select([bad]),(None,'NO_CERTIFIED_CANDIDATE'))
  for key in ('solve_valid','goal_pass'):
   self.assertIsNone(select([dict(a,**{key:False})])[0])
  self.assertIsNone(select([dict(a,QJ=float('nan'))])[0])
 def test_T05_TEACHER_IDENTITY(self):
  import sys
  sys.path.insert(0,str(ROOT/'tests'));from test_pg3_v2_1_no_corridor import fixture
  from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
  from query_minco.training.pg3_feasible_mode_v1.spec import recipe
  s,f=fixture();bank=TeacherBank(dict(recipe(),build_corridor=False));ok=bank.get(s,f);self.assertTrue(all(t['certified'] for t in ok['teachers']))
  broken=copy.deepcopy(f);broken.row['obstacles'][0]['min']=[.9,-.2,-.2];broken.row['obstacles'][0]['max']=[1.1,.2,.2]
  with self.assertRaises(ValueError):bank.get(s,broken)
 def test_T06_ADAM_REPLAY(self):
  p=evidence();r=read(p/'COUNTERFACTUAL/REPLAY_SUMMARY.json');self.assertTrue(r['all_replays_exact']);self.assertGreaterEqual(r['cases'],9)
 def test_T07_PROJECTION_MATH(self):
  a=np.array([-1.,2.]);G=np.array([[1.,0.]]);d,r=project(a,G);np.testing.assert_allclose(d,[0,2],atol=1e-8);self.assertLessEqual(r['kkt_max'],1e-7)
  G=np.array([[1.,0.],[-1.,0.],[0.,1.]]);d,r=project(a,G);x=minimize(lambda x:.5*np.sum((x-a)**2),np.zeros(2),jac=lambda x:x-a,constraints=[dict(type='ineq',fun=lambda x:G@x,jac=lambda x:G)],method='SLSQP');np.testing.assert_allclose(d,x.x,atol=1e-7)
  for bad in (np.array([[0.,0.]]),np.array([[np.nan,0.]])):
   with self.assertRaises(ValueError):project(a,bad)
  c=torch.zeros(3,3,6,dtype=torch.double);c[:,0,0]=torch.tensor([0,.8,1.6]);c[:,0,1]=2.;c[:,1,0]=.5;dur=torch.full((3,),.4,dtype=torch.double);boxes=[dict(name='a',min=[1,-.2,-.2],max=[1.1,.2,.2])]*2
  _,_,b,meta=times_and_active(c.numpy(),dur.numpy(),boxes);self.assertEqual(b,0);self.assertGreater(meta['tied_minima_count'],1)
  c.requires_grad_();h,_=proxy(c,dur,boxes);self.assertTrue(torch.isfinite(torch.autograd.grad(h,c)[0]).all())
 def test_T08_NONLINEAR_VALIDATION(self):
  from query_minco.trajectory import MincoAdapter
  head=torch.zeros(1,6,3,3);head[:,:,1,0]=1.2;tail=head.clone();tail[:,:,0,0]=2.4
  inner=torch.tensor([[[[.8,.3,0],[1.6,.3,0]]]*6]);dur=torch.full((1,6,3),.7)
  scene=dict(obstacles=[dict(name='lower',min=[1.15,-1,-1],max=[1.25,.04,1]),dict(name='upper',min=[1.15,1.1,-1],max=[1.25,2.,1])],flight_center_domain_m=[[0,-3,-2],[3,3,2]])
  def hard(pts):
   tr=MincoAdapter()(head,tail,pts,dur);return validate_teacher(tr.coeffs[0,0].numpy(),dur[0,0].numpy(),head[0,0].numpy(),tail[0,0].numpy(),pts[0,0].numpy(),scene)['passed']
  self.assertTrue(hard(inner));trial=inner.clone();trial[:,:,:,1]+=.6;self.assertFalse(hard(trial))
  differentiable_inner=inner.clone().requires_grad_();c=MincoAdapter()(head,tail,differentiable_inner,dur).coeffs;h,meta=proxy(c[0,0].double(),dur[0,0].double(),scene['obstacles']);self.assertEqual(meta['active_box_name'],'lower');gradient=torch.autograd.grad(h,differentiable_inner)[0];self.assertGreater(float(gradient[:,:,:,1].sum()*.6),0.)
  # Tangent proxy permits moving away from lower, yet nonlinear upper collision rejects the full step.
  def evaluate_trial(delta):
   pts=inner.clone();pts[:,:,:,1]+=float(delta[0]);return dict(protected_pass=hard(pts),all_numerical_valid=True,protected_goal_pass=True,relative_update_norm=abs(float(delta[0]))/.6)
  r=search(np.array([.6]),evaluate_trial,baseline_mode=1.);self.assertEqual(r['status'],'ACCEPTED');self.assertLess(r['chosen_alpha'],1.)
 def test_T09_PROGRESS_NOT_ZERO(self):
  r=search(np.zeros(2),lambda d:dict(protected_pass=True,protected_goal_pass=True,all_numerical_valid=True,relative_update_norm=0.),baseline_mode=1.)
  self.assertFalse(r['progress_not_zero']);p=evidence();x=read(p/'COUNTERFACTUAL/RESULT.json');self.assertIn(x['method_decision'],('NO_GO_FOR_METHOD_IMPLEMENTATION','POTENTIAL_REQUIRES_TRANSACTION_DESIGN'))
 def test_T10_TRANSACTIONAL_ISOLATION(self):
  p=evidence();self.assertEqual(read(p/'COUNTERFACTUAL/ISOLATION.json')['status'],'PASS')
 def test_T11_FOREST_OFFLINE_SHADOW(self):
  from query_minco.inference.observed_space import DepthHistory
  self.assertEqual(DepthHistory().diagnostics(np.zeros((1,3)),0)['reason'],'NO_FRESH_DEPTH')
  p=evidence();r=read(p/'READINESS/RESULT.json');self.assertTrue(r['all_fail_closed']);self.assertEqual(r['control_tracking'],'NOT_AVAILABLE')
 def test_T12_RUN_AUTHORITY(self):
  import tempfile,sys
  with tempfile.TemporaryDirectory() as d:
   forbidden=Path(d)/'forbidden_formal';proc=subprocess.run([sys.executable,'-m','experiments.pg3_next_exact_v1.run','formal128','--output',str(forbidden)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE);self.assertNotEqual(proc.returncode,0);self.assertFalse(forbidden.exists())
  p=evidence();r=read(p/'MINIMAL/QUALIFICATION.json');self.assertEqual(r['formal_training'],'NOT_RUN');self.assertFalse(r['formal_authorized']);self.assertNotEqual(r['method_id'],'query_minco_pg3_v21_ablation_m_v1')
