"""PG3 contract cases T01-T22. Mathematical tests never train the network."""
import unittest,copy,math
from types import SimpleNamespace
import numpy as np
import torch
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.trajectory.types import MincoTrajectory
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.training.pg3_feasible_mode_v1.assignment import assign
from query_minco.training.pg3_feasible_mode_v1.corridor import *
from query_minco.training.pg3_feasible_mode_v1.bottleneck import *
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import representability,inverse_raw,landmarks,TeacherBank
from query_minco.training.pg3_feasible_mode_v1.losses import mode_query,conditional_control,diversity
from query_minco.evaluation.validation import evaluate,validate_teacher
from query_minco.training.losses.base_imitation import NoTeacherError

def teacher(y=.5):
 return dict(inner=[[.8,y,0],[1.6,y,0]],tail=[[2.4,y,0],[0,0,0],[0,0,0]],durations=[.7,.7,.7],representable=True,certified=True,representability=dict(acceleration=True,duration=True),bottlenecks=[],mode_signature=[])
def curve():
 c=torch.zeros(3,3,6,dtype=torch.float64);c[:,0,0]=torch.tensor([0.,.8,1.6]);c[:,0,1]=2.;c[:,1,0]=.5
 return c,torch.full((3,),.4,dtype=torch.float64)
def scene():
 return dict(obstacles=[dict(name="thin",min=[1.,-.2,-.2],max=[1.1,.2,.2])],flight_center_domain_m=[[0,-1,-1],[3,1,1]])
class PG3(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_T01_decoder_two_inner_and_scores(self):
  raw=torch.zeros(2,6,21,requires_grad=True);state=torch.zeros(2,12)
  out=TrajectoryPostprocessor()(raw,state)
  self.assertEqual(tuple(out.parameters.inner_pts.shape),(2,6,2,3));self.assertEqual(tuple(out.trajectory.coeffs.shape),(2,6,3,3,6))
  self.assertTrue(torch.equal(out.parameters.score,raw[...,19]))
  g=torch.autograd.grad(out.parameters.inner_pts.sum()+out.parameters.total_time.sum(),raw)[0]
  self.assertEqual(int(torch.count_nonzero(g[...,19:21])),0)
 def test_T02_representability_rejects_and_inverse(self):
  t=teacher();r=inverse_raw(t["inner"],t["tail"],t["durations"]);self.assertEqual(len(r),21)
  bad=copy.deepcopy(t);bad["inner"][0][1]=5
  self.assertFalse(representability(bad["inner"],bad["tail"],bad["durations"])["positions"])
  with self.assertRaises(ValueError):inverse_raw(bad["inner"],bad["tail"],bad["durations"])
  with self.assertRaises(ValueError):inverse_raw(t["inner"],t["tail"],[1,1,1])
 def test_T03_capacity_M1_rotation(self):
  seen=set()
  for step in range(5):
   a=assign(torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0],teachers=[0]),torch.zeros(6,dtype=torch.bool),[teacher()],step)
   self.assertEqual(len(a["selected"]),1);self.assertEqual(a["occupancy"],[2]);seen.update(a["selected"])
  self.assertEqual(seen,{1,2,3,4,5})
 def test_T04_M4_balanced_slots(self):
  a=assign(torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0,1,2,3],teachers=[0,1,2,3]),torch.zeros(6,dtype=torch.bool),[teacher(y) for y in (.5,-.5,.7,-.7)])
  self.assertEqual(len(a["selected"]),2);self.assertEqual(sum(a["occupancy"]),6);self.assertLessEqual(max(a["occupancy"]),2)
 def test_T05_no_teacher_id_topology_inference(self):
  c,d=curve();tr=MincoTrajectory(c[None,None].expand(1,6,-1,-1,-1),d[None,None].expand(1,6,-1),torch.ones(1,6,dtype=torch.bool),{})
  a=dict(selected=[1,2],route_teacher_id=[0,0,1,-1,-1,-1])
  loss,rec=diversity(tr,[a],[dict(teachers=[teacher(),teacher()])])
  self.assertEqual(float(loss),0.);self.assertEqual(rec,[])
 def test_T06_M0_and_unknown_empty(self):
  a=assign(torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[],teachers=[]),torch.zeros(6,dtype=torch.bool),[])
  self.assertEqual(a["route_teacher_id"],[-1]*6);self.assertIn("UNKNOWN",a["reasons"][0])
  empty=dict(obstacles=[],flight_center_domain_m=[[0,-1,-1],[3,1,1]])
  with self.assertRaises(ValueError):validate_scene(empty)
  self.assertEqual(validate_scene(empty,True).shape,(2,3))
 def test_T07_direct_mode_raw_path_and_structural_coeff_none(self):
  raw=torch.zeros(1,6,21,requires_grad=True);o=TrajectoryPostprocessor()(raw,torch.zeros(1,12))
  value,way=mode_query(o.parameters.inner_pts[0,1],o.parameters.tail_pva[0,1],o.parameters.durations[0,1],teacher(),recipe())
  g=torch.autograd.grad(value,(raw,o.trajectory.coeffs),allow_unused=True,retain_graph=True)
  self.assertGreater(float(g[0][0,1,:6].norm()),0);self.assertIsNone(g[1]);self.assertEqual(float(g[0][...,19:21].norm()),0)
  self.assertGreater(float(way),0)
 def test_T07_mode_inner_finite_difference(self):
  w=torch.tensor([[.7,-.4,.1],[1.8,-.5,.1]],dtype=torch.float64,requires_grad=True)
  tail=torch.tensor([[2.4,0,0],[.1,0,0],[0,0,0]],dtype=torch.float64,requires_grad=True);d=torch.tensor([.7,.7,.7],dtype=torch.float64)
  self.assertTrue(torch.autograd.gradcheck(lambda a,b:mode_query(a,b,d,teacher(),recipe())[0],(w,tail),fast_mode=True))
 def test_T08_lateral_negative_gradient_correct_direction(self):
  c,d=curve();c[:,1,0]=-.4;c.requires_grad_();d.requires_grad_()
  label=[dict(valid=True,position=[1.2,.5,0],normal=[1,0,0],progress=.5)]
  loss,_=bottleneck_loss(c,d,label);g=torch.autograd.grad(loss,c)[0]
  self.assertLess(float(g[:,1,0].sum()),0)
 def test_T09_no_crossing_and_empty_loss(self):
  c,d=curve();self.assertEqual(crossing_events(c.numpy(),d.numpy(),5)["status"],"NO_CROSSING")
  cc=c.requires_grad_();dd=d.requires_grad_();loss,diag=bottleneck_loss(cc,dd,[])
  self.assertEqual(diag["count"],0);self.assertEqual(float(loss),0);self.assertEqual(float(torch.autograd.grad(loss,cc)[0].norm()),0)
 def test_T10_tangent_multiple_and_segment_boundary(self):
  c,d=curve();event=crossing_events(c.numpy(),d.numpy(),.8)
  self.assertEqual(len(event["events"]),1);self.assertEqual(event["selected"]["segment"],0)
  cc=np.zeros((3,3,6));cc[:,0,0]=1.;dd=np.ones(3)
  cc[0,0,:3]=[.25,-1,1]
  e=crossing_events(cc,dd,0);self.assertEqual(e["status"],"TANGENT_ONLY")
  cc=np.zeros((3,3,6));cc[:,0,0]=2.;cc[0,0,:4]=np.polynomial.polynomial.polyfromroots([.2,.5,.8])
  e=crossing_events(cc,dd,0);self.assertEqual(e["status"],"MULTIPLE_FORWARD_CROSSINGS")
  self.assertAlmostEqual(e["selected"]["time_s"],.2)
 def test_T11_phase_alignment_is_proxy(self):
  c,d=curve();z=at_progress(c,d,[.5]);changed=torch.tensor([.2,.7,.3],dtype=torch.float64)
  zz=at_progress(c,changed,[.5]);self.assertGreater(float((z-zz).norm()),.05)
 def test_T11_bottleneck_coeff_duration_FD(self):
  c,d=curve();c[:,1,1]=.2;c.requires_grad_();d.requires_grad_()
  labels=[dict(valid=True,position=[1.4,.8,.2],normal=[1,0,0],progress=.43)]
  self.assertTrue(torch.autograd.gradcheck(lambda a,b:bottleneck_loss(a,b,labels)[0],(c,d),fast_mode=True))
 def test_T12_power_Bernstein_random_equality(self):
  torch.manual_seed(4);c=torch.randn(7,3,6,dtype=torch.float64);d=torch.rand(7,dtype=torch.float64)+.2;B=bernstein(c,d)
  for u in (0.,.17,.5,.93,1.):
   lhs=(c*(d*u)[:,None,None]**torch.arange(6,dtype=torch.float64)).sum(-1)
   rhs=sum(B[:,k]*math.comb(5,k)*u**k*(1-u)**(5-k) for k in range(6))
   self.assertTrue(torch.allclose(lhs,rhs,atol=1e-12,rtol=1e-12))
 def test_T12_Bernstein_coeff_duration_FD(self):
  c,d=curve();c.requires_grad_();d.requires_grad_()
  self.assertTrue(torch.autograd.gradcheck(lambda a,b:bernstein(a,b).square().sum(),(c,d),fast_mode=True))
 def test_T12_Casteljau_subinterval(self):
  c,d=curve();B=bernstein(c,d);sub=restrict(B[1],.2,.7)
  for u in (.1,.4,.8):
   value=sum(sub[k]*math.comb(5,k)*u**k*(1-u)**(5-k) for k in range(6))
   expected=torch.tensor(evaluate(c[1].numpy(),float(d[1])*(.2+.5*u)))
   self.assertTrue(torch.allclose(value,expected,atol=1e-12))
 def test_T13_polytope_containment_and_independent_hard(self):
  c,d=curve();s=scene();corr=build_corridor(c.numpy(),d.numpy(),s)
  self.assertTrue(corr["complete"]);loss,diag=penalty(c,d,corr,s)
  self.assertEqual(float(loss),0);self.assertTrue(diag["contained"])
  head=np.stack([evaluate(c[0].numpy(),0.,k) for k in range(3)]);tail=np.stack([evaluate(c[-1].numpy(),float(d[-1]),k) for k in range(3)]);inner=np.stack([evaluate(c[i].numpy(),float(d[i])) for i in (0,1)])
  self.assertTrue(validate_teacher(c.numpy(),d.numpy(),head,tail,inner,s)["passed"])
 def test_T14_safe_points_do_not_prove_safe_chord(self):
  s=scene();B=np.tile([.5,0,0.],(6,1));B[3:]=[1.5,0,0]
  self.assertIsNone(polytope(B,s))
 def test_T15_missing_corridor_zero_no_certificate(self):
  c,d=curve();c[:,1,0]=0
  corr=build_corridor(c.numpy(),d.numpy(),scene(),max_depth=0)
  self.assertFalse(corr["complete"]);self.assertTrue(corr["missing"])
  fake=dict(leaves=[],missing=[1],complete=False,status="CORRIDOR_UNAVAILABLE")
  loss,diag=penalty(c.requires_grad_(),d.requires_grad_(),fake,scene())
  self.assertEqual(float(loss),0);self.assertFalse(diag["contained"])
 def test_T15_corridor_coefficient_duration_FD(self):
  c,d=curve();corr=build_corridor(c.numpy(),d.numpy(),scene());bad=c.clone();bad[:,1,0]=-.4;bad.requires_grad_();d.requires_grad_()
  self.assertTrue(torch.autograd.gradcheck(lambda a,b:penalty(a,b,corr,scene())[0],(bad,d),fast_mode=True))
 def test_T15_old_proof_shift_and_invalid_box_rejected(self):
  c,d=curve();corr=build_corridor(c.numpy(),d.numpy(),scene());proof=copy.deepcopy(corr["leaves"][0]["proof"]);proof["b"]=[x+100 for x in proof["b"]]
  self.assertFalse(verify_polytope(proof,scene()))
  bad=scene();bad["obstacles"][0]["min"]=[3,3,3]
  with self.assertRaises(ValueError):validate_scene(bad)
 def test_T16_safe_unmatched_never_pulled(self):
  hard=torch.ones(6,dtype=torch.bool)
  a=assign(torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0],teachers=[0]),hard,[teacher()])
  self.assertFalse(a["selected"]);self.assertEqual(a["route_teacher_id"][1:],[-1]*5)
 def test_T17_quality_gate_and_coefficient_duration_FD(self):
  torch.manual_seed(2);c=(torch.randn(1,6,3,3,6,dtype=torch.float64)*.01).requires_grad_();d=torch.full((1,6,3),.6,dtype=torch.float64,requires_grad=True)
  hard=torch.zeros(1,6,dtype=torch.bool);hard[0,0]=True;goal=torch.ones_like(hard);signed=torch.zeros(1,6,dtype=torch.float64)
  def fn(a,b):
   tr=MincoTrajectory(a,b,torch.ones_like(hard),{})
   z,_=conditional_control(tr,hard,goal,signed,recipe());return z["qa"]+z["qj"]+z["qt"]+z["lateral"]
  self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True,atol=1e-5))
  g=torch.autograd.grad(fn(c,d),c)[0];self.assertGreater(float(g[0,0].norm()),0);self.assertEqual(float(g[0,1:].norm()),0)
 def test_T18_time_energy_tradeoff(self):
  from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
  c,d=curve();c[:,:,2]=.3
  tr=MincoTrajectory(c[None,None].expand(1,6,-1,-1,-1),d[None,None].expand(1,6,-1),torch.ones(1,6,dtype=torch.bool),{})
  first=dynamic_quality_loss(tr)["integrals_si"]["acceleration"]
  scaled=c.clone()
  for k in range(6):scaled[:,:,k]*=1.25**k
  short=MincoTrajectory(scaled[None,None].expand(1,6,-1,-1,-1),d[None,None].expand(1,6,-1)/1.25,tr.solve_valid,{})
  self.assertTrue(bool((dynamic_quality_loss(short)["integrals_si"]["acceleration"]>first).all()))
 def test_T19_old_thresholds(self):
  c=recipe();self.assertEqual(c["dynamic_limits"],[3.,8.,100.]);self.assertEqual((c["radius"],c["margin"]),(.12,.03))
 def test_T20_physical_no_score_supervision(self):
  r=torch.randn(1,6,21)*.1;r.requires_grad_();out=TrajectoryPostprocessor()(r,torch.zeros(1,12))
  d=dict(valid=True,position=[1.2,.6,0],normal=[1,0,0],progress=.4)
  loss,_=bottleneck_loss(out.trajectory.coeffs[0,1],out.parameters.durations[0,1],[d])
  g=torch.autograd.grad(loss,r)[0];self.assertEqual(float(g[...,19:21].norm()),0);self.assertGreater(float(g[...,15:19].norm()),0)
 def test_T21_uncertified_teacher_and_illegal_duration_rejected(self):
  t=teacher();t["certified"]=False
  with self.assertRaises(ValueError):mode_query(torch.zeros(2,3),torch.zeros(3,3),torch.ones(3),t,recipe())
  c,d=curve()
  with self.assertRaises(ValueError):bernstein(c,-d)
 def test_T22_experimental_method_namespace(self):
  from query_minco.training.pg3_feasible_mode_v1.spec import METHODS
  self.assertEqual(len(METHODS),3)
  self.assertTrue(all(k.startswith("query_minco_pg3_feasible_mode_v1") for k in METHODS))

 def test_T22_run_spec_rejects_extra_updates_and_gpu(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from query_minco.training.io import sha
  from query_minco.training.pg3_feasible_mode_v1.execution import make_spec,validate,authorization,BASELINE
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/"binding.json";p.write_text(json.dumps(dict(code=dict(allow_uncommitted_sources=False))))
   ref=dict(path=str(p),sha256=sha(p));s=make_spec(ref,{}, {},BASELINE,"A42_test")
   for key,value in (("updates",33),("device","cuda"),("batch_size",1)):
    bad=copy.deepcopy(s);bad[key]=value
    with self.assertRaises(ValueError):validate(bad,authorization(bad,True,"tester"))
   with patch("query_minco.training.pg3_feasible_mode_v1.execution.verify_code"):
    a=authorization(s,True,"tester");a["spec_sha256"]="0"*64
    with self.assertRaises(ValueError):validate(s,a)

 def test_full_raw_route_directional_finite_difference(self):
  raw=torch.zeros(1,6,21);raw[...,:9]=torch.tensor([-1.5,.1,.1,-1.1,.2,0,-.6,0,0]);raw[...,15]=2.
  raw.requires_grad_();state=torch.zeros(1,12)
  direction=torch.zeros_like(raw);direction[0,1,:19]=torch.linspace(-.5,.5,19);direction=direction/direction.norm()
  def value(r):
   o=TrajectoryPostprocessor()(r,state)
   a,_=mode_query(o.parameters.inner_pts[0,1],o.parameters.tail_pva[0,1],o.parameters.durations[0,1],teacher(),recipe())
   b,_=bottleneck_loss(o.trajectory.coeffs[0,1],o.parameters.durations[0,1],[dict(valid=True,position=[1.2,.5,0],normal=[1,0,0],progress=.41)])
   return a+b
  ad=float((torch.autograd.grad(value(raw),raw)[0]*direction).sum())
  fd=float((value(raw.detach()+.001*direction)-value(raw.detach()-.001*direction))/.002)
  self.assertAlmostEqual(ad,fd,delta=.02*max(1.,abs(ad)))
 def test_coincident_plane_is_not_fake_crossing(self):
  c,d=curve();c[:,0]=0.;c[:,0,0]=1.
  e=crossing_events(c.numpy(),d.numpy(),1.)
  self.assertEqual(e["status"],"COINCIDENT_PLANE_SEGMENT_UNSUPPORTED");self.assertIsNone(e["selected"])

 def test_certified_empty_teacher_metadata_is_strict_json(self):
  import json
  c=torch.zeros(1,3,3,6);c[0,:,0,0]=torch.tensor([0.,.8,1.6]);c[0,:,0,1]=2.
  d=torch.full((1,3),.4);state=torch.zeros(12);state[0]=2.
  sample=SimpleNamespace(sample_id="empty_boundary",state=state,teacher_coeffs=c,teacher_durations=d)
  field=SimpleNamespace(row=dict(obstacles=[],flight_center_domain_m=[[0.,-1.,-1.],[3.,1.,1.]],empty_certified=True))
  result=TeacherBank(recipe()).get(sample,field)
  json.dumps(result,allow_nan=False)
  self.assertTrue(result["teachers"][0]["certified"])
  self.assertEqual(result["teachers"][0]["geometry_bound_semantics"],"UNBOUNDED_CERTIFIED_EMPTY")
  field.row["empty_certified"]=False
  with self.assertRaises(ValueError):TeacherBank(recipe()).get(sample,field)

if __name__=="__main__":unittest.main()
