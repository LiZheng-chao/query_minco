"""PG2 mathematics, detached guidance/labels, counterexamples. No optimizer."""
import copy,math,unittest
from types import SimpleNamespace
import numpy as np
import torch
from test_pg1_v2_r1 import tr,raw_fixture,np_integral
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.pg1.losses import analytic_samples,dynamic_limit_loss
from query_minco.training.losses.set_physical_risk import segment_samples,box_sdf
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
from query_minco.training.pg2_guided_v1.spec import recipe,validate_recipe,METHODS,draft_spec
from query_minco.training.pg2_guided_v1.losses import safety_loss,corridor_loss,guidance_loss,feasibility_loss,control_loss
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,hard_labels,verify_label_binding,tensor_digest
from query_minco.training.pg2_guided_v1.ranking import ranking_loss,safety_priority_projection

def field(boxes,empty=False,dtype=torch.float64):
 return SimpleNamespace(boxes=torch.tensor(boxes,dtype=dtype).reshape(-1,2,3),
  row=dict(empty_certified=empty,obstacles=[dict(name="b%d"%i,min=b[0],max=b[1]) for i,b in enumerate(boxes)],
   flight_center_domain_m=[[-1.,-2.,-2.],[10.,2.,2.]]))
def positions():
 x=torch.linspace(0,2,192,dtype=torch.float64)
 p=torch.zeros(1,6,192,3,dtype=torch.float64);p[:,:,:,0]=x
 v=torch.zeros_like(p);v[...,0]=1.;w=torch.ones_like(p[...,0])/192
 return p,v,w

class PG2(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_method_identity_and_switches(self):
  self.assertEqual(len(METHODS),4)
  for m in METHODS:self.assertEqual(validate_recipe(recipe(m))["method_id"],m)
  for k,val in [("radius",.1),("margin",.02),("points_per_segment",32),("dynamic_limits",[4.,8.,100.])]:
   c=recipe();c[k]=val
   with self.assertRaises(ValueError):validate_recipe(c)
 def test_invalid_config(self):
  for k,v in [("risk_eps_m",0.),("braking_acceleration_mps2",-1.),("lse_temperature",float("nan")),("tilt_max_rad",math.pi)]:
   c=recipe();c[k]=v
   with self.assertRaises(ValueError):validate_recipe(c)
 def test_draft_has_no_authorization(self):
  s=draft_spec({}, {}, {}, next(iter(METHODS)), "A42_PG2_CPU_Pilot32_run001")
  self.assertFalse(s["approved"]);self.assertIsNone(s["authorization"]);self.assertEqual(s["updates"],32)
 def test_empty_and_untrusted(self):
  p,v,w=positions();p.requires_grad_()
  z=safety_loss(p,v,w,[field([],True)],recipe())
  self.assertTrue(torch.equal(z["clearance"],torch.zeros(1,6,dtype=p.dtype)))
  self.assertEqual(torch.count_nonzero(torch.autograd.grad(z["clearance"].sum(),p)[0]),0)
  with self.assertRaises(ValueError):safety_loss(p,v,w,[field([],False)],recipe())
 def test_illegal_and_contradictory_boxes(self):
  p,v,w=positions()
  for f in [field([[[1,1,1],[0,2,2]]]),field([[[.8,-.2,-.2],[1.2,.2,.2]]],True)]:
   with self.assertRaises(ValueError):safety_loss(p,v,w,[f],recipe())
 def test_mixed_geometry(self):
  p,v,w=positions();z=safety_loss(p.repeat(2,1,1,1),v.repeat(2,1,1,1),w.repeat(2,1,1),[field([],True),field([[[.8,-.2,-.2],[1.2,.2,.2]]])],recipe())
  self.assertEqual(z["clearance"].shape,(2,6));self.assertEqual(float(z["clearance"][0].sum()),0.)
  self.assertTrue(bool(z["severe"][1].all()))
 def test_approach_only_braking(self):
  p,v,w=positions();p[:,:,:,0]=.1
  f=field([[[.8,-1,-1],[1.2,1,1]]])
  a=safety_loss(p,v,w,[f],recipe());b=safety_loss(p,-v,w,[f],recipe())
  self.assertTrue(bool((a["clearance"]>b["clearance"]).all()))
 def test_retreat_negative_gradient_points_to_anchor(self):
  p,v,w=positions();p.requires_grad_();f=field([[[.8,-.2,-.2],[1.2,.2,.2]]])
  z=safety_loss(p,v,w,[f],recipe());g=torch.autograd.grad(z["retreat"].sum(),p)[0]
  for q,i in enumerate(z["first_indices"][0]):
   target=z["retreat_targets"][0,q]
   self.assertGreater(float((-g[0,q,i]*(target-p[0,q,i].detach())).sum()),0)
   self.assertGreaterEqual(float(box_sdf(target[None],f.boxes)[0]),.15)
 def test_no_anchor_is_not_safe(self):
  p,v,w=positions();p[...,0]=1.
  z=safety_loss(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]])],recipe())
  self.assertEqual(z["reasons"],[["NO_PREVIOUS_SAFE_ANCHOR"]*6]);self.assertFalse(z["certificate"])
 def test_safety_planar_gradcheck(self):
  p,v,w=positions();p[...,0]=.2;p=p.requires_grad_();v=v.requires_grad_()
  f=field([[[.8,-2,-2],[1.2,2,2]]])
  def fun(pp,vv):return safety_loss(pp,vv,w,[f],recipe())["clearance"]
  self.assertTrue(torch.autograd.gradcheck(fun,(p,v),fast_mode=True,eps=1e-6,atol=1e-5,rtol=1e-4))
 def test_thin_gap_is_unknown_proxy(self):
  p,v,w=positions();p[:,:,:96,0]=0.;p[:,:,96:,0]=2.
  f=field([[[.99,-.2,-.2],[1.01,.2,.2]]])
  z=safety_loss(p,v,w,[f],recipe())
  self.assertIn("CHORD_UNPROVEN_PROXY_ONLY",z["reasons"][0]);self.assertFalse(z["certificate"])
 def test_corridor_target_detached_and_nonzero(self):
  p,v,w=positions();p.requires_grad_();target=p.detach().clone();target[...,1]=.5;target.requires_grad_()
  z=dict(target=target,radius=w*0,available=torch.ones(1,6,dtype=torch.bool))
  loss=corridor_loss(p,w,z);g,gt=torch.autograd.grad(loss.sum(),[p,target],allow_unused=True)
  self.assertIsNone(gt);self.assertTrue(bool((g[...,1]<0).all()))
 def test_corridor_no_teacher_stays_unknown(self):
  p,v,w=positions();z=dict(target=p,radius=w*0,available=torch.zeros(1,6,dtype=torch.bool))
  self.assertEqual(float(corridor_loss(p,w,z).sum()),0.)
 def test_teacher_reverification_and_balanced_assignment(self):
  c=torch.zeros(4,3,3,6,dtype=torch.float64);d=torch.full((4,3),.6,dtype=c.dtype)
  for j in range(3):c[:,j,0,0]=j*.6;c[:,j,0,1]=1
  state=torch.tensor([1.,0,0,0,0,0,3.,0,0,0,0,-1.],dtype=c.dtype)
  s=SimpleNamespace(sample_id="linear",teacher_coeffs=c,teacher_durations=d,state=state)
  p,v,w=positions();f=field([],True)
  z=TeacherGuidance().build([s],[f],[dict(queries=[0,1,2,3],teachers=[0,1,2,3])],p,w)
  self.assertEqual(set(z["assignments"][0]),set(range(4)));self.assertLessEqual(max(z["records"][0]["teacher_usage"]),2)
  self.assertFalse(z["certificate"])
 def test_bad_teacher_rejected(self):
  c=torch.zeros(1,3,3,6,dtype=torch.float64);d=torch.full((1,3),.6,dtype=c.dtype)
  c[:,:,0,1]=10
  s=SimpleNamespace(sample_id="bad",teacher_coeffs=c,teacher_durations=d,state=torch.zeros(12,dtype=c.dtype))
  p,v,w=positions()
  with self.assertRaises(ValueError):TeacherGuidance().build([s],[field([],True)],[dict(queries=[],teachers=[])],p,w)
 def test_near_guidance_survives_hinge_satisfaction(self):
  p=torch.tensor([[[2.35,0,0]]*6],dtype=torch.float64,requires_grad=True);zero=p*0
  z,_=guidance_loss(p,zero,zero,p.new_tensor([[2.4,0,0]]),torch.tensor([True]),torch.zeros(1,6,dtype=torch.bool))
  g=torch.autograd.grad(z.sum(),p)[0];self.assertGreater(float(z.sum()),0.);self.assertTrue(bool((g[...,0]<0).all()))
 def test_severe_collision_no_direct_goal_push(self):
  p=torch.tensor([[[1.,0,0]]*6],requires_grad=True);zero=p*0;severe=torch.ones(1,6,dtype=torch.bool)
  z,_=guidance_loss(p,zero,zero,p.new_tensor([[6.,0,0]]),torch.tensor([True]),severe)
  self.assertEqual(float(z.sum()),0.);self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(z.sum(),p)[0])),0)
 def test_far_does_not_demand_arrival(self):
  p=torch.tensor([[[2.,0,0]]*6],requires_grad=True);v=torch.ones_like(p);a=p*0
  z,diag=guidance_loss(p,v,a,p.new_tensor([[20.,0,0]]),torch.tensor([True]),torch.zeros(1,6,dtype=torch.bool))
  self.assertFalse(bool(diag["near"][0]));self.assertTrue(torch.isfinite(z).all())
 def test_control_integral_independent_and_gradcheck(self):
  t=tr();c=t.coeffs.requires_grad_();d=t.durations.requires_grad_()
  f=lambda cc,dd:control_loss(tr(cc,dd),recipe())[0]
  self.assertTrue(torch.autograd.gradcheck(f,(c,d),fast_mode=True,atol=1e-6,rtol=1e-4))
  _,q=control_loss(t,recipe())
  for k,o in [("acceleration",2),("jerk",3)]:np.testing.assert_allclose(q["integrals_si"][k].detach(),np_integral(c.detach().numpy(),d.detach().numpy(),o),atol=1e-11,rtol=1e-11)
 def test_time_gradient_and_physical_score_isolation(self):
  raw,state=raw_fixture();raw.requires_grad_();out=TrajectoryPostprocessor()(raw,state)
  p,v,w=segment_samples(out.trajectory,64);s=safety_loss(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]],dtype=p.dtype)],recipe())
  val=control_loss(out.trajectory,recipe())[0].mean()+s["clearance"].mean()
  g=torch.autograd.grad(val,raw)[0]
  self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g[...,15:19].norm()),0.);self.assertEqual(int(torch.count_nonzero(g[...,19:21])),0)
 def test_original_dynamic_limit_preserved(self):
  t=tr();a=analytic_samples(t);cfg=recipe()
  v,diag=feasibility_loss(a,t,a["p"].new_tensor([[2.4,0,0]]),a["p"].new_tensor([[[-1,-2,-2],[10,2,2]]]),cfg)
  old,terms=dynamic_limit_loss(a["v"],a["a"],a["jerk"],a["weights"])
  self.assertTrue(torch.equal(diag["limit"],old))
  for k in terms:self.assertTrue(torch.equal(terms[k],diag["limit_terms"][k]))
 def test_invalid_duration(self):
  t=tr();t.durations[0,0,0]=0
  with self.assertRaises(ValueError):control_loss(t,recipe())
 def test_ranking_all_failed_returns_none(self):
  raw=torch.zeros(1,6,21,requires_grad=True);labels=dict(passed=torch.zeros(1,6,dtype=torch.bool),clearance=torch.zeros(1,6))
  z,d=ranking_loss(raw,labels,torch.zeros(1,6),torch.zeros(1,6));g=torch.autograd.grad(z.sum(),raw)[0]
  self.assertEqual(d["targets"],[None]);self.assertEqual(d["oracle_restricted_selected"],[None])
  self.assertEqual(int(torch.count_nonzero(g[...,:20])),0);self.assertGreater(float(g[...,20].norm()),0.)
 def test_ranking_target_is_detached(self):
  raw=torch.zeros(1,6,21,requires_grad=True);q=torch.ones(1,6,requires_grad=True)
  labels=dict(passed=torch.tensor([[True,False,False,False,False,False]]),clearance=torch.ones(1,6,requires_grad=True))
  z,d=ranking_loss(raw,labels,q,q);g=torch.autograd.grad(z.sum(),[raw,q,labels["clearance"]],allow_unused=True)
  self.assertIsNone(g[1]);self.assertIsNone(g[2]);self.assertEqual(d["targets"],[0]);self.assertEqual(int(torch.count_nonzero(g[0][...,:19])),0)
 def test_stale_labels_100m_translation_rejected(self):
  raw,state=raw_fixture();out=TrajectoryPostprocessor()(raw,state)
  sample=SimpleNamespace(state=state[0]);f=field([],True,dtype=torch.float32)
  labels=hard_labels(out,[sample],[f]);verify_label_binding(labels,out,[sample],[f])
  out.trajectory.coeffs=out.trajectory.coeffs.clone()
  out.trajectory.coeffs[:,:,:,:,0]+=100
  with self.assertRaises(ValueError):verify_label_binding(labels,out,[sample],[f])
 def test_projection_first_order_only(self):
  a=torch.tensor([1.,0.]);b=torch.tensor([-2.,1.])
  q,on=safety_priority_projection(a,b,False);self.assertFalse(on);self.assertTrue(torch.equal(q,b))
  q,on=safety_priority_projection(a,b,True);self.assertTrue(on);self.assertGreaterEqual(float(a.dot(q)),-1e-6)
 def test_identity_digest_detects_any_change(self):
  a=torch.zeros(1,6,21);x=tensor_digest(a);a[0,0,19]=.1;self.assertNotEqual(x,tensor_digest(a))
 def test_flatness_hover_and_invalid_gravity(self):
  t=tr(torch.zeros(1,6,3,3,6,dtype=torch.float64));a=analytic_samples(t);cfg=recipe();cfg["flatness_enabled"]=True
  goals=a["p"].new_tensor([[2.4,0,0]]);bounds=a["p"].new_tensor([[[-1,-2,-2],[10,2,2]]])
  loss,diag=feasibility_loss(a,t,goals,bounds,cfg,a["p"].new_tensor([[0,0,-9.81]]))
  self.assertEqual(float(diag["flatness"].sum()),0.)
  self.assertTrue(torch.allclose(diag["thrust_acceleration"],torch.full_like(diag["thrust_acceleration"],9.81)))
  with self.assertRaises(ValueError):feasibility_loss(a,t,goals,bounds,cfg,a["p"].new_zeros(1,3))
 def test_safety_coeff_duration_gradcheck(self):
  t=tr();c=t.coeffs.requires_grad_();d=t.durations.requires_grad_()
  f=field([[[1.,-4,-4],[1.2,4,4]]])
  def fun(cc,dd):
   p,v,w=segment_samples(tr(cc,dd),64)
   return safety_loss(p,v,w,[f],recipe())["clearance"]
  self.assertTrue(torch.autograd.gradcheck(fun,(c,d),fast_mode=True,atol=1e-5,rtol=1e-4))
 def test_zero_quality_switches(self):
  cfg=recipe();cfg["quality_scales"]={k:0. for k in cfg["quality_scales"]}
  x=control_loss(tr(),cfg)[0]
  self.assertEqual(float(x.sum()),0.)
 def test_goal_disabled_by_authenticated_gate(self):
  p=torch.ones(1,6,3,requires_grad=True)
  val,_=guidance_loss(p,p,p,p.new_tensor([[2.4,0,0]]),torch.tensor([False]),torch.zeros(1,6,dtype=torch.bool))
  self.assertEqual(float(val.sum()),0.);self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(val.sum(),p)[0])),0)
 def test_lse_independent_numpy(self):
  p,v,w=positions();cfg=recipe()
  z=safety_loss(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]])],cfg)
  risk=z["point_risk"].numpy();alpha=w.numpy()/w.numpy().sum(-1,keepdims=True);tau=cfg["lse_temperature"]
  peak=risk.max(-1,keepdims=True)
  expected=peak[...,0]+tau*np.log((alpha*np.exp((risk-peak)/tau)).sum(-1))
  np.testing.assert_allclose(z["clearance"].detach(),expected,rtol=1e-12,atol=1e-12)
 def test_old_proof_invalidated_by_state_and_geometry(self):
  raw,state=raw_fixture();out=TrajectoryPostprocessor()(raw,state)
  sample=SimpleNamespace(state=state[0].clone());f=field([],True,dtype=torch.float32)
  labels=hard_labels(out,[sample],[f])
  sample.state[0]+=.1
  with self.assertRaises(ValueError):verify_label_binding(labels,out,[sample],[f])
 def test_no_training_dispatch_in_research_cli(self):
  from query_minco.training.pg2_guided_v1 import cli
  import inspect
  self.assertNotIn("optim",inspect.getsource(cli))

if __name__=="__main__":unittest.main()
