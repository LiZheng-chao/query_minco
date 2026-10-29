"""V2 math, rejection and isolated score contracts. No optimizer step."""
import math,copy,unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from test_pg1_v2_r1 import tr,raw_fixture,np_integral
from test_pg2_guided_v1 import field,positions
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.pg1.losses import goal_task_loss,analytic_samples,dynamic_limit_loss
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.initialization import head_rows,rows_unchanged
from query_minco.training.pg2_guided_v2.spec import *
from query_minco.training.pg2_guided_v2.losses import safety,buffer,goal,lateral,feasibility,control
from query_minco.training.pg2_guided_v2.ranking import losses as rankloss,metrics
from query_minco.training.pg2_guided_v2.gradients import calibrate,budget_record,guard_pass
from query_minco.training.pg2_guided_v2.execution import score_guard,gradient_score_guard,validate_training_spec,preflight_identity

def route(p):
 return dict(target=p.detach().clone(),radius=torch.zeros_like(p[...,0]),available=torch.ones(p.shape[:2],dtype=torch.bool))
def fake_model():
 data={"trajectory_head.output.weight":torch.zeros(21,2),"trajectory_head.output.bias":torch.zeros(21)}
 return SimpleNamespace(named_parameters=lambda:list(data.items()),parameters=lambda:list(data.values())),data
class V2(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_new_method_ids_and_constants(self):
  self.assertEqual(len(METHODS),6)
  for m in METHODS:
   c=recipe(m);self.assertEqual(validate(c),c);self.assertEqual(c["dynamic_limits"],[3.,8.,100.])
   c["margin"]=.02
   with self.assertRaises(ValueError):validate(c)
 def test_confirmed_vs_chord_no_retreat(self):
  p,v,w=positions();p[:,:,:96,0]=0.;p[:,:,96:,0]=2.
  z=safety(p,v,w,[field([[[.99,-.2,-.2],[1.01,.2,.2]]])],route(p),recipe())
  self.assertTrue(bool(z["chord_unproven"].all()));self.assertFalse(bool(z["confirmed"].any()))
  self.assertEqual(float(z["retreat"].sum()),0.)
  self.assertEqual(z["records"][0][0]["status"],"CHORD_UNPROVEN")
 def test_confirmed_with_anchor(self):
  p,v,w=positions();p.requires_grad_()
  z=safety(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]])],route(p),recipe())
  self.assertTrue(bool(z["confirmed"].all()));self.assertGreater(float(z["retreat"].sum()),0.)
  self.assertTrue(all(r["status"]=="CONFIRMED_VIOLATION" for r in z["records"][0]))
  g=torch.autograd.grad(z["retreat"].sum(),p)[0];self.assertGreater(float(g.norm()),0.)
 def test_start_invalid_not_safe(self):
  p,v,w=positions();p[...,0]=1
  z=safety(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]])],route(p),recipe())
  self.assertTrue(bool(z["confirmed"].all()));self.assertEqual(z["records"][0][0]["retreat"],"START_INVALID")
  self.assertEqual(float(z["retreat"].sum()),0.);self.assertFalse(z["certificate"])
 def test_near_start_invalid(self):
  p,v,w=positions();p[:,:,:1,0]=0.;p[:,:,1:,0]=1.
  z=safety(p,v,w,[field([[[.8,-.2,-.2],[1.2,.2,.2]]])],route(p),recipe())
  self.assertEqual(z["records"][0][0]["status"],"CONFIRMED_VIOLATION")
 def test_no_discrete_failure_not_certificate(self):
  p,v,w=positions()
  z=safety(p,v,w,[field([],True)],route(p),recipe())
  self.assertEqual(z["records"][0][0]["status"],"NO_DISCRETE_FAILURE");self.assertFalse(z["certificate"])
 def test_untrusted_empty_rejected(self):
  p,v,w=positions()
  with self.assertRaises(ValueError):safety(p,v,w,[field([],False)],route(p),recipe())
 def test_invalid_geometry_rejected(self):
  p,v,w=positions()
  with self.assertRaises(ValueError):safety(p,v,w,[field([[[1,1,1],[0,2,2]]])],route(p),recipe())
 def test_buffer_high_speed_is_capped_not_certified(self):
  v=torch.tensor([0.,.1,1.,10.,100.],dtype=torch.float64,requires_grad=True)
  b,required,exceeded=buffer(v,recipe())
  self.assertTrue(bool((b<=.15).all()));self.assertGreater(float(required[-1]),1000.);self.assertTrue(bool(exceeded[-1]))
  self.assertTrue(torch.isfinite(torch.autograd.grad(b.sum(),v)[0]).all())
 def test_tangential_narrow_pass_has_zero_braking_increment(self):
  p,v,w=positions()
  z=safety(p,v,w,[field([[[-1.,.3,-2.],[3.,1.,2.]],[[-1.,-1.,-2.],[3.,-.3,2.]]])],route(p),recipe())
  self.assertTrue(torch.allclose(z["braking"],torch.zeros_like(z["braking"]),atol=1e-12))
 def test_buffer_gradcheck(self):
  x=torch.tensor([.2,.7,2.],dtype=torch.float64,requires_grad=True)
  self.assertTrue(torch.autograd.gradcheck(lambda a:buffer(a,recipe())[0],(x,),fast_mode=True))
 def test_safe_coeff_duration_fd(self):
  t=tr();c=t.coeffs.requires_grad_();d=t.durations.requires_grad_();f=field([[[1,-4,-4],[1.2,4,4]]])
  from query_minco.training.losses.set_physical_risk import segment_samples
  def fn(cc,dd):
   p,v,w=segment_samples(tr(cc,dd),64);z=safety(p,v,w,[f],route(p),recipe())
   return z["clearance"]+z["braking"]
  self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True,atol=2e-5,rtol=2e-4))
 def test_original_near_task_exact(self):
  e=torch.tensor([[[1.,.2,0]]*6],dtype=torch.float64,requires_grad=True);v=e*0;a=e*0;goals=e.new_tensor([[2.4,0,0]])
  original,_=goal_task_loss(e,v,goals);r=route(e[:,:,None].expand(-1,-1,192,-1))
  task,dense,_=goal(e,v,a,goals,torch.tensor([True]),torch.zeros(1,6,dtype=torch.bool),r)
  self.assertTrue(torch.equal(task,original));self.assertGreater(float(dense.sum()),0.)
 def test_far_all_original_conditions_remain(self):
  for e in [[0.,0,0],[1.,2.,0],[-1.,0,0],[2.,0,0]]:
   p=torch.tensor([[e]*6],dtype=torch.float64);v=p*0;g=p.new_tensor([[6.,0,0]])
   old,_=goal_task_loss(p,v,g);r=route(p[:,:,None].expand(-1,-1,192,-1))
   task,_,_=goal(p,v,v,g,torch.tensor([True]),torch.zeros(1,6,dtype=torch.bool),r)
   self.assertTrue(torch.equal(old,task))
 def test_unsafe_goal_value_kept_gradient_masked(self):
  p=torch.ones(1,6,3,requires_grad=True);v=p*0;g=p.new_tensor([[2.4,0,0]])
  r=route(p[:,:,None].expand(-1,-1,192,-1));r["target"][:,:,-1]=0.
  task,dense,diag=goal(p,v,v,g,torch.tensor([True]),torch.ones(1,6,dtype=torch.bool),r)
  old,_=goal_task_loss(p,v,g)
  self.assertTrue(torch.equal(task,old))
  self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(task.sum(),p,retain_graph=True)[0])),0)
  self.assertGreater(float(torch.autograd.grad(dense.sum(),p)[0].sum()),0.)
 def test_teacher_targets_detached(self):
  p=torch.ones(1,6,3,requires_grad=True);t=torch.zeros(1,6,192,3,requires_grad=True)
  r=dict(target=t,radius=t[...,0]*0,available=torch.ones(1,6,dtype=torch.bool))
  task,dense,_=goal(p,p*0,p*0,p.new_tensor([[2.4,0,0]]),torch.tensor([True]),torch.ones(1,6,dtype=torch.bool),r)
  self.assertIsNone(torch.autograd.grad(dense.sum(),t,allow_unused=True)[0])
 def test_lateral_both_velocity_acceleration_gradients(self):
  v=torch.tensor([[[[1.,.2,0]]*4]*6],dtype=torch.float64,requires_grad=True)
  a=torch.tensor([[[[0.,9.,1.]]*4]*6],dtype=torch.float64,requires_grad=True)
  w=torch.ones(1,6,4,dtype=v.dtype)/4
  z,_=lateral(v,a,w);gv,ga=torch.autograd.grad(z.sum(),[v,a])
  self.assertGreater(float(gv.norm()),0.);self.assertGreater(float(ga.norm()),0.)
 def test_lateral_gradcheck(self):
  v=torch.tensor([[[[1.,.2,0]]*4]*6],dtype=torch.float64,requires_grad=True);a=torch.ones_like(v).detach()*9;a.requires_grad_()
  w=torch.ones(1,6,4,dtype=v.dtype)/4
  self.assertTrue(torch.autograd.gradcheck(lambda x,y:lateral(x,y,w)[0],(v,a),fast_mode=True,atol=1e-6))
 def test_low_speed_lateral_finite(self):
  for speed in (0.,1e-12,1e-6,.01):
   v=torch.full((1,6,4,3),speed,dtype=torch.float64,requires_grad=True);a=torch.full_like(v,9.).requires_grad_();w=torch.ones(1,6,4,dtype=v.dtype)/4
   z,_=lateral(v,a,w)
   self.assertTrue(all(torch.isfinite(g).all() for g in torch.autograd.grad(z.sum(),[v,a])))
 def test_dynamic_limit_exact(self):
  t=tr();a=analytic_samples(t);vals,diag=feasibility(a,a["p"].new_tensor([[[-1,-2,-2],[10,2,2]]]),recipe())
  old,terms=dynamic_limit_loss(a["v"],a["a"],a["jerk"],a["weights"])
  self.assertTrue(torch.equal(vals["dynamic_limit"],old))
  for k in terms:self.assertTrue(torch.equal(terms[k],diag["original_dynamic_terms"][k]))
 def test_quality_integrals_and_time_separate(self):
  t=tr();z=control(t)
  self.assertEqual(set(z["terms"]),{"acceleration","jerk","time"})
  for k,o in [("acceleration",2),("jerk",3)]:
   np.testing.assert_allclose(z["integrals_si"][k],np_integral(t.coeffs.numpy(),t.durations.numpy(),o),atol=1e-11)
 def test_quality_duration_fd(self):
  t=tr();c=t.coeffs.requires_grad_();d=t.durations.requires_grad_()
  self.assertTrue(torch.autograd.gradcheck(lambda cc,dd:sum(control(tr(cc,dd))["terms"].values()),(c,d),fast_mode=True,atol=1e-6))
 def test_time_compression_increases_energy(self):
  t=tr(d=torch.tensor([.4,.5,.6],dtype=torch.float64).expand(1,6,3).clone())
  scale=.9;compressed=tr(t.coeffs/scale**torch.arange(6,dtype=t.coeffs.dtype),t.durations*scale)
  a=control(t);b=control(compressed)
  self.assertTrue(bool((b["terms"]["time"]<a["terms"]["time"]).all()))
  self.assertTrue(bool((b["terms"]["acceleration"]>a["terms"]["acceleration"]).all()))
  self.assertTrue(bool((b["terms"]["jerk"]>a["terms"]["jerk"]).all()))
 def test_rank_prefers_safety_and_goal_then_quality(self):
  raw=torch.zeros(1,6,21,requires_grad=True);physical=torch.tensor([[True,True,True,False,False,False]]);goals=torch.tensor([[False,True,True,True,False,False]])
  quality=torch.tensor([[0.,3.,1.,0,0,0]],requires_grad=True)
  z,d=rankloss(raw,physical,goals,quality,quality,"rank_pg2_feasible_goal_quality")
  self.assertEqual(d["targets"],[2]);self.assertIsNone(torch.autograd.grad(z["ranking"].sum(),quality,allow_unused=True)[0])
 def test_pg2_all_invalid_none(self):
  raw=torch.zeros(1,6,21,requires_grad=True);physical=torch.ones(1,6,dtype=torch.bool);g=torch.zeros_like(physical)
  z,d=rankloss(raw,physical,g,torch.zeros(1,6),torch.zeros(1,6),"rank_pg2_feasible_goal_quality")
  self.assertEqual(d["targets"],[None]);self.assertGreater(float(z["validity"].sum()),0.)
 def test_yopo_literal_target_and_fallback(self):
  raw=torch.zeros(2,6,21);p=torch.tensor([[True,False,True,False,False,False],[False]*6]);q=torch.tensor([[3.,0,1,2,2,2],[4.,3,2,1,0,5.]])
  _,d=rankloss(raw,p,p,q,q,"rank_yopo_style")
  expected=(q+(q.max()+1)*(~p).float()).argmin(-1).tolist()
  self.assertEqual(d["targets"],expected);self.assertEqual(expected[1],4)
 def test_none_has_no_score_gradient(self):
  raw=torch.zeros(1,6,21,requires_grad=True);p=torch.ones(1,6,dtype=torch.bool);q=torch.ones(1,6)
  z,_=rankloss(raw,p,p,q,q,"rank_none")
  g=torch.autograd.grad(sum(z.values()).sum(),raw)[0];self.assertEqual(int(torch.count_nonzero(g)),0)
 def test_selector_metrics_do_not_oracle_filter(self):
  raw=torch.zeros(1,6,21);raw[0,0,19]=10.;p=torch.tensor([[False,True,False,False,False,False]])
  m=metrics(raw,p,p);self.assertTrue(m["oracle_at6"][0]);self.assertFalse(m["score_selected_at6"][0])
 def test_pg1_frozen_scores_and_v2_policy(self):
  model,data=fake_model();initial=head_rows(model)
  data["trajectory_head.output.weight"][19]=1.
  self.assertFalse(rows_unchanged(model,initial))
  self.assertTrue(score_guard(model,initial,FULL))
  with self.assertRaises(ValueError):score_guard(model,initial,BASELINE)
  with self.assertRaises(ValueError):score_guard(model,initial,PREFIX+"safe_v1")
 def test_yopo_raw20_stays_frozen(self):
  model,data=fake_model();initial=head_rows(model);data["trajectory_head.output.bias"][20]=1.
  with self.assertRaises(ValueError):score_guard(model,initial,PREFIX+"yopo_rank_v1")
 def test_no_unrecognized_method_can_change_scores(self):
  model,data=fake_model()
  with self.assertRaises(ValueError):score_guard(model,head_rows(model),"made_up")
 def test_draft_and_old_authorization_rejected(self):
  s=draft({}, {}, {},FULL,"A42_test")
  with self.assertRaises(ValueError):validate_training_spec(s,{})
 def test_preflight_identity_not_circular(self):
  s=dict(a=1,preflight=None);h=preflight_identity(s);s["preflight"]={"sha256":"abc"}
  self.assertEqual(h,preflight_identity(s));s["a"]=2;self.assertNotEqual(h,preflight_identity(s))
 def test_calibration_budget_and_fixed_old_scales(self):
  rr=[]
  for scale in (1.,2.,4.):
   norms=np.array([1.]+[scale*(i+1) for i in range(len(COMPONENTS)-1)],dtype=float)
   for k in FIXED:
    if k!="imitation":norms[COMPONENTS.index(k)]=0.
   rr.append(dict(components={k:dict(parameter_norm=float(n)) for k,n in zip(COMPONENTS,norms)},gram=np.diag(norms**2).tolist()))
  c=calibrate(rr);self.assertFalse(c["uses_development"]);self.assertTrue(c["gradient_guard_passed"])
  for k,v in FIXED.items():self.assertEqual(c["weights"][k],v)
  self.assertTrue(all(b["ratio"]<=2.00001 for b in c["budget_records"]))
 def test_guard_not_relaxed(self):
  self.assertFalse(guard_pass([11.,11.]));self.assertTrue(guard_pass([11.,1.,11.]))
 def test_m1_m4_balanced_teacher_reuse(self):
  for m in (1,4):
   c=torch.zeros(m,3,3,6,dtype=torch.float64);d=torch.full((m,3),.6,dtype=c.dtype)
   for j in range(3):c[:,j,0,0]=j*.6;c[:,j,0,1]=1
   s=SimpleNamespace(sample_id="x",teacher_coeffs=c,teacher_durations=d,state=torch.tensor([1.,0,0,0,0,0,3.,0,0,0,0,-1.],dtype=c.dtype))
   p,v,w=positions();z=TeacherGuidance().build([s],[field([],True)],[dict(queries=list(range(m)),teachers=list(range(m)))],p,w)
   self.assertEqual(set(z["assignments"][0]),set(range(m)));self.assertFalse(z["certificate"])
 def test_raw_time_and_physical_score_isolation(self):
  raw,state=raw_fixture();raw.requires_grad_();t=TrajectoryPostprocessor()(raw,state).trajectory
  z=control(t);g=torch.autograd.grad(sum(z["terms"].values()).sum(),raw)[0]
  self.assertGreater(float(g[...,15:19].norm()),0.);self.assertEqual(int(torch.count_nonzero(g[...,19:21])),0)
 def test_original_hard_verifier_sha_unchanged(self):
  from query_minco.training.run_spec import ROOT
  from query_minco.training.io import sha
  self.assertEqual(sha(ROOT/"query_minco/evaluation/validation.py"),"ccfa44f4129d6be811f040b270ac0bf6885b6a795c8a1fbbe3c674b3112d9abf")
 def test_baseline_dispatch_is_original_object(self):
  from query_minco.training.pg2_guided_v2.objective import forward
  sentinel={"loss":"original"}
  with patch("query_minco.training.pg2_guided_v2.objective.old_forward",return_value=sentinel) as old:
   self.assertIs(forward(None,[],None,BASELINE),sentinel);old.assert_called_once()
 def test_all_rank_modes_immutable_recipe_identity(self):
  self.assertEqual(recipe(PREFIX+"full_v1")["ranking"],"rank_pg2_feasible_goal_quality")
  self.assertEqual(recipe(PREFIX+"yopo_rank_v1")["ranking"],"rank_yopo_style")
  self.assertEqual(recipe(PREFIX+"no_rank_v1")["ranking"],"rank_none")
 def test_preflight_rejects_dirty_before_any_cuda(self):
  from query_minco.training.pg2_guided_v2.execution import preflight
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"spec.json";p.write_text(json.dumps(dict(schema="pg2_v2_pilot32_spec_v1",status="READY_FOR_MANUAL_PILOT",preflight=None,code=dict(allow_uncommitted_sources=True))))
   with patch("torch.cuda.mem_get_info",side_effect=AssertionError("GPU must not be reached")):
    with self.assertRaises(ValueError):preflight(p,Path(d)/"out")
 def test_quality_gradient_each_time_parameter(self):
  raw,state=raw_fixture();raw.requires_grad_();z=control(TrajectoryPostprocessor()(raw,state).trajectory)
  for name in ("acceleration","jerk","time"):
   g=torch.autograd.grad(z["terms"][name].sum(),raw,retain_graph=True)[0]
   self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g[...,15:19].norm()),0.)
 def test_goal_obstacle_conflict_is_gated(self):
  p=torch.tensor([[[1.,0,0]]*6],requires_grad=True);v=p*0;g=p.new_tensor([[2.4,0,0]])
  r=route(p[:,:,None].expand(-1,-1,192,-1));r["target"][:,:,-1,0]=.5
  task,dense,_=goal(p,v,v,g,torch.tensor([True]),torch.ones(1,6,dtype=torch.bool),r)
  direct=torch.autograd.grad(task.sum(),p,retain_graph=True)[0];repair=torch.autograd.grad(dense.sum(),p)[0]
  self.assertEqual(float(direct.norm()),0.);self.assertTrue(bool((repair[...,0]>0).all()))

 def test_flatness_optional_assumptions_and_hover(self):
  from query_minco.training.pg2_guided_v2.losses import flatness_diagnostic
  t=tr(torch.zeros(1,6,3,3,6,dtype=torch.float64));a=analytic_samples(t)
  g=a["p"].new_tensor([[2.4,0,0]]);b=a["p"].new_tensor([[[-1,-2,-2],[10,2,2]]]);gravity=a["p"].new_tensor([[0,0,-9.81]])
  args=dict(frame="fixed B0 FLU",assumptions=("zero_drag","fixed_yaw","specific_thrust","measured_gravity"))
  loss,diag=flatness_diagnostic(a,t,g,b,gravity,**args)
  self.assertEqual(float(loss.sum()),0.)
  with self.assertRaises(ValueError):flatness_diagnostic(a,t,g,b,gravity*0,**args)

if __name__=="__main__":unittest.main()
