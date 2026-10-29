"""Region geometry, mode binding, gradients and historical contract tests."""
import unittest,copy
import numpy as np
import torch
from .feasible_section import *
from .spec import recipe
from query_minco.trajectory import TrajectoryPostprocessor
class RegionTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def fixture(self,axis=1,sign=1):
  scene=dict(obstacles=[dict(name="central",min=[.9,-.2,-.2],max=[1.1,.2,.2])],flight_center_domain_m=[[0,-1,-1],[3,1,1]])
  point=[1.,0.,0.];point[axis]=sign*.7
  label=dict(valid=True,box=0,plane_x=1.,position=point,normal=[1.,0.,0.],progress=.5,side=[axis,sign])
  teacher=dict(teacher=0,cache_identity="synthetic_teacher0",certified=True,representable=True,bottlenecks=[label])
  return teacher,label,scene
 def test_1d_interval(self):
  u=torch.tensor([[0.],[2.]],requires_grad=True);A=torch.tensor([[1.],[-1.]]);b=torch.tensor([1.,1.]);v=halfspace_loss(u,A,b);self.assertEqual(float(v[0]),0.);self.assertAlmostEqual(float(v[1]),.5)
 def test_2d_inside_zero_gradient(self):
  t,l,s=self.fixture();proof,_=build_region(t,l,s);u=torch.tensor([[.6,.8]],requires_grad=True);v=halfspace_loss(u,torch.tensor(proof["A"]),torch.tensor(proof["b"]));v.sum().backward()
  self.assertEqual(float(v),0.);self.assertFalse(torch.count_nonzero(u.grad))
 def test_left_right_up_down(self):
  for axis in (1,2):
   for sign in (-1,1):
    t,l,s=self.fixture(axis,sign);p,_=build_region(t,l,s);self.assertTrue(verify_region(p,t,l,s))
    u=torch.zeros(1,2,requires_grad=True);v=halfspace_loss(u,torch.tensor(p["A"]),torch.tensor(p["b"]));v.backward()
    self.assertLess(float(u.grad[0,axis-1])*sign,0)
 def test_boundaries_and_labels_stop_gradient(self):
  u=torch.zeros(1,2,requires_grad=True);A=torch.eye(2,requires_grad=True);b=torch.full((2,),-.1,requires_grad=True);v=halfspace_loss(u,A,b).sum()
  a,bg=torch.autograd.grad(v,(A,b),allow_unused=True);self.assertIsNone(a);self.assertIsNone(bg)
 def test_disconnected_sides_not_merged(self):
  t,l,s=self.fixture(1,1);p,_=build_region(t,l,s);tt,ll,_=self.fixture(1,-1);tt["teacher"]=1;tt["cache_identity"]="teacher1";q,_=build_region(tt,ll,s)
  self.assertGreater(p["lower"][0],.35);self.assertLess(q["upper"][0],-.35)
  bad=copy.deepcopy(p);bad["lower"][0]=-1.;bad["b"][1]=1.;self.assertFalse(verify_region(bad,t,l,s))
 def test_multiple_obstacles(self):
  t,l,s=self.fixture();l["position"][2]=-.7;s["obstacles"].append(dict(name="upper",min=[.8,-1,.2],max=[1.2,1,.4]))
  p,_=build_region(t,l,s);self.assertTrue(verify_region(p,t,l,s));self.assertLess(p["upper"][1],.05)
 def test_missing_side_explicit(self):
  t,l,s=self.fixture();l["side"]=None;p,reason=build_region(t,l,s);self.assertIsNone(p);self.assertIn("SIDE",reason)
 def test_unrepresentable_teacher(self):
  t,l,s=self.fixture();t["representable"]=False;p,reason=build_region(t,l,s);self.assertIsNone(p);self.assertEqual(reason,"TEACHER_UNREPRESENTABLE")
 def test_uncertified_rejected(self):
  t,l,s=self.fixture();t["certified"]=False;self.assertIsNone(build_region(t,l,s)[0])
 def test_illegal_geometry_rejected(self):
  t,l,s=self.fixture();s["obstacles"][0]["min"][1]=float("nan")
  with self.assertRaises(ValueError):build_region(t,l,s)
 def test_stale_proof_and_mode_reuse_rejected(self):
  t,l,s=self.fixture();p,_=build_region(t,l,s);tt=copy.deepcopy(t);tt["teacher"]=2;self.assertFalse(verify_region(p,tt,l,s))
  ss=copy.deepcopy(s);ss["obstacles"][0]["max"][1]=.8;self.assertFalse(verify_region(p,t,l,ss))
 def test_point_outside_region_covered_by_obstacle_unavailable(self):
  t,l,s=self.fixture();s["obstacles"].append(dict(name="cover",min=[.8,.5,-.2],max=[1.2,.9,.2]))
  p,status=build_region(t,l,s);self.assertIsNone(p);self.assertIn("NO_CONSERVATIVE",status)
 def test_invalid_region_loss_zero_not_fake_label(self):
  t,l,s=self.fixture();t["representable"]=False;c=torch.zeros(3,3,6,requires_grad=True);d=torch.tensor([.7,.7,.7],requires_grad=True)
  v,diag=RegionBottleneck()(c,d,t,dict(scene=s,empty_certified=False));self.assertEqual(float(v),0);self.assertEqual(diag["count"],0)
 def test_coeff_duration_finite_differences(self):
  t,l,s=self.fixture();c=torch.zeros(3,3,6,dtype=torch.float64);c[:,1,0]=.05;c[:,1,1]=.2;c[:,2,0]=1.3;c.requires_grad_();d=torch.tensor([.7,.7,.7],dtype=torch.float64,requires_grad=True);policy=RegionBottleneck();bank=dict(scene=s,empty_certified=False)
  v,_=policy(c,d,t,bank);gc,gd=torch.autograd.grad(v,(c,d))
  for idx in [(1,1,0),(1,1,1),(1,2,0)]:
   a=c.detach().clone();b=a.clone();a[idx]+=1e-6;b[idx]-=1e-6;fd=(policy(a,d.detach(),t,bank)[0]-policy(b,d.detach(),t,bank)[0])/2e-6
   self.assertAlmostEqual(float(fd),float(gc[idx]),places=6)
  for idx in range(3):
   a=d.detach().clone();b=a.clone();a[idx]+=1e-6;b[idx]-=1e-6;fd=(policy(c.detach(),a,t,bank)[0]-policy(c.detach(),b,t,bank)[0])/2e-6
   self.assertAlmostEqual(float(fd),float(gd[idx]),places=6)
 def test_raw_chain_and_score_zero(self):
  t,l,s=self.fixture();raw=torch.zeros(1,6,21,requires_grad=True);state=torch.tensor([[.8,0,0,0,0,0,2.4,0,0,0,0,-1.]])
  out=TrajectoryPostprocessor()(raw,state);v,_=RegionBottleneck()(out.trajectory.coeffs[0,0],out.parameters.durations[0,0],t,dict(scene=s,empty_certified=False));v.backward()
  self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad[0,0,:6].norm()),0);self.assertFalse(torch.count_nonzero(raw.grad[...,19:21]))
if __name__=="__main__":unittest.main()
