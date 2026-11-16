import unittest,math
from pathlib import Path
import numpy as np,torch
from query_minco.integration.yopo_startup_v2.readiness import Readiness
from query_minco.training.peak_dynamics_proxy_v1.proxy import peak_proxy
from query_minco.evaluation.validation import norm_max
class Startup(unittest.TestCase):
 def inputs(self,r,omit=None,stamp=10.,phase='HELD_BEFORE_INTEGRATION'):
  values=dict(odom=dict(p=[0,0,2],v=[0,0,0]),imu={},depth={},request=[[0,0,2],[0,0,0],[0,0,0]],raw=dict(force=[0,0,.98*9.81]),
   plant=dict(phase=phase,integration_steps=0,final_command_stamp=0.,command_valid=False,mass=.98,gravity=9.81))
  for _ in range(5):
   for k,v in values.items():
    if k!=omit:r.observe(k,stamp,10.,v)
  return r
 def test_missing_depth(self):
  r=self.inputs(Readiness(True),'depth');self.assertEqual(r.tick(10),'WAIT');self.assertFalse(r.forward_allowed(10))
 def test_zero_imu_stamp(self):
  r=self.inputs(Readiness(True));r.observe('imu',0,10,{});self.assertEqual(r.tick(10),'ABORT')
 def test_no_request(self):
  r=self.inputs(Readiness(True),'request');self.assertFalse(r.forward_allowed(10))
 def test_no_authority_in_wait(self):
  r=self.inputs(Readiness(False));self.assertEqual(r.tick(10),'WAIT');self.assertFalse(r.forward_allowed(10))
 def test_raw_so3_not_execution_ack(self):
  r=self.inputs(Readiness(True));self.assertEqual(r.tick(10),'PREPARE_HOLD')
  self.assertNotEqual(r.mode,'PREPARED_HOLD_ACK')
 def test_ack_handshake(self):
  r=self.inputs(Readiness(True));r.tick(10)
  p=dict(phase='HELD_BEFORE_INTEGRATION',integration_steps=0,final_command_stamp=10.01,command_valid=True,mass=.98,gravity=9.81)
  for i in range(5):
   p['final_command_stamp']=10.001+i*.001;r.observe('plant',10.01,10.01,dict(p));r.tick(10.01)
  self.assertEqual(r.mode,'PREPARED_HOLD_ACK')
  p=dict(p,phase='INTEGRATING',integration_steps=1);r.observe('plant',10.02,10.02,p);self.assertEqual(r.tick(10.02),'ARMED_HOLD')
 def test_free_fall_before_prepare_is_rejected(self):
  r=self.inputs(Readiness(True),phase='INTEGRATING');self.assertEqual(r.tick(10),'ABORT')
  self.assertAlmostEqual(-9.81*.12,-1.1772);self.assertFalse(r.forward_allowed(10))
 def test_fault_latched(self):
  r=self.inputs(Readiness(True));r.observe('imu',0,10,{});self.inputs(r)
  self.assertEqual(r.tick(10),'ABORT');self.assertFalse(r.forward_allowed(10))
 def test_invalid_clock_and_late_raw_force(self):
  r=self.inputs(Readiness(True));r.tick(10);r.observe('raw',10.01,10.01,dict(force=[float('nan'),0,9.6]));self.assertFalse(r.forward_allowed(10.01))
  r=self.inputs(Readiness(True));self.assertEqual(r.tick(float('nan')),'ABORT')
 def test_stale_nan_nonhold(self):
  r=self.inputs(Readiness(True));r.tick(10);self.assertEqual(r.tick(10.3),'ABORT')
  r=self.inputs(Readiness(True));r.observe('raw',float('nan'),10,{});self.assertEqual(r.tick(10),'ABORT')
  r=self.inputs(Readiness(True));r.observe('request',10,10,[[0,0,2],[.1,0,0],[0,0,0]]);self.assertEqual(r.tick(10),'ABORT')
class Peak(unittest.TestCase):
 def data(self):
  g=torch.Generator().manual_seed(203);c=torch.randn(1,1,3,3,6,generator=g,dtype=torch.float64)*.2
  return c,torch.full((1,1,3),2/3,dtype=torch.float64)
 def test_upper_and_gap_bound_and_depth(self):
  c,d=self.data();prev=None
  for depth in [0,2,4,6]:
   p=peak_proxy(c,d,depth)
   for order,key,L in [(1,'speed',3),(2,'acceleration',8),(3,'jerk',100)]:
    true=max(norm_max(x,2/3,order) for x in c[0,0].numpy())/L;u=float(p['upper_ratios'][key]);gap=float(p['error_bounds_ratio'][key])
    self.assertGreaterEqual(u+1e-12,true);self.assertLessEqual(u-true,gap+1e-10)
    if prev:self.assertLessEqual(u,float(prev['upper_ratios'][key])+1e-12)
   prev=p
 def test_fd_fp64_coefficients(self):
  c,d=self.data();c=(c*12).requires_grad_();p=peak_proxy(c,d)['F_peak'].sum();g=torch.autograd.grad(p,c)[0];v=g/g.norm();e=1e-5
  fd=(peak_proxy(c.detach()+e*v,d)['F_peak']-peak_proxy(c.detach()-e*v,d)['F_peak'])/(2*e)
  self.assertAlmostEqual(float(fd),float((g*v).sum()),places=6)
 def test_k1_k6_mean_and_gradient_normalization(self):
  c,d=self.data();a=(c*12).requires_grad_();x=peak_proxy(a,d)['F_peak'].mean();ga=torch.autograd.grad(x,a)[0]
  b=a.detach().repeat(1,6,1,1,1).requires_grad_();y=peak_proxy(b,d.repeat(1,6,1))['F_peak'].mean();gb=torch.autograd.grad(y,b)[0]
  self.assertAlmostEqual(float(x),float(y),places=12);torch.testing.assert_close(ga,gb.sum(1,keepdim=True));torch.testing.assert_close(gb[:,0:1]*6,ga)
 def test_zero_boundary_tie_finite(self):
  for speed in [0.,3.,3.2]:
   c=torch.zeros((1,1,3,3,6),dtype=torch.float64);c[:,:,:,0,1]=speed;c.requires_grad_();d=torch.full((1,1,3),2/3,dtype=torch.float64)
   p=peak_proxy(c,d);g=torch.autograd.grad(p['F_peak'].sum(),c)[0];self.assertTrue(torch.isfinite(g).all())
   self.assertAlmostEqual(float(p['F_peak']),max(speed/3-1,0)**2,places=12)
 def test_invalid_contract(self):
  c,d=self.data()
  for bad in [d*0,-d,d*float('nan'),d*float('inf'),d[0]]:
   with self.assertRaises(ValueError):peak_proxy(c,bad)
  with self.assertRaises(ValueError):peak_proxy(c.repeat(1,2,1,1,1),d.repeat(1,2,1))
  with self.assertRaises(ValueError):peak_proxy(c*float('nan'),d)
  with self.assertRaises(ValueError):peak_proxy(c,d,epsilon_ratio=float('nan'))
 def test_fp32_fp64_same_coefficients(self):
  c,d=self.data();c=c.float()*12;d=d.float();a=peak_proxy(c,d);b=peak_proxy(c,d,accumulation=torch.float32)
  for k in a['upper_ratios']:self.assertLess(abs(float(a['upper_ratios'][k])-float(b['upper_ratios'][k])),1e-5)
if __name__=='__main__':unittest.main(verbosity=2)
