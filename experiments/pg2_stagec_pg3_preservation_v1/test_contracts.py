import unittest
import numpy as np
import torch
from types import SimpleNamespace
from query_minco.integration.pg2_stagec_v1.lease import Arbiter,Packet,State,Gates
from query_minco.integration.pg2_stagec_v1.reference import Reference,controller_semantics
from query_minco.integration.pg2_stagec_v1.volume import NativeVolume
from query_minco.training.pg3_preservation_v1.mechanism import local_direction,witness,retained,superior

class LeaseTests(unittest.TestCase):
 def armed(self):
  a=Arbiter();a.shadow(1);a.receive(Packet('query',1,1,1,2,'abc'),1);return a
 def test_cold_start(self):
  a=Arbiter();self.assertEqual(a.tick(1),State.DISARMED)
  with self.assertRaises(ValueError):a.receive(Packet('q',1,1,1,2,'a'),1)
 def test_expiry_latched(self):
  a=self.armed();self.assertEqual(a.tick(1.101),State.REFERENCE_EXPIRED)
  with self.assertRaises(ValueError):a.receive(Packet('query',2,2,1.11,2,'b'),1.11)
 def test_duplicate_and_old(self):
  a=self.armed()
  for p,t in [(Packet('query',1,2,1,2,'a'),1.01),(Packet('query',2,2,1,2,'a'),1.11)]:
   with self.assertRaises(ValueError):a.receive(p,t)
 def test_clock_reset_clears(self):
  a=self.armed()
  with self.assertRaises(ValueError):a.tick(.5)
  self.assertIsNone(a.packet);self.assertEqual(a.state,State.ABORT)
  with self.assertRaises(ValueError):a.receive(Packet('query',2,2,1,2,'b'),1)
 def test_owner_conflict_latched(self):
  a=self.armed()
  with self.assertRaises(ValueError):a.receive(Packet('other',2,2,1,2,'a'),1.01)
  self.assertEqual(a.state,State.ABORT)
  with self.assertRaises(ValueError):a.shadow(1.02)
 def test_persistent_hover_not_retimed(self):
  a=Arbiter();a.shadow(1);a.receive(Packet('q',1,1,1,2,'h','PERSISTENT_HOVER'),1)
  a.receive(Packet('q',1,2,1,2,'h','PERSISTENT_HOVER'),1.08)
  a.receive(Packet('q',1,3,1,2,'h','PERSISTENT_HOVER'),1.16)
  self.assertEqual(a.packet.source_stamp,1)
  with self.assertRaises(ValueError):a.receive(Packet('q',1,4,1,3,'h','PERSISTENT_HOVER'),1.17)
 def test_no_fake_authority(self):
  a=self.armed()
  with self.assertRaises(ValueError):a.request_authority(Gates(),1)
  self.assertFalse(a.query_permission)
 def test_timer_not_physical_fallback(self):
  a=self.armed();a.tick(1.11);a.request_fallback(1.12);self.assertEqual(a.tick(1.2),State.FALLBACK_REQUESTED)
  for evidence in [{},{'scope':'ANALYTIC_FIXTURE'},{'scope':'QUALIFIED_SENSOR_SIMULATION','real_simulation_execution':True}]:
   with self.assertRaises(ValueError):a.verify_fallback(evidence,1.2)
  self.assertNotEqual(a.state,State.FALLBACK_VERIFIED)
 def test_bad_time_and_nonfinite(self):
  a=Arbiter();a.shadow(1)
  for p in [Packet('q',1,1,0,2,'a'),Packet('q',1,1,1,0,'a'),Packet('q',1,1,float('nan'),2,'a')]:
   with self.assertRaises(ValueError):a.receive(p,1)

class ReferenceTests(unittest.TestCase):
 def ref(self):
  c=np.zeros((3,3,6));c[:,0,1]=1;c[:,0,0]=[0,1,2]
  R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
  return Reference(c,np.ones(3),np.array([1.,2,3]),R,10.,1)
 def test_fixed_frame_no_gravity(self):
  r=self.ref();np.testing.assert_allclose(r.pva(10.5),[[1,2.5,3],[0,1,0],[0,0,0]])
 def test_seams_and_boundaries(self):
  r=self.ref();self.assertEqual(r.seam_residuals(),[0,0,0])
  for t in [10,11,12,13]:np.testing.assert_allclose(r.pva(t)[0],[1,t-8,3])
 def test_expired_or_future(self):
  r=self.ref()
  for t in [9.999,13.001,float('nan')]:
   with self.assertRaises(ValueError):r.pva(t)
 def test_splice(self):
  r=self.ref();np.testing.assert_allclose(r.splice_residuals(r,11),0)
  c=r.coefficients.copy();c[0,0,0]+=1
  other=Reference(c,r.durations,r.origin,r.rotation,10,2)
  self.assertGreater(r.splice_residuals(other,10)[0],.9)
 def test_bad_solve_and_immutability(self):
  r=self.ref()
  with self.assertRaises(ValueError):r.coefficients[0,0,0]=3
  with self.assertRaises(ValueError):Reference(np.full((3,3,6),np.nan),np.ones(3),np.zeros(3),np.eye(3),10,1)
 def test_physical_derivatives_finite_difference(self):
  rng=np.random.RandomState(42);c=rng.randn(3,3,6)*.02;r=Reference(c,np.ones(3),np.array([1,2,3.]),np.eye(3),10,1);t=10.4;h=1e-4
  a=r.pva(t-h);b=r.pva(t);d=r.pva(t+h)
  np.testing.assert_allclose((d[0]-a[0])/(2*h),b[1],atol=1e-8,rtol=1e-6)
  np.testing.assert_allclose((d[1]-a[1])/(2*h),b[2],atol=1e-8,rtol=1e-6)
 def test_controller_ready(self):self.assertEqual(controller_semantics(1),'ACCELERATION_ONLY')

class VolumeTests(unittest.TestCase):
 def cert(self,v,now):return v.certificate(np.zeros((3,3,6)),np.ones(3)*.1,now)
 def test_empty_and_invalid_depth(self):
  v=NativeVolume();self.assertEqual(self.cert(v,1)['status'],'UNKNOWN')
  for value in [float('nan'),float('inf')]:
   with self.assertRaises(ValueError):v.add(1,np.full((2,128,160),value,dtype=np.float32),np.zeros(3),np.eye(3))
 def test_maxrange_edges_occlusion_gaps_stay_unknown(self):
  for kind in ['maxrange','unknown','edge','gap','fov']:
   d=np.ones((2,128,160),dtype=np.float32);d[0]*=.5
   if kind=='maxrange':d[:]=0 # 20m raw depth decodes to invalid; separately tested below
   if kind=='unknown':d[:]=0
   if kind=='edge':d[0,:,79]=.05
   if kind=='gap':d[:,::2]=0
   v=NativeVolume();v.add(1,d,np.zeros(3),np.eye(3));z=self.cert(v,1)
   self.assertEqual(z['status'],'UNKNOWN');self.assertFalse(z['online_permission']);self.assertEqual(z['braking_space'],'UNPROVEN')
 def test_stale_and_reset(self):
  v=NativeVolume();d=np.zeros((2,128,160),np.float32);v.add(1,d,np.zeros(3),np.eye(3))
  self.assertEqual(self.cert(v,2)['reason'],'NO_FRESH_DEPTH')
  with self.assertRaises(ValueError):v.add(.5,d,np.zeros(3),np.eye(3))
  self.assertEqual(len(v.history.frames),0)
 def test_gt_injection_forbidden(self):
  with self.assertRaises(ValueError):NativeVolume().certify_external('GT')

class MechanismTests(unittest.TestCase):
 def test_norm_mask_and_frozen_shared(self):
  p=SimpleNamespace(named=[('shared',torch.zeros(3)),('query_decoder.learned_queries',torch.zeros(6,128))],parts=[torch.ones(3,dtype=torch.bool),torch.ones((6,128),dtype=torch.bool)])
  a=torch.arange(771,dtype=torch.float64)+1;d,info=local_direction(p,a,[2,2]);self.assertAlmostEqual(float(d.norm()),float(a.norm()),places=9);self.assertEqual(int((d!=0).sum()),128);self.assertTrue(torch.equal(d[:3],torch.zeros(3,dtype=d.dtype)))
  with self.assertRaises(ValueError):local_direction(p,a,[])
  with self.assertRaises(ValueError):local_direction(p,torch.zeros_like(a),[2])
 def test_mode_witness_and_unknown_individual(self):
  p=[[True,True,False,False,False,False]];g=[[True]*6];s=[['right','right',None,None,None,None]];w=witness(p,g,s)
  self.assertTrue(retained(w,[[False,True,False,False,False,False]],g,s));self.assertFalse(retained(w,[[False]*6],g,s))
  w=witness(p,g,[[None,'right',None,None,None,None]])
  self.assertFalse(retained(w,[[False,True,False,False,False,False]],g,s))
 def test_zero_or_cosine_not_success(self):
  c=dict(accepted=True,relative_update_norm=0,pre_fail_to_pass=2,pre_safe_pass_to_fail=0);b=dict(c,relative_update_norm=1,pre_fail_to_pass=1)
  self.assertFalse(superior(c,b));c['relative_update_norm']=1;self.assertTrue(superior(c,b));c['pre_safe_pass_to_fail']=1;self.assertFalse(superior(c,b))

class DomainAndSweepTests(unittest.TestCase):
 def test_depth_raw_invalid_never_free(self):
  from query_minco.inference.shadow import decode_image
  for value in [0.,20.,21.,float('nan'),float('inf'),-1.]:
   raw=np.full((128,160),value,dtype='<f4');d=decode_image(raw.tobytes(),'32FC1',160,128,640,False)
   self.assertEqual(int(d[1].sum()),0);v=NativeVolume();v.add(1,d,np.zeros(3),np.eye(3))
   z=v.certificate(np.zeros((3,3,6)),np.ones(3)*.1,1);self.assertEqual(z['status'],'UNKNOWN')
 def test_operating_domain_no_threshold_relaxation(self):
  from query_minco.inference.shadow import make_state
  _,diag=make_state(np.zeros(3),np.eye(3),np.array([.8,0,0]),np.zeros(3),np.array([10,0,0]));self.assertEqual(diag['ood'],[])
  _,diag=make_state(np.zeros(3),np.eye(3),np.array([6.,0,0]),np.zeros(3),np.array([10,0,0]));self.assertIn('VELOCITY_NOT_F1_POINT8',diag['ood'])
  for goal in [[1,0,0],[-10,0,0],[10,5,0]]:
   with self.assertRaises(ValueError):make_state(np.zeros(3),np.eye(3),np.array([.8,0,0]),np.zeros(3),np.array(goal))
 def test_swept_body_fixture_scope_only(self):
  from query_minco.inference.observed_space import FreeVolume,Scope,box_cell,ErrorBudget
  c=np.zeros((3,3,6));d=np.ones(3)*.1
  err=ErrorBudget(body_radius=.12,margin=.03,tracking=0,translation=0,extrinsic=0,rotation_rad=0,depth_reserve=0)
  vol=FreeVolume((box_cell([-1,-1,-1],[1,1,1],10,'UNIT_ONLY'),),Scope.ANALYTIC_FIXTURE,err)
  z=vol.certify(c,d,1,required_until=1.3);self.assertEqual(z['status'],'CERTIFIED_FREE');self.assertFalse(z['selected_permission'])
  # Every center stays inside, but the full ball crosses a wall.
  c[:,0,0]=.9;self.assertEqual(vol.certify(c,d,1,required_until=1.3)['status'],'UNKNOWN')
  c[:]=0;self.assertEqual(vol.certify(c,d,9.9,required_until=10.2)['status'],'UNKNOWN')
 def test_midsegment_excursion_not_endpoint_sampling(self):
  from query_minco.inference.observed_space import FreeVolume,Scope,box_cell,ErrorBudget
  c=np.zeros((3,3,6));c[0,0,1]=8;c[0,0,2]=-8 # endpoints0, midpoint2m
  vol=FreeVolume((box_cell([-1]*3,[1]*3,10,'UNIT_ONLY'),),Scope.ANALYTIC_FIXTURE,ErrorBudget())
  self.assertEqual(vol.certify(c,np.ones(3),1,required_until=4)['status'],'UNKNOWN')

if __name__=='__main__':unittest.main()
