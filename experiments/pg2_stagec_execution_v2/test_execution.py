import copy,unittest,math
import numpy as np
from query_minco.integration.stagec_execution_v2.reference import Lease,brake,evaluate,extrema,pva
from query_minco.integration.stagec_execution_v2.renderer_volume import NativeModel,RendererHistory
from query_minco.inference.observed_space import Scope,ErrorBudget

def msg(stamp=10.):return dict(owner='owner',sequence=1,heartbeat=1,plan=dict(source_stamp=stamp,coefficients=np.zeros((1,3,6)).tolist(),durations=[2.]))
class ReferenceTests(unittest.TestCase):
 def test_brake_start_end_and_bounds(self):
  for v in [.0,.4,.8]:
   z=brake(np.array([0.,0,2]),np.array([v,0,0]),np.zeros(3));c=z['coefficients'][0];T=z['durations'][0]
   np.testing.assert_allclose(evaluate(c,0),[0,0,2],atol=1e-12);np.testing.assert_allclose(evaluate(c,0,1),[v,0,0],atol=1e-12);np.testing.assert_allclose(evaluate(c,T,1),0,atol=1e-12);np.testing.assert_allclose(evaluate(c,T,2),0,atol=1e-12);self.assertLessEqual(z['max_reference_a'],.8);self.assertLessEqual(z['max_reference_j'],2.)
 def test_nonzero_initial_acceleration(self):
  a=np.array([.1,.02,0]);z=brake(np.array([0,0,2]),np.array([.4,0,0]),a);np.testing.assert_allclose(evaluate(z['coefficients'][0],0,2),a)
 def test_mixed_degree_tiny_axis_motion(self):
  z=brake(np.array([1.4,0,2]),np.array([.4007,7.6e-9,4.5e-7]),np.array([.006,-1.5e-7,-4.6e-6]));self.assertLessEqual(z['max_reference_a'],.8);self.assertLessEqual(z['max_reference_j'],2.)
 def test_finite_difference(self):
  z=brake(np.array([0,0,2]),np.array([.8,0,0]),np.zeros(3));c=z['coefficients'][0];h=1e-5;t=.5
  np.testing.assert_allclose((evaluate(c,t+h)-evaluate(c,t-h))/(2*h),evaluate(c,t,1),atol=1e-9)
 def test_no_unbounded_or_nonfinite_brake(self):
  with self.assertRaises(ValueError):brake([0,0,2],[100,0,0],[0,0,0])
  with self.assertRaises(ValueError):brake([0,0,2],[float('nan'),0,0],[0,0,0])
 def test_physical_time_expiration(self):
  p=msg()['plan'];np.testing.assert_equal(pva(p,11),0)
  for t in [9.,13.,float('nan')]:
   with self.assertRaises(ValueError):pva(p,t)
class LeaseTests(unittest.TestCase):
 def test_initial_and_immutable_heartbeat(self):
  a=Lease('owner');m=msg();a.receive(m,10,1);m['heartbeat']=2;a.receive(m,10.5,1.02);self.assertEqual(a.plan['source_stamp'],10.);self.assertIsNone(a.fault)
 def test_changed_same_sequence(self):
  a=Lease('owner');m=msg();a.receive(m,10,1);m['heartbeat']=2;m['plan']['source_stamp']=10.01
  with self.assertRaises(ValueError):a.receive(m,10.02,1.02)
  self.assertIsNotNone(a.fault)
 def test_duplicate_does_not_refresh(self):
  a=Lease('owner');m=msg();a.receive(m,10,1)
  with self.assertRaises(ValueError):a.receive(m,10.01,1.01)
  self.assertEqual(a.rx,1)
 def test_stale_future_expired(self):
  for stamp in [8.,11.,1.]:
   a=Lease('owner')
   with self.assertRaises(ValueError):a.receive(msg(stamp),10,1)
   self.assertIsNotNone(a.fault)
 def test_owner_conflict_and_latch(self):
  a=Lease('owner');m=msg();m['owner']='other'
  with self.assertRaises(ValueError):a.receive(m,10,1)
  with self.assertRaises(ValueError):a.receive(msg(),10,1)
 def test_clock_rollback(self):
  a=Lease('owner');a.receive(msg(),10,1)
  with self.assertRaisesRegex(ValueError,'CLOCK_RESET'):a.check(9.,1.01)
 def test_publisher_loss_or_timeout(self):
  for now in [10.11,11.]:
   a=Lease('owner');a.receive(msg(),10,1)
   with self.assertRaisesRegex(ValueError,'RECEIPT_TIMEOUT'):a.check(now,1.11)
 def test_expiry_despite_fresh_heartbeat(self):
  a=Lease('owner');a.receive(msg(),10,1)
  with self.assertRaisesRegex(ValueError,'EXPIRED'):a.check(12.1,1.01)
 def test_malformed_and_nonfinite(self):
  for m in [{},dict(msg(),heartbeat=True),dict(msg(),plan={'source_stamp':10,'coefficients':[[[float('nan')]]],'durations':[1]})]:
   a=Lease('owner')
   with self.assertRaises(ValueError):a.receive(m,10,1)
   self.assertIsNotNone(a.fault)
class CoverageTests(unittest.TestCase):
 def test_full_resolution_vs_query_cover(self):
  self.assertTrue(NativeModel().proof()['covered']);self.assertFalse(NativeModel(160,128,100,100,79.5,63.5).proof()['covered'])
 def test_unknown_invalid_maxrange(self):
  c=np.zeros((3,3,6));c[:,:,0]=[1,0,2];d=np.ones(3)*.03
  for val in [float('nan'),float('inf'),0.,20.]:
   h=RendererHistory();h.add(1,np.full((512,800),val),np.array([0,0,2]),np.eye(3));z=h.certify(c,d,1);self.assertEqual(z['status'],'UNKNOWN');self.assertFalse(z['online_permission'])
 def test_uniform_prefix_positive_is_conditional_only(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3));c=np.zeros((3,3,6));c[:,:,0]=[1,0,2];z=h.certify(c,np.ones(3)*.03,1)
  self.assertEqual(z['status'],'CERTIFIED_FREE');self.assertEqual(z['scope'],Scope.RENDERER_VOXELS.value);self.assertFalse(z['online_permission']);self.assertFalse(z['selected_permission']);self.assertIn('UNQUALIFIED',z['tracking_localization_latency_braking_bounds'])
 def test_fov_and_behind_and_occlusion(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3))
  for point in [[-1,0,2],[1,2,2],[4,0,2],[1,0,4]]:
   c=np.zeros((3,3,6));c[:,:,0]=point;self.assertEqual(h.certify(c,np.ones(3)*.03,1)['status'],'UNKNOWN')
 def test_rollback_and_stale_and_full_horizon(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3));c=np.zeros((3,3,6));c[:,:,0]=[1,0,2]
  self.assertEqual(h.certify(c,np.ones(3),1)['status'],'UNKNOWN');self.assertEqual(h.certify(c,np.ones(3)*.03,2)['status'],'UNKNOWN')
  with self.assertRaises(ValueError):h.add(.5,np.ones((512,800)),np.array([0,0,2]),np.eye(3))
  self.assertFalse(h.frames)
 def test_zero_uncertainty_and_clock_reset_refused(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3));c=np.zeros((3,3,6));c[:,:,0]=[1,0,2]
  with self.assertRaises(ValueError):h.certify(c,np.ones(3)*.03,1,ErrorBudget(tracking=0))
  self.assertEqual(h.certify(c,np.ones(3)*.03,.5)['status'],'UNKNOWN');self.assertFalse(h.frames)
 def test_query_clock_rollback_clears_even_after_source_stamp(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3));h.volume(1.2);h.volume(1.1);self.assertFalse(h.frames)
 def test_declared_depth_reserve_is_applied(self):
  h=RendererHistory();h.add(1,np.ones((512,800))*3,np.array([0,0,2]),np.eye(3));v,_=h.volume(1,ErrorBudget(depth_reserve=.15));w,_=h.volume(1,ErrorBudget(depth_reserve=.3));self.assertAlmostEqual(v.cells[0].b[1]-w.cells[0].b[1],.15)
 def test_no_unbounded_pose(self):
  with self.assertRaises(ValueError):RendererHistory().add(1,np.ones((512,800)),np.array([11,0,0]),np.eye(3))
if __name__=='__main__':unittest.main()
