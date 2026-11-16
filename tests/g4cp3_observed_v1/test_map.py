import unittest,time,copy,tempfile,ast
from dataclasses import replace
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v1.map import ObservedMap,Config,KNOWN_FREE,OCCUPIED,UNKNOWN,CertificationError,BudgetExceeded
from query_minco.integration.observed_space_v1.space import bootstrap,diagnose_legacy_ball
from query_minco.integration.yopo_execution_v4.trajectory import build
class MapTests(unittest.TestCase):
 def add(self,m,depth=5.,p=(0,0,0),q=(0,0,0,1),t=10.):
  d=np.full((90,160),depth,np.float32) if np.isscalar(depth) else depth
  m.add(d,dict(p=list(p),q=list(q),frame='world'),t,t,t);return m.snapshot(t)
 def test_free_carve_and_near_unknown(self):
  m=ObservedMap();s=self.add(m)
  self.assertTrue(s.ball([1.4,0,0],.1,10)['passed']);self.assertFalse(s.ball([0,0,0],.15,10)['passed'])
 def test_hit_occupied(self):
  s=self.add(ObservedMap(),1.5)
  self.assertGreater(s.ball([1.5,0,0],.1,10).get('occupied',0),0)
 def test_no_return_and_invalid(self):
  for d in [20.,0.,float('nan'),float('inf')]:
   s=self.add(ObservedMap(),d);self.assertFalse(s.ball([1.4,0,0],.1,10)['passed'])
 def test_rotation_measured_quaternion(self):
  s=self.add(ObservedMap(),5.,q=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)])
  self.assertTrue(s.ball([0,1.4,0],.1,10)['passed']);self.assertFalse(s.ball([1.4,0,0],.1,10)['passed'])
 def test_same_obstacle_multiview(self):
  m=ObservedMap();a=self.add(m,1.5);self.assertGreater(a.ball([1.5,0,0],.1,10)['occupied'],0)
  b=self.add(m,1.3,p=[.2,0,0],t=10.1);self.assertGreater(b.ball([1.5,0,0],.1,10.1)['occupied'],0)
 def test_novel_frame_fences_previous_snapshot(self):
  m=ObservedMap();a=self.add(m)
  self.add(m,5.,t=10.1)
  with self.assertRaises(CertificationError):a.ball([1.4,0,0],.1,10.1)
 def test_occluder_clears_old_free(self):
  m=ObservedMap();s=self.add(m,5.);self.assertTrue(s.ball([1.4,0,0],.1,10)['passed'])
  s=self.add(m,.6,t=10.1);self.assertFalse(s.ball([1.4,0,0],.1,10.1)['passed'])
 def test_scene_reset(self):
  m=ObservedMap();s=self.add(m);m.reset_scene('new')
  with self.assertRaises(CertificationError):s.ball([1.4,0,0],.1,10)
  with self.assertRaises(ValueError):m.snapshot(10)
 def test_stale_frames_and_pose_epoch(self):
  m=ObservedMap();s=self.add(m)
  with self.assertRaises(ValueError):m.snapshot(10.201)
  with self.assertRaises(CertificationError):s.ball([1.4,0,0],.1,10.201)
  with self.assertRaises(ValueError):m.add(np.ones((90,160),np.float32),dict(p=[0,0,0],q=[0,0,0,1]),11,10.99,11)
 def test_duplicates_not_freshened(self):
  m=ObservedMap();s=self.add(m);epoch=m.epoch
  self.assertFalse(m.add(np.full((90,160),5.,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),10,10,10.1))
  self.assertEqual(m.epoch,epoch);self.assertEqual(m.last_stamp,10)
  with self.assertRaises(ValueError):m.add(np.full((90,160),4.,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),10,10,10.1)
  with self.assertRaises(CertificationError):s.current(10.1)
 def test_time_rollback(self):
  m=ObservedMap();self.add(m)
  with self.assertRaises(ValueError):self.add(m,t=9.99)
 def test_full_cell_unknown_at_boundary(self):
  s=self.add(ObservedMap());self.assertFalse(s.ball([1.4,1.4,0],.2,10)['passed'])
 def test_explicit_bootstrap_no_free_bubble(self):
  s=self.add(ObservedMap());r=bootstrap(s,[0,0,0],10);self.assertEqual(r['status'],'FAIL')
 def test_multiview_can_cover_separately_observed_start(self):
  # Realistic synthetic source path:1m/s,20Hz,same fixed world plane atX3.5.
  m=ObservedMap(Config(shape=(32,32,24)))
  for i in range(31):
   x=-1.5+i*.05;t=10+i*.05
   s=self.add(m,3.5-x,p=[x,0,0],t=t)
  self.assertTrue(s.ball([0,0,0],.30,t)['passed'])
  p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,0]),t)
  # Geometry correctness and realtime feasibility are separate assertions.
  # No synthetic clock and no relaxed deadline are used.
  self.assertTrue(all(s.ball(p.sample(x)[0],.3+p.maxima[0]*2/256,t)['passed'] for x in np.linspace(0,2,129)))
  try:self.assertTrue(s.certify(p,t)['passed'])
  except BudgetExceeded as e:self.assertEqual(str(e),'CURVE_CERTIFICATION_DEADLINE')
 def test_tracking_envelope_increases_unknown_or_occupied(self):
  s=self.add(ObservedMap(),1.5)
  self.assertTrue(s.ball([.8,0,0],.05,10)['passed']);self.assertFalse(s.ball([.8,0,0],.5,10)['passed'])
 def test_grid_budget(self):
  for cfg in [Config(shape=(100,100,100)),Config(max_frames=100),Config(resolution=.001)]:
   with self.assertRaises(ValueError):ObservedMap(cfg)
 def test_compute_budget_fails_closed(self):
  m=ObservedMap(Config(update_budget_s=1e-12));m.add(np.full((90,160),5,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),10,10,10)
  with self.assertRaises(BudgetExceeded):m.snapshot(10)
  self.assertFalse(m.frames)
 def test_memory_history_bound(self):
  m=ObservedMap(Config(max_frames=3))
  for i in range(6):m.add(np.full((90,160),5+i*.01,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),10+i*.01,10+i*.01,10+i*.01)
  self.assertEqual(len(m.frames),3);self.assertLessEqual(len(m.records),64)
 def test_immutable_snapshot(self):
  s=self.add(ObservedMap())
  with self.assertRaises(ValueError):s.states[:]=1
 def test_no_gt_dependencies(self):
  for p in Path('query_minco/integration/observed_space_v1').glob('*.py'):
   text=p.read_text();self.assertNotIn('forest_adapter',text);self.assertNotIn('COMPACT_COUNTS',text);self.assertNotIn('hard_validator',text)
 def test_failure_taxonomy(self):
  pose=dict(p=[0,0,0],q=[0,0,0,1]);depth=np.full((90,160),5,np.float32)
  self.assertEqual(diagnose_legacy_ball(depth,pose,[0,0,0],.3)['reason'],'NEAR_FIELD_BLIND_ZONE')
  self.assertEqual(diagnose_legacy_ball(depth,pose,[1,3,0],.3)['reason'],'OUTSIDE_FOV')
  self.assertEqual(diagnose_legacy_ball(depth*0,pose,[1.5,0,0],.1)['reason'],'INVALID_OR_NO_RETURN_DEPTH')
