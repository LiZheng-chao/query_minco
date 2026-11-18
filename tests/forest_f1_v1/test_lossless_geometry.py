import unittest,time,math
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from query_minco.integration.observed_space_v1.map import Frame,Config,ObservedMap,BudgetExceeded,KNOWN_FREE,OCCUPIED,UNKNOWN
from query_minco.integration.depth_minco_t2_v1.engine import Engine
class Geometry(unittest.TestCase):
 def frame(self,depth=None):return Frame(np.full((90,160),5.,np.float32) if depth is None else depth,dict(p=[0,0,0],q=[0,0,0,1]),100.,100.,100.,'scene','camera')
 def test_error_bound_off_axis_extremes(self):
  cfg=Config();pix=cfg.pixel_guard+.5;rng=np.random.default_rng(83)
  for u,v,z in [(0,0,.5),(159,89,19.9),(80,45,1.)]:
   ray=np.array([(u-80)/80,(v-45)/80,1.]);U=cfg.position_error+cfg.depth_error*np.linalg.norm(ray)+(z+cfg.depth_error)*math.sqrt(2)*pix/80+2*math.sin(cfg.angle_error/2)*z*np.linalg.norm(ray)
   for _ in range(200):
    axis=rng.normal(size=3);axis/=np.linalg.norm(axis);R=Rotation.from_rotvec(axis*cfg.angle_error).as_matrix();dp=np.array([pix/80,pix/80,0])*rng.choice([-1,1],3);point=R@((z+cfg.depth_error)*(ray+dp));error=np.linalg.norm(point-z*ray)
    self.assertLessEqual(error,U+1e-12)
 def test_near_obstacle_survives_far_hit(self):
  d=np.full((90,160),np.nan,np.float32);d[45,80]=.5;d[0,0]=19.9;f=self.frame(d);cfg=Config(shape=(2,2,2));origin=np.array([.4,-.1,-.1]);centers=origin+(np.indices(cfg.shape).transpose(1,2,3,0).reshape(-1,3)+.5)*.1
  free,occ,vis=f.carve(centers,origin,cfg,time.perf_counter()+.15);self.assertTrue(occ.any());self.assertFalse(free.any())
 def test_farther_large_radius_hit_is_not_missed(self):
  # Artificial hit cloud isolates variable-radius union query, not a sensor fixture.
  f=self.frame();f.hit_points=np.array([[.5,0,0],[.7,0,0]]);f.hit_depths=np.array([.1,19.]);f.hit_ray_norm=np.ones(2);f.occupied_groups={};cfg=Config(shape=(2,2,2));origin=np.array([-.1,-.1,-.1]);centers=origin+(np.indices(cfg.shape).transpose(1,2,3,0).reshape(-1,3)+.5)*.1
  _,occ,_=f.carve(centers,origin,cfg,time.perf_counter()+.15);self.assertTrue(occ.all())
 def test_missing_depth_does_not_create_free(self):
  f=self.frame(np.full((90,160),np.nan,np.float32));cfg=Config(shape=(2,2,2));origin=np.zeros(3);centers=(np.indices(cfg.shape).transpose(1,2,3,0).reshape(-1,3)+.5)*.1
  free,occ,vis=f.carve(centers,origin,cfg,time.perf_counter()+.15);self.assertFalse(free.any());self.assertFalse(occ.any())
 def test_failed_snapshot_retains_observations_and_revokes(self):
  m=ObservedMap(Config(shape=(4,4,4)));p=dict(p=[0,0,0],q=[0,0,0,1]);m.add(np.full((90,160),5.,np.float32),p,100.,100.,100.);epoch=m.epoch
  with patch.object(Frame,'carve',side_effect=BudgetExceeded('TEST')):
   with self.assertRaises(BudgetExceeded):m.snapshot(100.)
  self.assertEqual(len(m.frames),1);self.assertGreater(m.epoch,epoch);self.assertIsNone(m.latest_snapshot)
  snap=m.snapshot(100.);self.assertTrue(snap.frames)
 def test_stale_snapshot_fails(self):
  m=ObservedMap(Config(shape=(4,4,4)));p=dict(p=[0,0,0],q=[0,0,0,1]);m.add(np.full((90,160),5.,np.float32),p,100.,100.,100.);snap=m.snapshot(100.)
  with self.assertRaises(ValueError):snap.current(100.201)
 def test_fusion_not_throttled_with_pending(self):
  from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
  now=[100.];e=Engine(lambda n:BirthGrant((0,0,0),.6,n+.2,'SYNTHETIC'),lambda:now[0]);p=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);e.next_plan_time=101.
  try:
   for i in range(10):
    now[0]=100.+i*.01;e.observe(np.full((90,160),5.,np.float32),p,now[0],now[0]);e.set_state(dict(p,a=[0,0,0]),now[0]);e.set_goal([.28,0,0],now[0]);e.run_once();self.assertEqual(len(e.pending),0)
   self.assertFalse(e.evidence_overflow)
  finally:e.close()

 def test_transient_fusion_failure_does_not_accumulate_to_eight(self):
  from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
  now=[100.];e=Engine(lambda n:BirthGrant((0,0,0),.6,n+.2,'SYNTHETIC'),lambda:now[0]);pose=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);depth=np.full((90,160),5.,np.float32)
  try:
   original=e.ledger.snapshot;count=[0]
   def fail_once(n):
    count[0]+=1
    if count[0]==1:
     with patch.object(Frame,'carve',side_effect=BudgetExceeded('TEST')):return original(n)
    return original(n)
   e.ledger.snapshot=fail_once
   for i in range(10):
    now[0]=100.+i*.01;e.observe(depth,pose,now[0],now[0]);e.set_state(dict(pose,a=[0,0,0]),now[0]);e.set_goal([.28,0,0],now[0]);e.run_once()
   self.assertFalse(e.evidence_overflow);self.assertEqual(len(e.pending),0)
  finally:e.close()
 def test_new_obstacle_beats_birth(self):
  m=ObservedMap(Config(shape=(8,8,8)));pose=dict(p=[0,0,0],q=[0,0,0,1]);m.add(np.full((90,160),.15,np.float32),pose,100.,100.,100.);snap=m.snapshot(100.)
  from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot
  snap=authorized_snapshot(snap,BirthGrant((0,0,0),.6,100.2,'SYNTHETIC'),100.,snap.epoch,lambda:snap.epoch)
  self.assertFalse(snap.ball([0,0,0],.3,100.)['passed'])
 def test_rolling_and_history_does_not_reuse_stale_permission(self):
  m=ObservedMap(Config(shape=(6,6,6)));pose=dict(p=[0,0,0],q=[0,0,0,1]);m.add(np.full((90,160),5.,np.float32),pose,100.,100.,100.);old=m.snapshot(100.)
  moved=dict(p=[.2,0,0],q=[0,0,0,1]);m.add(np.full((90,160),np.nan,np.float32),moved,100.1,100.1,100.1);new=m.snapshot(100.1)
  with self.assertRaises(ValueError):old.current(100.1)
  with self.assertRaises(ValueError):new.current(102.2)

 def test_variable_radius_query_matches_exhaustive_union(self):
  rng=np.random.default_rng(91);f=self.frame();f.hit_points=rng.uniform(-1,1,(30,3));f.hit_depths=rng.uniform(.1,19.9,30);f.hit_ray_norm=rng.uniform(1,1.5,30);f.occupied_groups={}
  cfg=Config(shape=(6,6,6));origin=np.array([-.3,-.3,-.3]);centers=origin+(np.indices(cfg.shape).transpose(1,2,3,0).reshape(-1,3)+.5)*.1
  _,actual,_=f.carve(centers,origin,cfg,time.perf_counter()+.15)
  U=cfg.position_error+cfg.depth_error*f.hit_ray_norm+(f.hit_depths+cfg.depth_error)*np.sqrt(2)*(cfg.pixel_guard+.5)/80+2*np.sin(cfg.angle_error/2)*f.hit_depths*f.hit_ray_norm
  radii=(np.ceil(U/.1)+1.5)*.1
  expected=(np.max(np.abs(centers[:,None,:]-f.hit_points[None,:,:]),axis=2)<=radii[None,:]).any(1)
  np.testing.assert_array_equal(actual.ravel(),expected)
 def test_failed_cached_roi_is_retired_without_discarding_evidence(self):
  from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
  now=[100.];e=Engine(lambda n:BirthGrant((0,0,0),.6,n+.2,'SYNTHETIC'),lambda:now[0]);pose=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);d=np.full((90,160),5.,np.float32)
  try:
   e.observe(d,pose,100.,100.);e.set_state(dict(pose,a=[0,0,0]),100.);e.set_goal([.28,0,0],100.);self.assertTrue(e.run_once())
   now[0]=100.01;pose['p']=[.001,0,0];e.observe(d,pose,now[0],now[0]);e.set_state(dict(pose,a=[0,0,0]),now[0]);e.set_goal([.28,0,0],now[0])
   with patch('query_minco.integration.depth_minco_t2_v1.engine.latest_proof',side_effect=ValueError('OCCUPIED_ENVELOPE')):self.assertFalse(e.run_once())
   self.assertIsNone(e.last_computed_plan);self.assertTrue(e.pending);self.assertIsNone(e.command(now[0]))
  finally:e.close()
