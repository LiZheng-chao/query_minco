import unittest,time
import numpy as np
from query_minco.integration.f0_continuity_v1.authority import BirthAuthority
from query_minco.integration.f0_continuity_v1.admission import ContinuousAdmission
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
from query_minco.integration.yopo_execution_v4.trajectory import build,Admission
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.28,0,0],100.)
 def authority(self):
  self.now=100.
  a=BirthAuthority(lambda t:BirthGrant((0,0,0),.6,100.2,'SYNTHETIC_ONLY'),lambda:self.now)
  a.observe(np.full((90,160),5.,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),100.)
  return a
 def test_valid_full_tube_and_tracking(self):
  a=self.authority();p=a.check(plan=self.plan)
  self.assertEqual(p['plan_id'],self.plan.id);self.assertLessEqual(p['certificate_expiry'],100.2)
 def test_new_occupied_veto_after_cache(self):
  a=self.authority();a.check(plan=self.plan);self.now=100.03
  a.observe(np.full((90,160),.15,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),self.now)
  with self.assertRaisesRegex(ValueError,'NEW_OCCUPIED'):a.check(plan=self.plan)
 def test_stale_sensor_not_refreshed_by_grant(self):
  a=self.authority();self.now=100.201
  with self.assertRaises(ValueError):a.check(plan=self.plan)
 def test_source_rollback(self):
  a=self.authority()
  with self.assertRaises(ValueError):a.observe(np.full((90,160),5.,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),99.99)
 def test_finite_grant_no_expansion(self):
  a=self.authority();p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.5,0,0],100.)
  with self.assertRaisesRegex(ValueError,'OUTSIDE_BIRTH'):a.check(plan=p)
 def test_prepared_identity_mismatch(self):
  other=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.1,0,0],100.)
  p=self.plan
  with self.assertRaisesRegex(ValueError,'PREPARED_PLAN_IDENTITY'):
   Admission().check(dict(kind='trajectory',plan=p.record,pva=p.at(100.).tolist(),stamp=100.),dict(p=[0,0,0],v=[0,0,0],a=[0,0,0]),100.,prepared=other)
 def test_pending_reference_is_certified_hold_and_bounded(self):
  gate=ContinuousAdmission();gate.preparation.builder=lambda rec:(time.sleep(.04),self.plan)[1]
  env=dict(kind='trajectory',plan=self.plan.record,pva=self.plan.at(100.).tolist(),stamp=100.)
  try:
   t=time.perf_counter();v=gate.check(env,dict(p=[0,0,0],v=[0,0,0],a=[0,0,0]),100.)
   self.assertLess(time.perf_counter()-t,.02);self.assertTrue(np.all(v==0))
   self.authority().check(pva=v)
   gate.preparation.began-=1
   with self.assertRaisesRegex(ValueError,'PLANNING_DEADLINE'):gate.check(env,dict(p=[0,0,0],v=[0,0,0],a=[0,0,0]),100.)
  finally:gate.close()
 def test_hold_not_unbounded_brake(self):
  gate=ContinuousAdmission()
  try:
   with self.assertRaisesRegex(ValueError,'INITIAL_PVA_HOLD'):gate.check(dict(kind='hold',pva=np.zeros((3,3)).tolist(),stamp=100.),dict(p=[0,0,0],v=[.5,0,0],a=[0,0,0]),100.)
  finally:gate.close()
