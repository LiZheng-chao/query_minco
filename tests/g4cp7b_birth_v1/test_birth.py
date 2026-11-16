import unittest,tempfile,json,time
from pathlib import Path
import numpy as np
from query_minco.integration.birth_execution_v1.contract import certify
from query_minco.integration.yopo_execution_v4.trajectory import build,Admission
class Tests(unittest.TestCase):
 def fixture(self,path):
  (path/'truth_data').mkdir()
  (path/'BIRTH_CERTIFICATE.json').write_text(json.dumps(dict(scope='ENGINEERING_BIRTH_BALL_ONLY',radius_m=.6,center_world=[0.,0.,2.],issued=10,expires=40,map_id='TEST_ONLY')))
  (path/'truth_data/HEARTBEAT.json').write_text(json.dumps(dict(status='PASS',monotonic=time.monotonic(),map_id='TEST_ONLY')))
 def test_plan_switch_within_ball(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r)
   p=build(np.array([0,0,2]),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.28,0,2]),10)
   self.assertTrue(certify(r,plan=p,now=10))
   state=p.sample(1.8);q=build(state[0],state[1],state[2],[0,0,0,1],np.array([.28,0,2]),11.8)
   self.assertTrue(certify(r,plan=q,now=11.8))
   np.testing.assert_allclose(q.sample(0),p.sample(1.8),atol=1e-5)
 def test_boundary_no_corridor(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r)
   with self.assertRaises(ValueError):certify(r,pva=np.array([[.4,0,2],[0,0,0],[0,0,0]]),now=10)
   with self.assertRaises(ValueError):certify(r,pva=np.array([[0,0,2],[0,0,0],[0,0,0]]),now=41)
 def test_map_change_denies(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.fixture(r)
   t=r/'truth_data/HEARTBEAT.json';v=json.loads(t.read_text());v['map_id']='NEW';t.write_text(json.dumps(v))
   with self.assertRaises(ValueError):certify(r,pva=np.zeros((3,3)),now=10)
 def test_static_owner_prearm_guard(self):
  s=Path('experiments/g4cp7b_birth_v1/manual.py').read_text();compile(s,'owner','exec')
  self.assertLess(s.index("BIRTH_SPHERE_NOT_CONFIRMED"),s.index("launch('arm_permit'"))
  self.assertIn("ref['replans']>=2",s)

 def test_truth_failure_specific(self):
  from query_minco.integration.birth_execution_v1.evidence import CASES,expected_fault
  self.assertEqual(len(CASES),7)
  self.assertTrue(expected_fault('truth_failure','TRUTH_COLLISION_UNKNOWN_OR_STALE'))
  self.assertFalse(expected_fault('truth_failure','REQUEST some_error'))
