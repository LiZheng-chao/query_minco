import unittest
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *
from query_minco.inference.constrained_prefix import *
class Tests(unittest.TestCase):
 def test_exact_time_scaling(self):
  c=np.zeros((1,3,6));c[0,0]=[0,.8,.1,-.2,.03,.001];p=Program(c,[1.])
  q=slowed(p,.4)
  for t in [0,.1,.5,2.5]:
   expected=p.pva(t*.4);expected[1]*=.4;expected[2]*=.4**2
   np.testing.assert_allclose(q.pva(t),expected,atol=1e-12)
 def test_actual_bundle_limits_brake_and_pva(self):
  with np.load(Path(__file__).resolve().parents[2]/'research_outputs/rsps_forest_shadow_v1/low_speed_shadow001/frame_000080.npz') as z:
   w=Program(world_coefficients(z['C1_coefficients'][0],z['position_W'],z['R_WB0']),z['C1_durations'][0])
   m=Measurement(0,np.stack([z['position_W'],z['R_WB0']@z['state'][:3],z['R_WB0']@z['state'][3:6]]))
   v=FreeVolume((box_cell([-5,-5,0],[40,5,4],20,'external-fixture'),),Scope.EXTERNAL_SUPERVISOR)
   source=m;m=Measurement(.06,source.predict(.06))
   ans=constrained_prefix(w,m,.12,v,.06,source_stamp=0);ans['full'].validate(MotionLimits())
   np.testing.assert_allclose(ans['full'].pva(0),m.predict(.12),atol=1e-10)
   self.assertLess(np.linalg.norm(ans['full'].pva(ans['full'].duration)[1:]),1e-8)
   self.assertGreater(ans['retained_progress_m'],.005)
   with self.assertRaisesRegex(ValueError,'STALE_CURRENT_ODOM'):constrained_prefix(w,source,.12,v,.06,source_stamp=0)
   with self.assertRaises(ValueError):constrained_prefix(w,m,.12,FreeVolume((),Scope.NATIVE_RAYS),.06)
 def test_activation_guard_rejects_recorded_native_lateness(self):
  import json
  p=Path(__file__).resolve().parents[2]/'research_outputs/rsps_f3_low_speed_v1/f3_execution002/NATIVE_MEASURED_RESULTS.json'
  events=json.loads(p.read_text())['activations'];rejected=0
  for e in events:
   if e['lateness_ms']>10:
    with self.assertRaisesRegex(ValueError,'LATE_ACTIVATION'):validate_activation(e['requested'],e['first_published'])
    rejected+=1
   else:validate_activation(e['requested'],e['first_published'])
  self.assertEqual(rejected,3)
 def test_real_head_gate_and_early_activation(self):
  with self.assertRaisesRegex(ValueError,'EARLY'):validate_activation(1,.999)
  self.assertAlmostEqual(validate_activation(1,1.009),.009)
  with self.assertRaises(ValueError):validate_activation(float('nan'),1)
 def test_reference_only_nominal_and_watchdog_plans(self):
  h0=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float);h1=np.array([[0,0,2],[.8,0,0],[0,0,0]],float)
  p=concatenate(quintic(h0,h1,4),terminal(h1,[3,0,2]));p.validate(MotionLimits())
  np.testing.assert_allclose(p.pva(4),h1,atol=1e-12)
  np.testing.assert_allclose(p.pva(p.duration),np.array([[3,0,2],[0,0,0],[0,0,0]]),atol=1e-10)
  q=terminal(np.array([[3,0,2],[0,0,0],[0,0,0]],float),[0,0,2]);b,_=brake(q.pva(1.56));b.validate(MotionLimits())
if __name__=='__main__':unittest.main(verbosity=2)
