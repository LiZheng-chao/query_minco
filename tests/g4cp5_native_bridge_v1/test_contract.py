import unittest,ast
from pathlib import Path
import numpy as np
from query_minco.integration.yopo_native_bridge_v1.contract import inputs,reference,acceleration,imu_fresh
class Tests(unittest.TestCase):
 def state(self):return dict(stamp=10.,p=[0,0,2],v=[0,0,0],q=[0,0,0,1])
 def test_source_time(self):
  with self.assertRaises(ValueError):inputs(np.ones((90,160),np.float32),self.state(),np.zeros(3),[1,0,2],[80.,0.,80.,0.,80.,45.,0.,0.,1.],9.9,10)
 def test_native_zero_imu_is_rejected(self):
  with self.assertRaises(ValueError):imu_fresh(0,10)
 def test_real_minco_limits(self):
  p=reference(self.state(),np.zeros(3),[.4,0,2]);self.assertLess(p.maxima[0],1)
  np.testing.assert_allclose(p.sample(0)[0],[0,0,2],atol=1e-5);np.testing.assert_allclose(p.sample(2)[0],[.4,0,2],atol=1e-5)
 def test_odom_derivative(self):
  before=self.state();before['stamp']=9.99
  np.testing.assert_equal(acceleration(before,self.state()),np.zeros(3))
 def test_no_actuation_topic(self):
  text=Path('ros/g4cp5_native_bridge_v1/bridge.py').read_text();compile(text,'bridge','exec')
  self.assertNotIn("Publisher('/so3_control/pos_cmd'",text);self.assertIn("Publisher('/query_minco/p5/position_candidate'",text)
