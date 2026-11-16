import unittest
from query_minco.integration.yopo_forest_bridge_v1.pairing import Pairing
class Tests(unittest.TestCase):
 def test_depth_before_pose(self):
  p=Pairing();p.add('depth',10,'d',10);self.assertIsNone(p.latest(10));p.add('pose',10,'p',10.01);self.assertEqual(p.latest(10.01),(10,'d','p'))
 def test_pose_before_depth(self):
  p=Pairing();p.add('pose',10,'p',10);p.add('depth',10,'d',10.01);self.assertEqual(p.latest(10.01),(10,'d','p'))
 def test_no_nearest_pose_substitution(self):
  p=Pairing();p.add('pose',10.01,'p',10.01);p.add('depth',10,'d',10.01);self.assertIsNone(p.latest(10.02))
 def test_expired_not_restamped(self):
  p=Pairing();p.add('pose',10,'p',10);p.add('depth',10,'d',10);self.assertIsNone(p.latest(10.21))
 def test_bounded(self):
  p=Pairing(max_depth=2)
  for i in range(10):p.add('depth',10+i*.01,'d',10+i*.01)
  self.assertEqual(len(p.depth),2)
