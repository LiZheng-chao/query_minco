import unittest,json
from pathlib import Path
from query_minco.integration.forest_fix2_v1.pairing import Pairing
class Tests(unittest.TestCase):
 def test_reproduce_old_drop_then_fix(self):
  old_poses={};self.assertNotIn(10.,old_poses)
  q=Pairing();q.depth(10.,'DEPTH');self.assertEqual(q.take(10.),[])
  q.pose(10.,'POSE');self.assertEqual(q.take(10.01),[(10.,'DEPTH','POSE')])
 def test_other_arrival_order(self):
  q=Pairing();q.pose(10,'P');q.depth(10,'D');self.assertEqual(q.take(10),[(10,'D','P')])
 def test_no_nearest_or_expired_permission(self):
  q=Pairing();q.pose(10.01,'P');q.depth(10,'D');self.assertEqual(q.take(10.02),[])
  self.assertEqual(q.take(10.21),[]);self.assertEqual(q.drops[-1][1],'SOURCE_EXPIRED')
 def test_queue_bound(self):
  q=Pairing()
  for i in range(6):q.depth(10+i*.01,'D')
  self.assertEqual(len(q.depths),4);self.assertEqual(len(q.drops),2)
 def test_no_rollback_acceptance(self):
  q=Pairing();q.pose(10.1,'P');q.depth(10.1,'D');q.take(10.1)
  q.pose(10.05,'P');q.depth(10.05,'D');self.assertEqual(q.take(10.1),[])
 def test_birth_protocol_and_registration(self):
  import xml.etree.ElementTree as ET
  o=Path('research_outputs/g4cp7b_fix2_20261112_v1');birth=json.loads((o/'BIRTH_SELECTION.json').read_text())
  self.assertGreaterEqual(birth['selected']['clearance_lower_bound_m'],.60)
  self.assertLess(birth['candidates_evaluated'][0]['clearance_lower_bound_m'],.60)
  scene=json.loads((o/'SCENES.json').read_text())['open']
  self.assertEqual(scene['spawn'],birth['selected']['position'])
  n=ET.parse('ros/g4cp7b_fix2_v1/controller.launch').getroot().find('node')
  params={p.attrib['name']:p.attrib['value'] for p in n.findall('param')}
  self.assertEqual([float(params['so3_control/init_state_'+a]) for a in 'xyz'],scene['spawn'])
 def test_e1b_has_no_geometry_oracle(self):
  s=Path('query_minco/integration/forest_fix2_v1/depth_planner.py').read_text()
  self.assertNotIn('native_grid',s);self.assertNotIn('BIRTH_SELECTION',s);self.assertIn('snap.certify',s)

 def test_relay_cleanup_and_tracing(self):
  s=Path('ros/g4cp7b_fix2_v1/relay.py').read_text()
  self.assertIn('except rospy.ROSInterruptException:pass',s)
  self.assertIn('receipt_monotonic',s);self.assertIn('published=not filtered',s)
 def test_e1b_unknown_rejection(self):
  import numpy as np
  from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
  p=DepthPlanner(clock=lambda:10.)
  state=dict(p=[0,0,0],q=[0,0,0,1],v=[0,0,0],a=[0,0,0])
  p.observe(np.full((90,160),20,np.float32),state,10.)
  with self.assertRaises(ValueError):p.propose(state,[.5,0,0],10.)
