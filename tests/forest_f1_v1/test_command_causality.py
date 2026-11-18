import unittest
import numpy as np
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
class Causality(unittest.TestCase):
 def setup_engine(self):
  self.now=100.;e=Engine(lambda n:BirthGrant((0,0,0),.6,n+.2,'SYNTHETIC_NOT_FLIGHT'),lambda:self.now);self.pose=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);self.depth=np.full((90,160),5.,np.float32)
  e.observe(self.depth,self.pose,self.now,self.now);e.set_state(dict(self.pose,a=[0,0,0]),self.now);e.set_goal([.28,0,0],self.now);return e
 def test_commit_without_execution_state_can_supply_command(self):
  e=self.setup_engine()
  try:self.assertTrue(e.run_once());self.assertIsNone(e.last_executed_plan);self.assertIsNotNone(e.command(self.now))
  finally:e.close()
 def test_new_geometry_revokes_immediately(self):
  e=self.setup_engine()
  try:
   self.assertTrue(e.run_once());self.assertIsNotNone(e.command(self.now));self.now+=.01;self.pose['p']=[.001,0,0];e.observe(self.depth,self.pose,self.now,self.now);self.assertIsNone(e.command(self.now))
  finally:e.close()
 def test_four_commits_can_be_one_plan(self):
  e=self.setup_engine()
  try:
   self.assertTrue(e.run_once())
   for _ in range(3):
    self.now+=.01;e.observe(self.depth,self.pose,self.now,self.now);e.set_state(dict(self.pose,a=[0,0,0]),self.now);e.set_goal([.28,0,0],self.now);self.assertTrue(e.run_once());self.assertIsNotNone(e.command(self.now))
   rows=[x for x in e.take_events() if x['event']=='CERTIFIED_COMMITTED'];self.assertEqual(len(rows),4);self.assertEqual(len(set(x['plan_id'] for x in rows)),1)
   self.now+=.201;self.assertIsNone(e.command(self.now))
  finally:e.close()
