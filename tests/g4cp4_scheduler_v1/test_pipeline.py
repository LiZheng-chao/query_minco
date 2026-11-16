import unittest,time
import numpy as np
from query_minco.integration.observed_scheduler_v1.pipeline import Pipeline
class Tests(unittest.TestCase):
 def test_ingress_fences_before_processing(self):
  p=Pipeline(clock=lambda:10.)
  try:
   p.proof=dict(generation=0,plan_id='x',valid_until=11)
   self.assertIsNotNone(p.consume('x'))
   p.ingest(np.full((90,160),5,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),10,10)
   self.assertIsNone(p.consume('x'))
  finally:p.close()
 def test_duplicate_conflict_latches(self):
  p=Pipeline(clock=lambda:10.);pose=dict(p=[0,0,0],q=[0,0,0,1]);d=np.full((90,160),5,np.float32)
  try:
   p.ingest(d,pose,10,10);p.ingest(d*.5,pose,10,10)
   self.assertEqual(p.status()['fault'],'DUPLICATE_CONTENT_CONFLICT');self.assertIsNone(p.consume('x'))
  finally:p.close()
 def test_closed_cannot_regrant(self):
  p=Pipeline();p.close();p.proof=dict(generation=p.generation,plan_id='x',valid_until=time.time()+1)
  self.assertIsNone(p.consume('x'))
 def test_invalid_time_fails(self):
  p=Pipeline(clock=lambda:10)
  try:
   with self.assertRaises(ValueError):p.ingest(np.ones((90,160),np.float32),{},9,9)
   self.assertIsNotNone(p.status()['fault'])
  finally:p.close()

 def test_positive_observed_curve_and_immediate_revocation(self):
  from query_minco.integration.yopo_execution_v4.trajectory import build
  p=Pipeline(clock=lambda:10.)
  try:
   plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,0]),10)
   p.request(plan.record)
   p.ingest(np.full((90,160),5,np.float32),dict(p=[-1.2,0,0],q=[0,0,0,1]),10,10)
   until=time.monotonic()+2
   while p.status()['processed']==0 and time.monotonic()<until:time.sleep(.001)
   self.assertIsNotNone(p.consume(plan.id),str(p.status()))
   p.fail('NEW_OBSTACLE');self.assertIsNone(p.consume(plan.id))
  finally:p.close()
