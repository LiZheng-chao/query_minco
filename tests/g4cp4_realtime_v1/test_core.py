import unittest,tempfile,json,time
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v2.core import sample_batch,Prepared,ObservedMap,AsyncCertificates
from query_minco.integration.observed_space_v2.bootstrap import validate
from query_minco.integration.yopo_execution_v4.trajectory import build
class Tests(unittest.TestCase):
 def plan(self):return build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,np.sin(.3),np.cos(.3)],np.array([.4,0,0]),10)
 def test_batch_equivalence(self):
  p=self.plan();t=np.unique(np.r_[np.linspace(0,2,129),np.cumsum(p.d)])
  np.testing.assert_allclose(sample_batch(p,t),np.stack([p.sample(x) for x in t]),rtol=1e-11,atol=1e-11)
 def test_time_contract(self):
  for t in [[-1],[float('nan')],[3]]:
   with self.assertRaises(ValueError):sample_batch(self.plan(),t)
 def test_repeated_geometry_reuse_and_epoch(self):
  m=ObservedMap();d=np.full((90,160),5,np.float32);pose=dict(p=[0,0,0],q=[0,0,0,1])
  m.add(d,pose,10,10,10);s=m.snapshot(10);cache=m.frames[-1].cache
  m.add(d,pose,10.1,10.1,10.1)
  self.assertIs(cache,m.frames[-1].cache)
  with self.assertRaises(ValueError):s.current(10.1)
  self.assertFalse(m.snapshot(10.1).ball([0,0,0],.3,10.1)['passed'])
 def test_new_obstacle_not_reused(self):
  m=ObservedMap();pose=dict(p=[0,0,0],q=[0,0,0,1]);d=np.full((90,160),5,np.float32)
  m.add(d,pose,10,10,10);m.snapshot(10)
  m.add(d*.2,pose,10.1,10.1,10.1);self.assertFalse(m.frames[-1].cache)
 def test_no_pending_or_failed_permission(self):
  w=AsyncCertificates(ObservedMap(),clock=lambda:10)
  self.assertIsNone(w.consume('anything'));w.submit(self.plan())
  w.future.result if False else None
  w.close();self.assertIsNone(w.consume('anything'))
 def test_measurement_not_implicit_free(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.json';p.write_text('{}')
   with self.assertRaises(ValueError):validate(p,10,'scene',[0,0,0])
 def test_prepared_same_values(self):
  p=self.plan();prepared=Prepared(p)
  np.testing.assert_allclose(prepared.values,np.stack([p.sample(t) for t in prepared.times]),atol=1e-11)
