"""Portable synthetic tests; no historical data or checkpoints."""
import hashlib,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from query_minco.registry import describe
from query_minco.trajectory.parameter_decoder import PhysicalParameterDecoder
from query_minco.trajectory.cumulative_time import cumulative_three
from query_minco.training.matching.hungarian import match
from query_minco.training.datasets.cache import load_samples
from query_minco.evaluation.goal import task_metrics
class ReferenceTests(unittest.TestCase):
 def test_unknown_method(self):
  with self.assertRaises(ValueError):describe('latest')
 def test_schema(self):
  raw=torch.zeros(1,6,21);state=torch.zeros(1,12);state[0,0]=.8;state[0,11]=-1
  p=PhysicalParameterDecoder()(raw,state)
  self.assertEqual(tuple(p.inner_pts.shape),(1,6,2,3))
  self.assertTrue(torch.allclose(p.durations.sum(-1),p.total_time))
  self.assertTrue(bool((p.durations>=.12).all()))
 def test_e1_gradient(self):
  d=torch.tensor([.2,.3,.4],requires_grad=True);c=cumulative_three(d);c.sum().backward()
  self.assertTrue(torch.equal(d.grad,torch.tensor([3.,2.,1.])))
 def test_near_boundary(self):
  self.assertEqual(task_metrics([3.4,0,0],[0,0,0],[3.4,0,0])['mode'],'near_arrival')
  self.assertEqual(task_metrics([1.5,0,0],[1,0,0],[3.400001,0,0])['mode'],'far_progress')
 def test_far_does_not_require_arrival(self):
  self.assertTrue(task_metrics([2.,0,0],[1,0,0],[5.8,0,0])['passed'])
 def test_hungarian_M1_M4(self):
  for M in [1,4]:
   c=(torch.arange(6)[:,None]-torch.arange(M)[None,:]).float().square().requires_grad_()
   loss,q,t=match(c,'B_set');loss.backward()
   self.assertEqual(len(q.unique()),M);self.assertEqual(float(loss),0.)
   self.assertEqual(int(torch.count_nonzero(c.grad)),M)
 def test_illegal_teacher_count(self):
  for M in [0,7]:
   with self.assertRaises(ValueError):match(torch.zeros(6,M),'B_set')
 def test_cache_hash_and_order(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'sample.npz'
   with p.open('xb') as f:np.savez(f,depth_input=np.zeros((2,128,160),np.float32),state=np.zeros(12,np.float32),
    teacher_coeffs=np.zeros((1,3,3,6),np.float32),teacher_durations=np.ones((1,3),np.float32))
   h=hashlib.sha256(p.read_bytes()).hexdigest();rows={k:dict(path=p.name,sha256=h) for k in ['a','b']}
   self.assertEqual([s.sample_id for s in load_samples(['b','a','b'],rows,tmp)],['b','a','b'])
   rows['a']['sha256']='0'*64
   with self.assertRaises(ValueError):load_samples(['a'],rows,tmp)
if __name__=='__main__':unittest.main()
