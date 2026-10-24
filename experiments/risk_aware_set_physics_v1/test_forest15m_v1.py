"""Finite CPU checks only; no capture, native executable, bulk evaluation, optimizer or ROS."""
import sys,json,unittest,contextlib,io
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import forest15m_v1 as f

class ForestTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.lib=f.legacy()
 def test_goal_contract_training_support_and_rotation(self):
  for yaw in [0.,.5,-.5]:
   R=f.rotation([0,0,np.sin(yaw/2),np.cos(yaw/2)])
   p=np.array([2.,-3.,2.]);g=p+R@np.array([15.,0,0])
   state,meta=f.local_state(g,p,R)
   support=f.read(f.PREP/'audit/TRAIN_STATE_SUPPORT.json')['unique_states']
   self.assertTrue(any(np.array_equal(state,np.asarray(s,np.float32)) for s in support))
   self.assertAlmostEqual(np.linalg.norm(state[6:9]),2.4,places=5)
   self.assertTrue(meta['clipped']);self.assertFalse(np.allclose(state[6:9],[15,0,0]))
   np.testing.assert_allclose(R.T@((R@np.array([1.,-.2,.1]))),[1,-.2,.1],atol=1e-12)
  with self.assertRaises(ValueError):f.local_state([1,0,0],np.zeros(3),np.eye(3))
 def test_encoding_invalid_nohit_unknown(self):
  raw=np.full((128,160),2.,np.float32);raw[0,:5]=[0,np.nan,np.inf,20,-1]
  enc=self.lib.encode(raw);self.assertTrue(np.array_equal(enc[:,0,:5],np.zeros((2,5))))
  self.assertEqual(enc[0,1,1],np.float32(.1));self.assertEqual(enc[1,1,1],1)
 def test_existing_geometry_positive_negative_unknown(self):
  with contextlib.redirect_stdout(io.StringIO()):self.lib.voxel.synthetic_tests()
 def test_thin_discrete_miss_not_safety(self):
  grid=np.zeros((1200,20,20),np.uint8);grid[500,10,10]=1
  world=self.lib.voxel.VoxelWorld(grid,[0,-.01,-.01],.001)
  points=np.stack([np.linspace(.02,.98,64),np.zeros(64),np.zeros(64)],1)
  # Select a narrow cell between two samples and use a tiny FIXTURE radius only.
  grid[:]=0;grid[500,10,10]=1
  sample_hit=any(world.point_witness(p,.001) is not None for p in points)
  cp=[np.array([[.02,0,0],[.98,0,0]])]
  result=self.lib.voxel.classify(world,cp,.001,0,16,50000)
  self.assertFalse(sample_hit);self.assertEqual(result['status'],'BODY_COLLISION')
 def test_geometry_connections_no_homotopy_claim(self):
  world=self.lib.voxel.VoxelWorld(np.zeros((80,80,80),np.uint8),[-2,-2,-2],.05)
  a=np.array([[[0,0,0],[.5,0,0]],[[.5,0,0],[.8,0,0]],[[.8,0,0],[1.,0,0]]])
  result=f.connected_groups(world,[a,a+np.array([0,.05,0])],self.lib)
  self.assertEqual(result['geometric_connected_group_count'],1)
  self.assertIsNone(result['known_homotopy_count'])
 def test_five_frozen_models_single_cpu_input(self):
  dev=f.read(f.ROOT/'research_outputs/rsps_gate_e_v1/configs/DEVELOPMENT_PROTOCOL.json')
  with np.load(dev['rows'][0]['tensors']['path'],allow_pickle=False) as z:depth=z['depth_input'].copy();state=z['state'].copy()
  raw0=None
  for binding in f.read(f.PREP/'configs/FROZEN_MODELS.json'):
   model=f.frozen(binding);before=f.state_sha(model);raw,out,_=f.predict(model,depth,state)
   self.assertEqual(tuple(raw.shape),(1,6,21));self.assertEqual(tuple(out.trajectory.coeffs.shape),(1,6,3,3,6))
   self.assertTrue(out.trajectory.solve_valid.all());self.assertEqual(before,f.state_sha(model))
   self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
   if binding['label']=='C1':
    raw0=raw.numpy()
    adapter=f.InferenceAdapter(binding['checkpoint']['path'])
    exported=adapter.forward(depth[None],state[None])
    self.assertTrue(np.array_equal(raw0,exported['raw']))
    self.assertTrue(np.array_equal(out.trajectory.coeffs.numpy(),exported['coefficients']))
  self.assertTrue(np.isfinite(raw0).all())

if __name__=='__main__':
 with patch('torch.cuda._lazy_init',side_effect=AssertionError('Codex GPU forbidden')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimizer.step forbidden')):
  unittest.main(verbosity=2)
