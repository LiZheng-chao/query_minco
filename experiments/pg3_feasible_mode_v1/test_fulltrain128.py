"""FullTrain128 execution-profile tests; no Query network training."""
import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from .fulltrain128 import lr_at,METHODS,SCHEDULE,committed
from .fulltrain128_evaluation import integral,signature
from query_minco.training.pg3_feasible_mode_v1.spec import effective
from query_minco.training.pg1.state_io import inspect_commit
from query_minco.training.checkpoints import load_checkpoint
class FullTrainTests(unittest.TestCase):
 def test_schedule_endpoints_monotone(self):
  values=[lr_at(i) for i in range(128)];self.assertEqual(values[0],2e-5);self.assertEqual(values[-1],1e-6);self.assertTrue(all(type(x) is float for x in values));self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
 def test_schedule_invalid_index(self):
  for i in (-1,128):
   with self.assertRaises(ValueError):lr_at(i)
 def test_method_switches(self):
  weights={k:1. for k in ("mode","bottleneck","convex_corridor","qa","qj","qt","lateral","diversity")}
  b=effective(weights,METHODS["B"]);c=effective(weights,METHODS["C"])
  self.assertEqual({k for k,v in b.items() if v},{"mode"});self.assertEqual({k for k,v in c.items() if v},{"mode","bottleneck","convex_corridor"})
 def test_exact_quality_integrals(self):
  c=np.zeros((3,3,6));d=np.array([.7,.7,.7]);c[:,0,2]=1.;self.assertAlmostEqual(integral(c,d,2),8.4);self.assertEqual(integral(c,d,3),0.)
  c[:]=0.;c[:,0,3]=1.;self.assertAlmostEqual(integral(c,d,3),75.6)
 def test_route_is_geometric_not_teacher_id(self):
  c=np.zeros((3,3,6));d=np.full(3,.7);c[:,0,0]=[0,.7,1.4];c[:,0,1]=1.;c[:,1,0]=.7;scene=dict(obstacles=[dict(name="central",min=[.9,-.2,-.2],max=[1.1,.2,.2])])
  self.assertIn('central',signature(c,d,scene));c[:,1,0]=0.;self.assertIsNone(signature(c,d,scene))
 def test_full_schedule_checkpoint_safe_roundtrip(self):
  with tempfile.TemporaryDirectory() as folder:
   m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=lr_at(0),weight_decay=0.);o.zero_grad();m(torch.ones(1,2)).square().sum().backward();o.step()
   spec=dict(device="cpu",schedule=SCHEDULE);p=committed(Path(folder),m,o,spec,1,dict(actual=1));inspect_commit(p,spec)
   mm=torch.nn.Linear(2,1);oo=torch.optim.AdamW(mm.parameters(),lr=.1);blob=load_checkpoint(p/"checkpoint_step0001.pt",mm,oo,spec,False)
   self.assertEqual(blob["scheduler"],SCHEDULE);self.assertTrue(all(int(s["step"])==1 for s in oo.state.values()))
   self.assertTrue(all(torch.equal(m.state_dict()[n],mm.state_dict()[n]) for n in m.state_dict()))
if __name__=="__main__":unittest.main()
