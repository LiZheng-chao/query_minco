"""PG3-only repair tests; tiny optimizer tests are isolated from network experiments."""
import tempfile,unittest,copy
from pathlib import Path
import torch
from .assignment import assign
from .state_io import save_commit,restore,lr_at
from query_minco.training.pg1.state_io import presets
from query_minco.training.io import read
class RepairTests(unittest.TestCase):
 def teacher(self):return dict(inner=[[.8,.4,0],[1.6,.4,0]],tail=[[2.4,0,0],[0,0,0],[0,0,0]],representable=True,certified=True)
 def test_fixed_capacity_and_rotation(self):
  p=torch.zeros(6,2,3);tail=torch.zeros(6,3);hp=torch.tensor([False]*5+[True]);matches=dict(queries=[5],teachers=[0]);ts=[self.teacher()]
  a=assign(p,tail,matches,hp,ts,step=0)
  b=assign(p,tail,matches,hp,ts,step=1)
  self.assertNotEqual(a["selected"],b["selected"])
  for step in range(5):
   x=assign(p,tail,matches,hp,ts,step=step,fixed_priority=[3,1,0,2,4])
   self.assertEqual(x["selected"],[3]);self.assertEqual(x["occupancy"],[2]);self.assertEqual(x["route_teacher_id"][5],0)
 def test_fixed_does_not_pull_safe_query(self):
  hp=torch.tensor([True,False,False,False,False,True])
  x=assign(torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[5],teachers=[0]),hp,[self.teacher()],fixed_priority=[0,2])
  self.assertEqual(x["selected"],[2]);self.assertEqual(x["route_teacher_id"][0],-1)
 def test_safe_roundtrip_optimizer_rng_scheduler(self):
  for schedule in ("original_cosine_2e-5_1e-6_32","preregistered_cosine_2e-4_1e-5_128"):
   with tempfile.TemporaryDirectory() as d:
    m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=lr_at(schedule,0));s=dict(device="cpu",schedule=schedule)
    for i in range(2):
     o.zero_grad();m(torch.ones(1,2)).square().sum().backward()
     for g in o.param_groups:g["lr"]=lr_at(schedule,i)
     o.step()
    path=save_commit(Path(d),m,o,s,2,dict(step=2))
    expected=torch.rand(3);mm=torch.nn.Linear(2,1);oo=torch.optim.AdamW(mm.parameters(),lr=.1)
    blob,c=restore(path,mm,oo,s,True)
    self.assertTrue(torch.equal(expected,torch.rand(3)))
    self.assertEqual(blob["scheduler"],schedule);self.assertEqual(blob["next_lr_index"],2)
    self.assertTrue(all(torch.equal(m.state_dict()[k],mm.state_dict()[k]) for k in m.state_dict()))
    self.assertTrue(all(int(v["step"])==2 for v in oo.state.values()))
    self.assertEqual(oo.param_groups[0]["lr"],lr_at(schedule,1))
    for p in (m,mm):
     for q in p.parameters():q.grad=None
    for p in (m,mm):p(torch.ones(1,2)).square().sum().backward()
    for opt in (o,oo):
     opt.param_groups[0]["lr"]=lr_at(schedule,2);opt.step()
    self.assertTrue(all(torch.equal(m.state_dict()[k],mm.state_dict()[k]) for k in m.state_dict()))
 def test_numpy_lr_fails_closed(self):
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters(),lr=1e-5);o.param_groups[0]["lr"]=np.float64(1e-5)
   with self.assertRaisesRegex(ValueError,"STANDARD_LR"):
    save_commit(Path(d),m,o,dict(device="cpu",schedule="original_cosine_2e-5_1e-6_32"),0,{})
 def test_scheduler_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters(),lr=2e-5);s=dict(device="cpu",schedule="invalid")
   with self.assertRaisesRegex(ValueError,"SCHEDULE"):save_commit(Path(d),m,o,s,0,{})
if __name__=="__main__":unittest.main()
