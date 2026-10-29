"""Shared-head projection and functional AdamW contracts."""
import unittest,copy,tempfile
from pathlib import Path
import torch
from .head_coordination import coordinate,functional_candidate
class RoutingTests(unittest.TestCase):
 def layout(self):return [("encoder.weight",torch.zeros(2)),("trajectory_head.weight",torch.zeros(2))]
 def test_orthogonal_exact(self):
  g=torch.tensor([1.,2.,3.,4.]);i=torch.tensor([0.,0.,1.,0.]);r=torch.tensor([0.,0.,0.,1.]);x,d=coordinate(g,i,r,self.layout());self.assertTrue(torch.equal(x,g));self.assertFalse(d["applied"])
 def test_opposite_projection(self):
  i=torch.tensor([1.,2.,1.,0.]);r=torch.tensor([3.,4.,-1.,0.]);x,d=coordinate(i,i,r,self.layout());self.assertTrue(torch.equal(x[:2],i[:2]));self.assertLess(abs(float(x[2])),1e-6);self.assertGreater(d["imitation_repair_dot_after"],-1e-6)
 def test_zero_repair(self):
  x=torch.tensor([1.,2.,3.,4.]);y,_=coordinate(x,x,torch.zeros_like(x),self.layout());self.assertTrue(torch.equal(x,y))
 def test_no_conflict_exact(self):
  x=torch.arange(4,dtype=torch.float32);y,_=coordinate(x,x,x,self.layout());self.assertTrue(torch.equal(x,y))
 def test_disabled_exact(self):
  x=torch.arange(4,dtype=torch.float32);y,d=coordinate(x,x,-x,self.layout(),False);self.assertTrue(torch.equal(x,y));self.assertFalse(d["applied"])
 def test_no_duplicate_other_loss(self):
  i=torch.tensor([1.,2.,1.,0.]);r=torch.tensor([0.,0.,-1.,0.]);other=torch.tensor([.1,.2,.3,.4])
  x,_=coordinate(i+r+other,i,r,self.layout());expected=i+r+other;expected[2]-=1.
  self.assertTrue(torch.allclose(x,expected,atol=1e-6))
 def test_score_rows_zero(self):
  p=[("trajectory_head.weight",torch.zeros(21,2))];i=torch.zeros(42);r=i.clone();i[0]=1;r[0]=-1
  x,_=coordinate(i+r,i,r,p);self.assertFalse(torch.count_nonzero(x.reshape(21,2)[19:]))
 def test_illegal_score_rejected(self):
  p=[("trajectory_head.bias",torch.zeros(21))];x=torch.zeros(21);x[20]=1
  with self.assertRaisesRegex(ValueError,"SCORE_GRADIENT"):coordinate(x,x,torch.zeros_like(x),p)
 def test_nonfinite_rejected(self):
  x=torch.zeros(4);x[0]=float("nan")
  with self.assertRaisesRegex(ValueError,"NONFINITE"):coordinate(x,x,x,self.layout())
 def test_functional_matches_adam_clip_and_safe_reload(self):
  torch.manual_seed(7);m=torch.nn.Linear(2,2);o=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=0.)
  for index in range(3):
   o.zero_grad();m(torch.ones(2,2)).square().sum().backward();o.step()
  before=copy.deepcopy(o.state_dict());params=[(n,p.detach().clone()) for n,p in m.named_parameters()]
  g=torch.cat([torch.arange(p.numel(),dtype=p.dtype).reshape(-1)+1 for p in m.parameters()])*100
  weights,state,delta,rec=functional_candidate(params,before,g,lr=1.3e-5)
  off=0
  for p in m.parameters():p.grad=g[off:off+p.numel()].reshape_as(p).clone();off+=p.numel()
  torch.nn.utils.clip_grad_norm_(m.parameters(),10.);o.param_groups[0]["lr"]=1.3e-5;o.step()
  self.assertTrue(rec["clipped"])
  for name,p in m.named_parameters():self.assertTrue(torch.equal(weights[name],p))
  for pid,s in o.state_dict()["state"].items():
   for k,v in s.items():self.assertTrue(torch.equal(v,state["state"][pid][k]))
  for pid,s in before["state"].items():self.assertEqual(int(s["step"]),3)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"temporary.pt";torch.save(dict(weights=weights,optimizer=state),p);x=torch.load(p,weights_only=True)
   for n in weights:self.assertTrue(torch.equal(x["weights"][n],weights[n]))
 def test_functional_zero_state(self):
  m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=0.);g=torch.ones(sum(p.numel() for p in m.parameters()))*.001
  w,s,d,r=functional_candidate(list(m.named_parameters()),o.state_dict(),g,2e-5)
  self.assertTrue(all(int(v["step"])==1 for v in s["state"].values()));self.assertEqual(len(o.state),0)
if __name__=="__main__":unittest.main()
