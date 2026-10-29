"""Adaptive capacity, detached priority, fairness and gradient contracts."""
import unittest,copy
import torch
from .adaptive_assignment import AdaptiveAssignment,configuration
from .losses import mode_query
from .spec import recipe
class AdaptiveTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def teachers(self,M):
  return [dict(certified=True,representable=True,inner=[[.8,.2*m,0],[1.6,.2*m,0]],tail=[[2.4,.2*m,0],[0,0,0],[0,0,0]],bottlenecks=[]) for m in range(M)]
 def case(self,M=1,safe=None,policy=None,requires=False):
  p=policy or AdaptiveAssignment();inner=torch.zeros(6,2,3,requires_grad=requires);tail=torch.zeros(6,3,requires_grad=requires)
  matches=dict(queries=list(range(M)),teachers=list(range(M)));hp=torch.tensor(safe or [False]*6)
  signed=torch.tensor([[.1,-.2]]*6,requires_grad=requires)
  out=p.choose("train",inner,tail,matches,hp,self.teachers(M),torch.zeros(6,3,3,6),torch.ones(6,3)*.7,signed,0,2)
  return p,out,inner,tail
 def test_M1_capacity(self):
  _,a,_,_=self.case();self.assertEqual(a["occupancy"],[2]);self.assertEqual(len(a["selected"]),1)
 def test_M4_capacity(self):
  _,a,_,_=self.case(4);self.assertTrue(all(x<=2 for x in a["occupancy"]));self.assertEqual(len(a["selected"]),2)
 def test_original_hungarian_reserved(self):
  _,a,_,_=self.case(4);self.assertEqual(a["route_teacher_id"][:4],[0,1,2,3])
 def test_detached_priority(self):
  _,a,_,_=self.case(requires=True)
  self.assertFalse(a["assignment_requires_grad"])
  self.assertTrue(all(type(e["priority"]) is float for r in a["adaptive"] for e in r["edges"]))
 def test_mode_reaches_inner(self):
  _,a,inner,tail=self.case(requires=True);q=a["selected"][0];pva=torch.stack([tail[q],tail[q]*0,tail[q]*0])
  value,_=mode_query(inner[q],pva,torch.ones(3)*.7,self.teachers(1)[0],recipe());value.backward()
  self.assertGreater(float(inner.grad[q].norm()),0.);self.assertTrue(torch.isfinite(inner.grad).all())
 def test_safe_not_pulled(self):
  _,a,_,_=self.case(safe=[False,True,True,True,True,True]);self.assertEqual(a["selected"],[]);self.assertEqual(a["route_teacher_id"][1:],[-1]*5)
 def test_empty_no_pseudo_labels(self):
  p=AdaptiveAssignment();a=p.choose("none",torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[],teachers=[]),torch.zeros(6,dtype=torch.bool),[],torch.zeros(6,3,3,6),torch.ones(6,3),None,0,2)
  self.assertEqual(a["selected"],[]);self.assertEqual(a["route_teacher_id"],[-1]*6)
 def test_uncertified_no_extra_slot(self):
  p=AdaptiveAssignment();t=self.teachers(1);t[0]["certified"]=False
  a=p.choose("none",torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0],teachers=[0]),torch.zeros(6,dtype=torch.bool),t,torch.zeros(6,3,3,6),torch.ones(6,3),None,0,2)
  self.assertEqual(a["selected"],[])
 def test_deterministic_tie(self):
  a=self.case()[1];b=self.case()[1];self.assertEqual(a,b)
 def test_no_starvation(self):
  p=AdaptiveAssignment()
  for _ in range(100):
   self.case(policy=p);p.commit()
  self.assertTrue(all(x["count"]>0 for x in p.state["train"][1:]))
  self.assertLessEqual(max(x["longest"] for x in p.state["train"]),9)
 def test_duplicate_rows_do_not_double_age(self):
  p=AdaptiveAssignment();self.case(policy=p);self.case(policy=p);p.commit()
  self.assertEqual(p.state["train"][1]["eligible"],1);self.assertLessEqual(max(x["count"] for x in p.state["train"]),1)
 def test_state_roundtrip(self):
  p=self.case()[0];p.commit();q=AdaptiveAssignment(state=p.state)
  self.assertEqual(self.case(policy=p)[1],self.case(policy=q)[1])
 def test_no_capacity(self):
  p=AdaptiveAssignment();m=dict(queries=[0,1],teachers=[0,0])
  a=p.choose("full",torch.zeros(6,2,3),torch.zeros(6,3),m,torch.zeros(6,dtype=torch.bool),self.teachers(1),torch.zeros(6,3,3,6),torch.ones(6,3),None,0,2)
  self.assertEqual(a["selected"],[])
if __name__=="__main__":unittest.main()
