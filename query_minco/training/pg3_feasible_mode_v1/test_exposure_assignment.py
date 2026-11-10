"""Capacity, detach, duplicate-batch, transactional and restore invariants."""
import copy,json,unittest
import torch
from .assignment import assign
from .exposure_assignment import ExposureRotation
from .losses import mode_query
from .spec import recipe

def inputs(M=1):
    inner=torch.full((6,2,3),.2,requires_grad=True)
    tail=torch.full((6,3),.3,requires_grad=True)
    ts=[dict(inner=(torch.ones(2,3)*(m+.5)).tolist(),
             tail=torch.stack([torch.ones(3)*(m+.6),torch.zeros(3),torch.zeros(3)]).tolist(),
             durations=[.6,.6,.6],certified=True,representable=True) for m in range(M)]
    matches=dict(queries=list(range(M)),teachers=list(range(M)))
    passed=torch.tensor([True]*M+[False]*(6-M))
    return inner,tail,matches,passed,ts

def choose(p,sid="s",step=0,values=None):
    i,t,m,h,ts=values or inputs()
    return p.choose(sid,i,t,m,h,ts,None,None,None,step,2)

class ExposureTests(unittest.TestCase):
    def test_m1_capacity_and_matching(self):
        p=ExposureRotation();a=choose(p)
        self.assertEqual(a["route_teacher_id"][0],0)
        self.assertEqual(a["occupancy"],[2]);self.assertEqual(len(a["selected"]),1)
    def test_m4_capacity_and_no_safe_pull(self):
        v=inputs(4);v[3][5]=True
        a=choose(ExposureRotation(),values=v)
        self.assertEqual(a["selected"],[4]);self.assertLessEqual(max(a["occupancy"]),2)
        self.assertEqual(a["route_teacher_id"][5],-1)
    def test_empty_teacher_unknown(self):
        i,t,_,h,_=inputs()
        a=choose(ExposureRotation(),values=(i,t,dict(queries=[],teachers=[]),h,[]))
        self.assertEqual(a["selected"],[]);self.assertEqual(a["route_teacher_id"],[-1]*6)
    def test_invalid_or_unrepresentable_not_selected(self):
        for key in ("certified","representable"):
            v=inputs();v[4][0][key]=False
            self.assertEqual(choose(ExposureRotation(),values=v)["selected"],[])
    def test_deterministic_ties(self):
        self.assertEqual(choose(ExposureRotation()),choose(ExposureRotation()))
    def test_no_starvation_under_constant_eligibility(self):
        p=ExposureRotation();seen=set()
        for k in range(5):
            a=choose(p,step=k);seen.update(a["selected"]);p.commit(k)
        self.assertEqual(seen,set(range(1,6)))
        self.assertEqual(p.state["s"]["count"],[0,1,1,1,1,1])
    def test_choose_and_abort_do_not_advance(self):
        p=ExposureRotation();s=p.state_dict();choose(p);p.abort();self.assertEqual(s,p.state_dict())
    def test_duplicate_rows_count_one_update(self):
        p=ExposureRotation();a=choose(p);self.assertEqual(a,choose(p))
        p.commit(0);h=p.state["s"];self.assertEqual(h["exposures"],1)
        self.assertEqual(sum(h["count"]),1)
    def test_inconsistent_duplicate_fails(self):
        p=ExposureRotation();choose(p);v=inputs();v[3][1]=True
        with self.assertRaisesRegex(ValueError,"DUPLICATE_STATE"):choose(p,values=v)
    def test_restore_exact_continuation(self):
        p=ExposureRotation()
        for k in range(3):choose(p,step=k);p.commit(k)
        q=ExposureRotation();q.load_state_dict(json.loads(json.dumps(p.state_dict())))
        self.assertEqual(choose(p,step=3),choose(q,step=3))
        self.assertEqual(p.commit(3),q.commit(3))
    def test_pending_cannot_checkpoint(self):
        p=ExposureRotation();choose(p)
        with self.assertRaises(ValueError):p.state_dict()
    def test_bad_state_rejected(self):
        p=ExposureRotation();choose(p);v=p.commit(0);v["samples"]["s"]["count"][2]=50
        with self.assertRaises(ValueError):ExposureRotation().load_state_dict(v)
    def test_nonsequential_step_rejected(self):
        with self.assertRaises(ValueError):choose(ExposureRotation(),step=2)
    def test_two_samples_independent(self):
        p=ExposureRotation();choose(p,"s");choose(p,"t");p.commit(0)
        a=choose(p,"s",1);b=choose(p,"new",1)
        self.assertNotEqual(a["selected"],b["selected"])
    def test_mode_gradient_and_detached_assignment(self):
        v=inputs();p=ExposureRotation();a=choose(p,values=v);q=a["selected"][0]
        self.assertFalse(a["assignment_requires_grad"])
        tail=torch.stack((v[1][q],torch.zeros(3),torch.zeros(3)))
        loss,_=mode_query(v[0][q],tail,torch.full((3,),.6),v[4][0],recipe())
        g=torch.autograd.grad(loss,(v[0],v[1]))
        self.assertGreater(float(g[0][q].norm()),0.)
        self.assertEqual(float(g[0][0].norm()),0.)
    def test_original_solver_preserved_for_equal_priority(self):
        v=inputs()
        a=assign(*v[:4],v[4],fixed_priority=list(range(6)))
        b=choose(ExposureRotation(),values=v)
        for k in ("route_teacher_id","selected","matched","occupancy"):
            self.assertEqual(a[k],b[k])
if __name__=="__main__":unittest.main()
