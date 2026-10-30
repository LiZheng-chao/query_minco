"""Finite differences, transaction replay, identity and end-to-end regressions."""
import copy,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.pg3_v2_no_corridor.revision import PerSampleFairAssignment,FrozenSafeReference,section_side_proxy,full_stopping_risk
from query_minco.training.pg3_v2_no_corridor.objective import Objective,KEYS,reliable_target,diversity
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from experiments.pg3_feasible_mode_v1.common import raw_base
from query_minco.training.pg2_guided_v2.spec import COMPONENTS,FIXED

def curve():
    c=torch.zeros(3,3,6,dtype=torch.double);c[:,0,0]=torch.tensor([0.,.8,1.6]);c[:,0,1]=2.;c[:,1,0]=.1;c[:,1,1]=.2
    return c,torch.full((3,),.4,dtype=torch.double)
def teacher():
    return dict(inner=[[.8,.5,0],[1.6,.5,0]],tail=[[2.4,.5,0],[0,0,0],[0,0,0]],representable=True,certified=True)
def label(progress=.43):
    return dict(valid=True,side=[1,1],box=0,plane_x=1.05,progress=progress,event_status="UNIQUE_FORWARD_CROSSING")
def scene():
    return dict(obstacles=[dict(min=[1.,-.2,-.2],max=[1.1,.2,.2])],flight_center_domain_m=[[0,-1,-1],[3,1,1]])
def fixture():
    c,d=curve();c[:,1]=0
    state=torch.zeros(12);state[0]=2.;state[6]=2.4
    s=NS(sample_id="synthetic",depth_input=torch.zeros(2,128,160),state=state,
         teacher_coeffs=c[None].float(),teacher_durations=d[None].float())
    f=NS(row=dict(obstacles=[],flight_center_domain_m=[[0.,-1.,-1.],[10.,1.,1.]],empty_certified=True),boxes=torch.empty(0,2,3))
    return s,f
class V2Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
    def test_section_implicit_coeff_duration_fd(self):
        c,d=curve();c.requires_grad_();d.requires_grad_()
        fn=lambda c,d:section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True,atol=1e-5))
        loss,info=section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())
        self.assertEqual(info["statuses"],["LOCAL_IMPLICIT_SECTION_PROXY"])
        self.assertLess(float(torch.autograd.grad(loss,c)[0][:,1,0].sum()),0)
    def test_section_fallback_fd(self):
        c,d=curve();c[:,0]=0;c[:,0,0]=.5;c.requires_grad_();d.requires_grad_()
        fn=lambda c,d:section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True))
        self.assertEqual(section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())[1]["statuses"],["TEACHER_PROGRESS_FALLBACK"])
    def test_section_slow_root_fallback_has_axial(self):
        c,d=curve();c[:,0]=0;c[0,0,0]=1.05-1e-6;c[0,0,1]=1e-5;c[1:,0,0]=1.06
        loss,info=section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())
        self.assertEqual(info["statuses"],["TEACHER_PROGRESS_FALLBACK"])
        c.requires_grad_();g=torch.autograd.grad(section_side_proxy(c,d,dict(bottlenecks=[label()]),scene())[0],c)[0]
        self.assertGreater(float(g[:,0].norm()),0.)
    def test_section_segment_join_and_multiple_root_fallback(self):
        c,d=curve();l=label();l["plane_x"]=.8
        self.assertEqual(section_side_proxy(c,d,dict(bottlenecks=[l]),scene())[1]["statuses"],["TEACHER_PROGRESS_FALLBACK"])
        c[:,0]=0;c[:,0,0]=2
        c[0,0,:4]=torch.tensor([-.00225,.0675,-.5,1.],dtype=torch.double)
        l["plane_x"]=0
        self.assertEqual(section_side_proxy(c,d,dict(bottlenecks=[l]),scene())[1]["statuses"],["TEACHER_PROGRESS_FALLBACK"])
    def test_section_negative_z_exact_si_formula(self):
        c,d=curve();c[:,2,0]=-.25;l=label();l["side"]=[2,-1]
        value,_=section_side_proxy(c,d,dict(bottlenecks=[l]),scene())
        # min_z=-.2 => bound=-.35, gap=-.10 => ((.03+.10)/.15)^2.
        self.assertAlmostEqual(float(value),(.13/.15)**2,places=6)
    def test_invalid_teacher_event_and_axis(self):
        c,d=curve();bad=label();bad["event_status"]="MULTIPLE_FORWARD_CROSSINGS"
        self.assertEqual(float(section_side_proxy(c,d,dict(bottlenecks=[bad]),scene())[0]),0.)
        bad=label();bad["side"]=[0,1]
        with self.assertRaises(ValueError):section_side_proxy(c,d,dict(bottlenecks=[bad]),scene())
    def test_stop_corner_position_velocity_weights_fd(self):
        p=torch.tensor([[[[.13,.27,.6],[.22,.32,.65]]]*6],dtype=torch.double,requires_grad=True)
        v=torch.full_like(p,-.7,requires_grad=True);w=torch.full(p.shape[:-1],.5,dtype=torch.double,requires_grad=True)
        f=NS(boxes=torch.tensor([[[-.1,-.1,-.1],[.1,.1,.1]]],dtype=torch.double),row=dict(empty_certified=False))
        self.assertTrue(torch.autograd.gradcheck(lambda p,v,w:full_stopping_risk(p,v,w,[f]),(p,v,w),fast_mode=True,atol=1e-4))
    def test_stop_inside_finite_and_uncapped(self):
        p=torch.zeros(1,6,2,3,dtype=torch.double,requires_grad=True);p.data[...,0]=.1
        v=torch.zeros_like(p,requires_grad=True);v.data[...,0]=-4
        w=torch.ones(1,6,2,dtype=torch.double)
        f=NS(boxes=torch.tensor([[[-.2,-.3,-.4],[.2,.3,.4]]],dtype=torch.double),row=dict(empty_certified=False))
        r=full_stopping_risk(p,v,w,[f]);g=torch.autograd.grad(r,(p,v))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in g));self.assertGreater(float(r),1000)
    def test_stop_empty_and_bad_mass(self):
        p=torch.zeros(1,6,2,3,requires_grad=True);v=p*0;w=torch.ones(1,6,2)
        f=NS(boxes=torch.empty(0,2,3),row=dict(empty_certified=False))
        with self.assertRaises(ValueError):full_stopping_risk(p,v,w,[f])
        f.row["empty_certified"]=True
        self.assertEqual(float(full_stopping_risk(p,v,w,[f])),0.)
        with self.assertRaises(ValueError):full_stopping_risk(p,v,w*0,[f])
    def choose(self,a,step=0,sid="s",safe=None):
        return a.choose(sid,torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0],teachers=[0]),
                        torch.zeros(6,dtype=torch.bool) if safe is None else safe,[teacher()],None,None,None,step,2)
    def test_duplicate_fairness_rollback_commit_and_resume(self):
        a=PerSampleFairAssignment();chosen=[self.choose(a)["selected"][0] for _ in range(5)]
        self.assertEqual(chosen,[1,2,3,4,5]);self.assertEqual(a.history,{})
        with self.assertRaises(ValueError):a.state_dict()
        a.discard();self.assertEqual(self.choose(a)["selected"],[1]);a.commit(0)
        state=a.state_dict();b=PerSampleFairAssignment();b.load_state_dict(state)
        self.assertEqual(self.choose(a,1),self.choose(b,1));a.commit(1);b.commit(1)
        self.assertEqual(a.state_dict(),b.state_dict())
        with self.assertRaises(ValueError):a.commit(1)
    def test_duplicate_counts_are_occurrences(self):
        a=PerSampleFairAssignment()
        for _ in range(10):self.choose(a)
        a.commit(0);self.assertEqual([r["direct"] for r in a.history["s"]],[0,2,2,2,2,2])
    def test_safe_unmatched_and_capacity(self):
        a=PerSampleFairAssignment();r=self.choose(a,safe=torch.ones(6,dtype=torch.bool))
        self.assertEqual(r["selected"],[]);self.assertEqual(r["route_teacher_id"],[0,-1,-1,-1,-1,-1])
    def test_reference_identity_cache_and_gradient_isolation(self):
        m=QueryMincoPlanner();identity=state_sha(m);s,f=fixture()
        with self.assertRaises(ValueError):FrozenSafeReference(m,expected_sha="0"*64)
        ref=FrozenSafeReference(m,expected_sha=identity);first=ref._reference(s,f)
        self.assertIs(first,ref._reference(s,f));self.assertFalse(any(p.requires_grad for p in ref.model.parameters()))
        s.state[0]+=.1
        with self.assertRaises(ValueError):ref._reference(s,f)
        self.assertEqual(state_sha(m),identity)
    def test_keep_hinge_fd_safe_only_global_mean(self):
        m=QueryMincoPlanner();ref=FrozenSafeReference(m,expected_sha=state_sha(m))
        inner=torch.full((1,6,2,3),.1,dtype=torch.double,requires_grad=True)
        tail=torch.full((1,6,3,3),.3,dtype=torch.double,requires_grad=True)
        d=torch.full((1,6,3),.5,dtype=torch.double,requires_grad=True)
        row=dict(safe=torch.tensor([True,False,False,False,False,False]),inner=torch.zeros(6,2,3),tail=torch.zeros(6,3,3),durations=torch.full((6,3),.4))
        def value(a,b,c):return ref.loss(dict(output=NS(parameters=NS(inner_pts=a,tail_pva=b,durations=c))),[None],[None])[0]
        with patch.object(ref,"_reference",return_value=row):
            self.assertTrue(torch.autograd.gradcheck(value,(inner,tail,d),fast_mode=True))
            g=torch.autograd.grad(value(inner,tail,d),(inner,tail,d))
            self.assertTrue(all(float(x[:,1:].norm())==0 for x in g))
    def test_no_corridor_teacher_and_disabled_exact_old_objective(self):
        s,f=fixture();m=QueryMincoPlanner();weights=dict({k:1. for k in COMPONENTS},**FIXED)
        raw=m(depth_input=s.depth_input[None],state=s.state[None]);base=raw_base(raw,s,f,weights)
        obj=Objective(m,state_sha(m),{k:0. for k in KEYS})
        with patch.object(obj.bank,"get",side_effect=AssertionError("must skip")):
            z=obj.augment(base,[s]);self.assertIs(z["loss"],base["loss"])
        ga=torch.autograd.grad(z["loss"],raw,retain_graph=True)[0];gb=torch.autograd.grad(base["loss"],raw)[0]
        self.assertTrue(torch.equal(ga,gb));self.assertEqual(float(ga[...,19:].norm()),0.)
        with patch("query_minco.training.pg3_feasible_mode_v1.teacher_labels.build_corridor",side_effect=AssertionError("corridor called")):
            bank=TeacherBank(dict(recipe(),build_corridor=False));self.assertEqual(bank.get(s,f)["teachers"][0]["corridor"]["status"],"V2_DISABLED")
    def test_network_minco_full_v2_backward_and_frozen_scores(self):
        s,f=fixture();m=QueryMincoPlanner();identity=state_sha(m);obj=Objective(m,identity,{k:.01 for k in KEYS})
        raw=m(depth_input=s.depth_input[None],state=s.state[None]);base=raw_base(raw,s,f,dict({k:1. for k in COMPONENTS},**FIXED))
        with patch("query_minco.training.pg3_feasible_mode_v1.teacher_labels.build_corridor",side_effect=AssertionError("corridor called")):
            z=obj.augment(base,[s]);z["loss"].backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
        self.assertEqual(float(next(p.grad for n,p in m.named_parameters() if n.startswith("trajectory_head.") and p.shape[0]==21)[19:].norm()),0.)
        self.assertTrue(all(p.grad is None for p in obj.reference.model.parameters()))
        self.assertEqual(state_sha(obj.reference.model),identity)
    def test_diversity_fd_and_exact_collapse_gradient(self):
        c,d=curve();c=c[None,None].repeat(1,6,1,1,1).requires_grad_();d=d[None,None].repeat(1,6,1).requires_grad_()
        a=dict(selected=[1,2],route_teacher_id=[-1,0,1,-1,-1,-1])
        x=dict(label(),position=[1.05,.6,0],normal=[1,0,0]);y=dict(x,side=[1,-1],position=[1.05,-.6,0])
        banks=[dict(teachers=[dict(mode_signature=[(0,1,1)],bottlenecks=[x]),dict(mode_signature=[(0,1,-1)],bottlenecks=[y])])]
        fn=lambda c,d:diversity(NS(coeffs=c,durations=d),[a],banks,.5)[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True))
        self.assertGreater(float(torch.autograd.grad(fn(c,d),c)[0].norm()),0.)
    def test_initial_matched_safe_uses_initial_query_identity(self):
        import tempfile,csv
        from pathlib import Path
        from experiments.pg3_v2_no_corridor.run import transition
        with tempfile.TemporaryDirectory() as folder:
            files=[Path(folder)/"a.csv",Path(folder)/"b.csv"]
            data=[[dict(sample_id="s",query=0,matched=True,hard_pass=True),dict(sample_id="s",query=1,matched=False,hard_pass=False)],
                  [dict(sample_id="s",query=0,matched=False,hard_pass=False),dict(sample_id="s",query=1,matched=True,hard_pass=True)]]
            for path,rows in zip(files,data):
                with path.open("w") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            result=transition(*files)
            self.assertEqual(result["initial_matched_safe_degraded"],1)
            self.assertEqual((result["fail_to_pass"],result["pass_to_fail"]),(1,1))
    def test_diversity_unreliable_labels_removed(self):
        t=dict(teacher(),bottlenecks=[dict(label(),event_status="MULTIPLE_FORWARD_CROSSINGS")])
        self.assertEqual(reliable_target(t)["mode_signature"],[])
if __name__=="__main__":unittest.main()
