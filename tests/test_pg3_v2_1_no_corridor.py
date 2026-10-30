"""PRD acceptance T01--T19. Real-data tests consume committed qualification evidence."""
import copy,hashlib,json,os,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import digest,read,artifact,sha
from query_minco.training.initialization import head_rows
from query_minco.training.pg3_v2_no_corridor.objective import ObjectiveV21,KEYS_V21,geometry_query_v21
from query_minco.training.pg3_v2_no_corridor.revision import (section_events_v21,full_stopping_risk_v21,UniqueSampleFairAssignment,
    FrozenSafeReference,save_v21_checkpoint,load_v21_checkpoint,optimizer_step_v21)
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,make_keep_probe_v21,ratios_from_gram_v21
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import inverse_raw
from query_minco.training.pg3_feasible_mode_v1.bottleneck import bottleneck_loss
from query_minco.training.pg2_guided_v2.spec import COMPONENTS,FIXED
from query_minco.training.pg1_v2_r12.diagnostics import flat
from query_minco.training.run_spec import ROOT
from experiments.pg3_feasible_mode_v1.common import raw_base

BASE="651152c1b3214101b54536c6904f4f38f13ceb57"
def curve():
    c=torch.zeros(3,3,6,dtype=torch.double);c[:,0,0]=torch.tensor([0.,.8,1.6]);c[:,0,1]=2.;c[:,1,0]=.1
    return c,torch.full((3,),.4,dtype=torch.double)
def scene():return dict(obstacles=[dict(name="box",min=[1.,-.2,-.2],max=[1.1,.2,.2])],flight_center_domain_m=[[0,-1,-1],[3,1,1]])
def label(x=1.05):return dict(valid=True,side=[1,1],box=0,plane_x=x,progress=x/2.4,event_status="UNIQUE_FORWARD_CROSSING",position=[x,.5,0],normal=[1,0,0])
def teacher(labels=None):return dict(certified=True,representable=True,inner=[[.8,0,0],[1.6,0,0]],tail=[[2.4,0,0],[2.,0,0],[0,0,0]],bottlenecks=[label()] if labels is None else labels)
class RawModel(torch.nn.Module):
    def __init__(self):
        super().__init__();t=teacher();raw=torch.tensor(inverse_raw(t["inner"],t["tail"],[.4,.4,.4]),dtype=torch.float32)
        self.theta=torch.nn.Parameter(raw[None,None].repeat(1,6,1));self.theta.data[...,:9]+=.004
    def forward(self,depth_input,state):return self.theta.expand(len(state),-1,-1)
def fixture():
    c,d=curve();c[:,1]=0.;state=torch.zeros(12);state[0]=2.;state[6]=4.0
    s=NS(sample_id="s",state=state,depth_input=torch.zeros(2,128,160),teacher_coeffs=c[None].float(),teacher_durations=d[None].float())
    boxes=[dict(name="offset",min=[1.,.4,-.2],max=[1.1,.6,.2])]
    f=NS(row=dict(obstacles=boxes,flight_center_domain_m=[[0.,-1.,-1.],[3.,1.,1.]],empty_certified=False),boxes=torch.tensor([[b["min"],b["max"]] for b in boxes]))
    return s,f

def base_forward(m):
    s,f=fixture();raw=m(s.depth_input[None],s.state[None]);return raw_base(raw,s,f,dict({k:1. for k in COMPONENTS},**FIXED)),s,f

def evidence():
    value=os.environ.get("PG3_V21_EVIDENCE")
    if not value:raise unittest.SkipTest("Final committed real Training384 evidence required")
    return Path(value)

class V21Acceptance(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
    def choose(self,a,sid="s",auth=None,hard=None,teachers=None):
        return a.choose_once(sid,torch.zeros(6,2,3),torch.zeros(6,3),dict(queries=[0],teachers=[0]),
            torch.zeros(6,dtype=torch.bool) if hard is None else hard,[teacher([])] if teachers is None else teachers,
            authenticated_digest=digest(dict(input="fixed")) if auth is None else auth)
    def commit(self,a,step):return a.commit_after_optimizer_step(step,optimizer_succeeded=True,finite=True,scores_unchanged=True)

    def test_T01_original_PG2_identity(self):
        for folder in ("query_minco/training/pg2_guided_v2","query_minco/training/pg3_feasible_mode_v1"):
            names=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,folder],cwd=str(ROOT),text=True).splitlines()
            for name in names:self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(["git","show",BASE+":"+name],cwd=str(ROOT)))
        m=RawModel();a,_,_=base_forward(m);b,_,_=base_forward(m)
        self.assertTrue(torch.equal(a["loss"],b["loss"]))
        ga=torch.autograd.grad(a["loss"],(a["raw"],m.theta),retain_graph=True)
        gb=torch.autograd.grad(b["loss"],(b["raw"],m.theta),retain_graph=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))

    def test_T02_legacy_all_zero_exact(self):
        m=RawModel();b,s,_=base_forward(m);obj=ObjectiveV21(m,state_sha(m),{k:0. for k in KEYS_V21},"legacy")
        with patch.object(obj.bank,"get",side_effect=AssertionError("new work forbidden")):z=obj.augment(b,[s])
        self.assertIs(z["loss"],b["loss"])
        self.assertTrue(torch.equal(torch.autograd.grad(z["loss"],m.theta,retain_graph=True)[0],torch.autograd.grad(b["loss"],m.theta)[0]))
        with self.assertRaises(ValueError):ObjectiveV21(m,state_sha(m),{k:0. for k in KEYS_V21},"replace_full_stop")

    def test_T03_weighted_braking_replacement(self):
        m=RawModel();b,s,_=base_forward(m);obj=ObjectiveV21(m,state_sha(m),{k:.1 for k in KEYS_V21})
        z=obj.augment(b,[s]);expanded=b["loss"]-b["weights"]["braking"]*b["terms"]["braking"]+sum(z["weights"][k]*z["terms"][k] for k in KEYS_V21)
        self.assertTrue(torch.equal(z["loss"],expanded));rec,_,_=audit_v21(m,z,False)
        self.assertEqual(rec["components"]["pg2/braking"]["effective_weight"],0.)
        self.assertGreater(b["weights"]["braking"],0.);obj.assignment.rollback(0)

    def test_T04_section_exact_015_all_faces(self):
        for axis in (1,2):
            for sign in (-1,1):
                for distance in (.14,.15,.16):
                    c,d=curve();c[:,axis,0]=sign*(.2+distance);l=label();l["side"]=[axis,sign]
                    value,info=section_events_v21(c,d,teacher([l]),scene())
                    self.assertEqual(info["count"],1);self.assertAlmostEqual(float(value),(max(0.,.15-distance)/.15)**2,places=13)

    def test_T05_unique_section_exclusive_finite_difference(self):
        c,d=curve();c.requires_grad_();d.requires_grad_()
        def fn(c,d):return geometry_query_v21(c,d,teacher(),scene(),matched=False,selected=True,hard_pass=False)[0]
        self.assertEqual(geometry_query_v21(c,d,teacher(),scene(),matched=False,selected=True,hard_pass=False)[1]["geom_source"],"SECTION")
        with patch("query_minco.training.pg3_v2_no_corridor.objective.bottleneck_loss",side_effect=AssertionError("double supervision")):fn(c,d)
        self.assertTrue(torch.autograd.gradcheck(fn,(c,d),fast_mode=True,atol=1e-5))

    def test_T06_invalid_crossing_bneck_fallback(self):
        for kind in ("missing","tangent","multiple","backward","join","slow"):
            c,d=curve();l=label()
            if kind=="missing":c[:,0]=0;c[:,0,0]=.5
            elif kind=="tangent":c[:,0]=0;c[:,0,0]=2;c[0,0,:3]=torch.tensor([l["plane_x"]+.04,-.4,1.])
            elif kind=="multiple":c[:,0]=0;c[:,0,0]=2;c[0,0,:4]=torch.tensor([l["plane_x"]-.00225,.0675,-.5,1.])
            elif kind=="backward":c[:,0,1]=-2;c[:,0,0]=torch.tensor([1.5,.7,-.1])
            elif kind=="join":l["plane_x"]=.8
            else:c[:,0]=0;c[:,0,0]=2;c[0,0,0]=l["plane_x"]-1e-6;c[0,0,1]=1e-5
            c.requires_grad_();d.requires_grad_()
            _,info=geometry_query_v21(c,d,teacher([l]),scene(),matched=False,selected=True,hard_pass=False)
            self.assertEqual(info["geom_source"],"BNECK_FALLBACK",kind)
            self.assertFalse(info["section"]["valid"])
        l=label();l["valid"]=False
        self.assertEqual(geometry_query_v21(c,d,teacher([l]),scene(),matched=False,selected=True,hard_pass=False)[1]["geom_source"],"NONE")

    def test_T07_matched_and_exclusions_mode_scope(self):
        c,d=curve()
        self.assertEqual(geometry_query_v21(c,d,teacher(),scene(),matched=True,selected=False,hard_pass=False)[1]["geom_source"],"BNECK_MATCHED")
        for target,matched,selected,hard in ((teacher(),False,True,True),(None,False,True,False),(dict(teacher(),certified=False),False,True,False),(teacher(),False,False,False)):
            self.assertEqual(geometry_query_v21(c,d,target,scene(),matched=matched,selected=selected,hard_pass=hard)[1]["geom_source"],"NONE")
        a=UniqueSampleFairAssignment();a.begin_step(0,["s"]);r=self.choose(a,hard=torch.tensor([False,True,False,False,False,False]))
        self.assertEqual(r["selected"],[2]);self.assertEqual(r["occupancy"],[2]);a.rollback(0)

    def test_T08_multiple_event_reduction_empty_zero(self):
        c,d=curve();c.requires_grad_();d.requires_grad_();labels=[label(1.05),label(1.2)]
        value,info=section_events_v21(c,d,teacher(labels),scene())
        separate=[section_events_v21(c,d,teacher([l]),scene())[0] for l in labels]
        self.assertTrue(torch.allclose(value,sum(separate)/2));self.assertEqual(info["event_denominator"],2)
        zero,zd=geometry_query_v21(c,d,teacher([]),scene(),matched=True,selected=False,hard_pass=False)
        self.assertEqual(zd["geom_source"],"NONE");self.assertEqual(float(zero),0.)
        self.assertTrue(torch.equal(torch.autograd.grad(zero,c)[0],torch.zeros_like(c)))

    def test_T09_fullstop_formula_finite_and_uncapped(self):
        p=torch.tensor([[[[.6,.02,0.],[.63,.03,0.]]]*6],dtype=torch.double,requires_grad=True)
        v=torch.zeros_like(p,requires_grad=True);v.data[...,0]=-4.;w=torch.ones(1,6,2,dtype=torch.double,requires_grad=True)
        f=NS(boxes=torch.tensor([[[-.1,-.2,-.3],[.1,.2,.3]]],dtype=torch.double),row=dict(empty_certified=False))
        risk=full_stopping_risk_v21(p,v,w,[f]);h=.15+16/8-(p[...,0]-.1)
        expected=((.05*torch.nn.functional.softplus(h/.05)/.15)**2).mean()
        self.assertTrue(torch.allclose(risk,expected));self.assertGreater(float(risk),100.)
        self.assertTrue(torch.autograd.gradcheck(lambda p,v,w:full_stopping_risk_v21(p,v,w,[f]),(p,v,w),fast_mode=True))
        for point in ([.01,.02,.03],[.15,.25,.4],[.0,.0,.0]):
            pp=torch.tensor(point,dtype=torch.double).expand(1,6,2,3).clone().requires_grad_();vv=torch.zeros_like(pp,requires_grad=True)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in torch.autograd.grad(full_stopping_risk_v21(pp,vv,w,[f]),(pp,vv))))
        corner=torch.tensor([.15,.25,.4],dtype=torch.double).expand(1,6,2,3).clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:full_stopping_risk_v21(a,b,w,[f]),(corner,v),fast_mode=True,atol=1e-4))
        overlap=NS(boxes=torch.cat([f.boxes,f.boxes+torch.tensor([.2,0.,0.],dtype=torch.double)]),row=dict(empty_certified=False))
        middle=torch.tensor([.1,.02,.03],dtype=torch.double).expand(1,6,2,3).clone().requires_grad_()
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(full_stopping_risk_v21(middle,v,w,[overlap]),middle)[0]).all()))
        f.boxes=torch.empty(0,2,3,dtype=torch.double)
        with self.assertRaises(ValueError):full_stopping_risk_v21(p,v,w,[f])
        f.row["empty_certified"]=True;self.assertEqual(float(full_stopping_risk_v21(p,v,w,[f])),0.)

    def test_T10_keep_initial_zero_and_active_safe_mask(self):
        m=RawModel();b,s,f=base_forward(m);ref=FrozenSafeReference(m,expected_sha=state_sha(m))
        value,_=ref.loss(b,[s],[f]);self.assertEqual(float(value),0.)
        self.assertEqual(float(torch.autograd.grad(value,m.theta)[0].norm()),0.)
        params=b["output"].parameters
        inner=params.inner_pts.detach().clone().requires_grad_();tail=params.tail_pva.detach().clone().requires_grad_();dur=params.durations.detach().clone().requires_grad_()
        inner.data+=.2;mask=torch.tensor([True,False,False,False,False,False]);ref.cache[s.sample_id]["safe"]=mask
        z=dict(output=NS(parameters=NS(inner_pts=inner,tail_pva=tail,durations=dur)))
        active,_=ref.loss(z,[s],[f]);g=torch.autograd.grad(active,inner)[0]
        self.assertGreater(float(g[:,0].norm()),0);self.assertEqual(float(g[:,1:].norm()),0)
        self.assertTrue(all(p.grad is None for p in ref.model.parameters()))

    def test_T11_keep_shadow_probe_identity_and_activation(self):
        m=QueryMincoPlanner();before=state_sha(m);opt=torch.optim.AdamW(m.parameters());state=copy.deepcopy(opt.state_dict())
        cfg=dict(bias_indices=[1,4,7],bias_offset=.12,status="PROBE_ONLY")
        a,meta=make_keep_probe_v21(m,cfg);b,_=make_keep_probe_v21(m,cfg)
        self.assertEqual(state_sha(m),before);self.assertEqual(state_sha(a),state_sha(b));self.assertEqual(opt.state_dict(),state)
        ev=evidence();doc=read(ev/"GRADIENT_AUDIT_V2_1.json")
        self.assertTrue(doc["keep_probe_active"]);self.assertTrue(all(r["components"]["keep"]["parameter_norm"]>0 for r in doc["keep_probes"]))
        self.assertEqual(doc["keep_probe_identity"]["status"],"PROBE_ONLY")

    def test_T12_duplicate_sid_commits_once(self):
        a=UniqueSampleFairAssignment();a.begin_step(0,["s"]*10)
        rows=[self.choose(a) for _ in range(10)];self.assertTrue(all(r==rows[0] for r in rows))
        receipt=self.commit(a,0);self.assertEqual((receipt["batch_occurrences"],receipt["unique_samples"]),(10,1))
        self.assertEqual(sum(r["direct"] for r in a.history["s"]),1)

    def test_T13_duplicate_conflicts_and_optimizer_rollback(self):
        a=UniqueSampleFairAssignment();before=digest(a.state_dict());a.begin_step(0,["s","s"]);self.choose(a)
        with self.assertRaisesRegex(ValueError,"DUPLICATE_SAMPLE_CONFLICT"):self.choose(a,auth=digest({"drift":True}))
        with self.assertRaisesRegex(ValueError,"DUPLICATE_SAMPLE_CONFLICT"):self.choose(a,hard=torch.ones(6,dtype=torch.bool))
        changed=teacher([]);changed["inner"][0][1]=.01
        with self.assertRaisesRegex(ValueError,"DUPLICATE_SAMPLE_CONFLICT"):self.choose(a,teachers=[changed])
        a.rollback(0);self.assertEqual(digest(a.state_dict()),before)
        m=NS()  # Real optimizer checks use a model with the fixed two score rows.
        net=torch.nn.Module();net.trajectory_head=torch.nn.Linear(1,21);opt=torch.optim.AdamW(net.parameters(),weight_decay=0.)
        a.begin_step(0,["s"]);self.choose(a)
        for p in net.parameters():p.grad=torch.zeros_like(p)
        with patch.object(opt,"step",side_effect=RuntimeError("optimizer failure")):
            with self.assertRaises(RuntimeError):optimizer_step_v21(net,opt,a,0,head_rows(net))
        self.assertEqual(digest(a.state_dict()),before)
        a.begin_step(0,["s"]);self.choose(a);next(net.parameters()).grad.fill_(float("nan"))
        with self.assertRaises(ValueError):optimizer_step_v21(net,opt,a,0,head_rows(net))
        self.assertEqual(digest(a.state_dict()),before)

    def test_T14_checkpoint_resume_exact_model_and_exposure(self):
        net=torch.nn.Linear(2,1);initial=copy.deepcopy(net);opt=torch.optim.AdamW(net.parameters(),lr=2e-5,weight_decay=0.);a=UniqueSampleFairAssignment()
        def step(model,optimizer,scheduler,index):
            scheduler.begin_step(index,["s","s"]);r=[self.choose(scheduler) for _ in range(2)]
            optimizer.zero_grad();model(torch.ones(1,2)).square().sum().backward();optimizer.step();self.commit(scheduler,index);return r
        spec=dict(source_sha="fixed",seed=42,order_sha="fixed")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"checkpoint.pt";step(net,opt,a,0);save_v21_checkpoint(path,net,opt,a,spec)
            expected=step(net,opt,a,1);other=initial;oo=torch.optim.AdamW(other.parameters(),lr=2e-5,weight_decay=0.);b=UniqueSampleFairAssignment()
            load_v21_checkpoint(path,other,oo,b,spec);actual=step(other,oo,b,1)
            self.assertEqual(actual,expected);self.assertEqual(state_sha(net),state_sha(other));self.assertEqual(digest(a.state_dict()),digest(b.state_dict()))
            with self.assertRaises(ValueError):self.commit(b,1)
            with self.assertRaises(ValueError):b.begin_step(1,["s"])
            b.begin_step(2,["s"])
            with self.assertRaises(ValueError):save_v21_checkpoint(Path(folder)/"pending.pt",other,oo,b,spec)

    def test_T15_complete_gradient_matrix_score_and_reconstruction(self):
        m=RawModel();base,s,_=base_forward(m);obj=ObjectiveV21(m,state_sha(m),{k:.1 for k in KEYS_V21});z=obj.augment(base,[s]);r,v,total=audit_v21(m,z)
        self.assertTrue(set(KEYS_V21)<=set(r["components"]));self.assertEqual(r["loss_reconstruction_abs"],0.)
        self.assertLess(r["gradient_reconstruction_max_abs"],3e-5);self.assertEqual(float(torch.autograd.grad(z["loss"],z["raw"])[0][...,19:].norm()),0.)
        self.assertEqual(set(r["ratios"]),{"R_PG2","R_base","R_D","R_increment"});obj.assignment.rollback(0)

    def test_T16_eight_training_batch_identity_and_calibration(self):
        ev=evidence();a=read(ev/"GRADIENT_AUDIT_V2_1.json");self.assertFalse(a["uses_development"])
        self.assertEqual((a["calibration_occurrences"],a["calibration_unique_samples"]),(64,62));self.assertEqual(len(a["records"]),8)
        for row in a["training_inputs"]:
            for item in row:artifact(item["artifact"])
        for record in a["records"]:
            for b in range(8):
                rows=[r for r in record["geometry"] if r["batch"]==b];count=sum(r["geom_source"]!="NONE" for r in rows)
                self.assertTrue(all(r["task_query_denominator"]==count for r in rows))
                self.assertTrue(all(r["event_denominator"]>0 for r in rows if r["geom_source"]!="NONE"))
        for record,ratios in zip(a["records"],a["calibration"]["joint_ratios"]):
            predicted=ratios_from_gram_v21(record,a["calibration"]["weights"])
            for k,v in predicted.items():self.assertAlmostEqual(v,ratios[k],places=9)

    def test_T17_final_CPU0_guard_and_no_mutation(self):
        ev=evidence();cpu=read(ev/"CPU0_V2_1.json")
        self.assertEqual(cpu["threshold"],10);self.assertEqual(cpu["status"],"PASS")
        self.assertLessEqual(cpu["ratios"]["R_D"],10);self.assertLessEqual(cpu["ratios"]["R_base"],10)
        self.assertEqual(cpu["weights_before"],cpu["weights_after"]);self.assertEqual(cpu["optimizer_steps"],0)
        self.assertTrue(cpu["exposure_unchanged"]);self.assertFalse(cpu["code"]["allow_uncommitted_sources"])
        self.assertEqual(cpu["code"]["git_commit"],subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip())

    def test_T18_real_obstacle_isolated_shadow_step(self):
        ev=evidence();probe=read(ev/"SHADOW_OPTIMIZER_PROBE.json")
        self.assertEqual(probe["status"],"PASS");self.assertEqual(probe["optimizer_steps"],1)
        self.assertEqual(probe["formal_optimizer_steps"],0);self.assertTrue(probe["initial_model_unchanged"])
        self.assertTrue(any(g["improvement_m"]>1e-7 for g in probe["geometry_gaps"]))
        self.assertIn("initial_safe_drift",probe);self.assertIn("geom",probe["delta_cosine_negative"])

    def test_T19_no_corridor_or_online_GT_and_history_protection(self):
        m=RawModel();b,s,_=base_forward(m);obj=ObjectiveV21(m,state_sha(m),{k:.1 for k in KEYS_V21})
        with patch("query_minco.training.pg3_feasible_mode_v1.teacher_labels.build_corridor",side_effect=AssertionError("corridor built")):
            z=obj.augment(b,[s]);z["loss"].backward()
        obj.assignment.rollback(0)
        for folder in ("query_minco/models","query_minco/inference","query_minco/trajectory"):
            names=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,folder],cwd=str(ROOT),text=True).splitlines()
            for name in names:self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(["git","show",BASE+":"+name],cwd=str(ROOT)))
        name="experiments/pg3_feasible_mode_v1/fulltrain128.py"
        self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(["git","show",BASE+":"+name],cwd=str(ROOT)))
if __name__=="__main__":unittest.main()
