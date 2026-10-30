"""Executor rejection, frozen baseline, protocol and exact checkpoint tests."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import torch
from query_minco.training.io import read,write,sha,digest
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_v2_no_corridor.executor_v21 import *
from query_minco.training.pg3_v2_no_corridor.audit import step_guard_v21
from query_minco.training.pg2_guided_v2.spec import COMPONENTS,FIXED

def toy():
    m=torch.nn.Module();m.trajectory_head=torch.nn.Linear(1,21);return m

def spec(m,arm="A"):
    return dict(arm=arm,seed=42,threads=1,batch_size=8,updates=128,device="cpu",precision="FP32",schedule=SCHEDULE,
        optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),execution=dict(limit=4,checkpoint_every=1,purpose="PG3_V21_SHORT4",wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4),
        stop_mode="legacy" if arm=="A" else "replace_full_stop",initialization=dict(weights_sha256=state_sha(m)),weights={} if arm=="A" else {k:.01 for k in KEYS_V21},
        pg2_weights=dict({k:1. for k in COMPONENTS},**FIXED),method_id="query_minco_pg2_guided_v2_safe_v1")
CFG=dict(increment_limit=2.,keep_active_max_ratio=.05)
def record():
    return dict(components={k:dict(parameter_norm=1.) for k in list(KEYS_V21)+["pg2/imitation"]},weights={k:.01 for k in KEYS_V21},ratios=dict(R_base=.5,R_D=1.,R_PG2=.5,R_increment=1.))
def advance(e):
    e.opt.zero_grad();e.model.trajectory_head(torch.ones(1,1))[...,:19].square().sum().backward()
    for g in e.opt.param_groups:g["lr"]=lr_at(e.completed)
    e.opt.step();e.exposure.last_step=e.completed;e.completed+=1

class ExecutorTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
    def test_increment_above_two_actually_rejects(self):
        r=record();r["ratios"]["R_increment"]=2.000001
        with self.assertRaisesRegex(ValueError,"INCREMENT_BUDGET"):step_guard_v21(r,CFG)
    def test_increment_at_two_passes(self):
        r=record();r["ratios"]["R_increment"]=2.
        self.assertTrue(step_guard_v21(r,CFG)["increment_budget_passed"])
    def reject_engine(self,nan=False):
        m=toy();e=Engine(m,spec(m,"D"),CFG);initial=state_sha(m);history=digest(e.exposure.state_dict())
        e.objective=Mock(return_value={});e.exposure.begin_step(0,["s"])
        r=record();r["ratios"]["R_increment"]=1. if nan else 3.
        total=torch.full((sum(p.numel() for p in m.parameters()),),float("nan") if nan else 0.)
        with patch("query_minco.training.pg3_v2_no_corridor.executor_v21.audit_v21",return_value=(r,{},total)),patch.object(e.opt,"step") as optimizer:
            with self.assertRaises(ValueError):e.step([] ,None)
            optimizer.assert_not_called()
        self.assertEqual(state_sha(m),initial);self.assertEqual(digest(e.exposure.state_dict()),history);self.assertEqual(e.completed,0);self.assertEqual(len(e.opt.state),0)
    def test_overbudget_no_optimizer_or_exposure_commit(self):self.reject_engine()
    def test_nonfinite_gradient_no_optimizer_or_exposure_commit(self):self.reject_engine(True)
    def test_nonfinite_ratio_rejected(self):
        r=record();r["ratios"]["R_increment"]=float("nan")
        with self.assertRaises(ValueError):step_guard_v21(r,CFG)
    def test_keep_active_budget_still_rejects(self):
        r=record();r["components"]["keep"]["parameter_norm"]=100.
        with self.assertRaisesRegex(ValueError,"KEEP_ACTIVE"):step_guard_v21(r,CFG)
    def test_main_ratio_two_step_guard_unchanged(self):
        r=record();r["ratios"]["R_D"]=11.
        self.assertEqual(step_guard_v21(r,CFG)["ratio_alarm_streak"],1)
        with self.assertRaisesRegex(ValueError,"TWO_STEP"):step_guard_v21(r,CFG,streak=1)
    def test_pg2_wrapper_preserves_loss_tensor_and_gradient(self):
        raw=torch.randn(1,6,21,requires_grad=True);loss=raw[...,:19].square().mean()
        z=dict(loss=loss,raw=raw,output=None,matched=None,labels=None,goal_labels=None);w=wrap_pg2(z)
        self.assertIs(w["loss"],loss);self.assertTrue(torch.equal(torch.autograd.grad(w["loss"],raw,retain_graph=True)[0],torch.autograd.grad(loss,raw)[0]))
    def test_schedule_matches_original_128(self):
        from experiments.pg3_feasible_mode_v1.fulltrain128 import lr_at as legacy
        self.assertEqual([lr_at(i) for i in range(128)],[legacy(i) for i in range(128)])
        with self.assertRaises(ValueError):lr_at(128)
    def test_protocol_drift_rejected(self):
        m=toy();s=spec(m)
        for key,value in (("seed",43),("batch_size",4),("updates",4),("precision","FP16")):
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):validate_protocol(bad)
    def test_frozen_runtime_weights_and_inputs_rejected(self):
        m=toy();s=spec(m);s.update(bindings={"id":"frozen"},order={"id":"frozen"},configuration={"id":"frozen"})
        validate_frozen_spec(s,copy.deepcopy(s))
        for key in ("weights","pg2_weights","order","initialization"):
            drift=copy.deepcopy(s);drift[key]={"modified":True}
            with self.assertRaisesRegex(ValueError,"FROZEN_SPEC_DRIFT"):validate_frozen_spec(drift,s)
    def test_authorization_exact_spec_scope_limit(self):
        m=toy();s=spec(m);a=dict(approved=True,spec_sha256=digest(s),max_updates=4,scope="PG3_V21_SHORT4",authorized_by="test")
        authorize_spec(s,a)
        for k,v in (("spec_sha256","x"),("scope","other"),("max_updates",128)):
            with self.assertRaises(ValueError):authorize_spec(s,dict(a,**{k:v}))
    def make_checkpoint(self,folder):
        m=toy();e=Engine(m,spec(m),CFG);zero=publish_checkpoint(Path(folder),e,dict(step=0));advance(e);one=publish_checkpoint(Path(folder),e,dict(step=1),sha(zero/"COMMIT.json"));return e,one
    def test_checkpoint_exact_model_optimizer_exposure_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            e,_=self.make_checkpoint(folder);fresh=copy.deepcopy(e.reference.model).train().requires_grad_(True)
            other=Engine(fresh,e.spec,CFG);restore_checkpoint(folder,other)
            self.assertEqual(state_sha(e.model),state_sha(other.model));self.assertEqual(e.completed,other.completed);self.assertEqual(e.exposure.state_dict(),other.exposure.state_dict())
            for left,right in zip(e.opt.state.values(),other.opt.state.values()):
                for k in left:self.assertTrue(torch.equal(left[k],right[k]))
            advance(e);advance(other);self.assertEqual(state_sha(e.model),state_sha(other.model))
    def test_checkpoint_corruption_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            e,p=self.make_checkpoint(folder)
            with (p/"checkpoint.pt").open("ab") as f:f.write(b"corruption")
            with self.assertRaisesRegex(ValueError,"CHECKPOINT_SHA"):inspect_chain(folder,e.spec)
    def test_monitor_cursor_mismatch_stops(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            e,p=self.make_checkpoint(folder);mon=read(p/"MONITOR.json");mon["completed"]=99;(p/"MONITOR.json").write_text(json.dumps(mon))
            c=read(p/"COMMIT.json");c["files"]["MONITOR.json"]=sha(p/"MONITOR.json");(p/"COMMIT.json").write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,"MONITOR_CURSOR"):inspect_chain(folder,e.spec)
    def test_duplicate_commit_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            e,_=self.make_checkpoint(folder)
            with self.assertRaisesRegex(ValueError,"DUPLICATE_COMMIT"):publish_checkpoint(folder,e,{})
    def test_pending_checkpoint_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            m=toy();e=Engine(m,spec(m,"D"),CFG);e.exposure.begin_step(0,["s"])
            with self.assertRaises(ValueError):publish_checkpoint(folder,e,{})
            self.assertFalse((Path(folder)/"commits/step0000").exists())
    def test_optimizer_cursor_mismatch_rejected(self):
        m=toy();e=Engine(m,spec(m),CFG);advance(e)
        next(iter(e.opt.state.values()))["step"].fill_(2.)
        with self.assertRaisesRegex(ValueError,"OPTIMIZER_CURSOR"):e.validate_state()
if __name__=="__main__":unittest.main()
