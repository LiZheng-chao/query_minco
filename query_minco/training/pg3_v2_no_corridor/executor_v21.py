"""Independent V2.1 transactional execution; no loss/weight changes."""
import os,uuid,random
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,require
from query_minco.inference.identity import state_sha
from query_minco.training.initialization import head_rows,rows_unchanged
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from .objective import ObjectiveV21,KEYS_V21
from .revision import UniqueSampleFairAssignment,FrozenSafeReferenceV21,optimizer_step_v21,save_v21_checkpoint,load_v21_checkpoint
from .audit import audit_v21,step_guard_v21
from query_minco.training.pg1_v2_r12.diagnostics import cosine
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
SCHEDULE="fulltrain128_cosine_2e-5_1e-6_endpoint_v1"
def lr_at(i):
    import math
    require(type(i) is int and 0<=i<128,"EXEC_LR_INDEX")
    return float(1e-6+.5*(2e-5-1e-6)*(1+math.cos(math.pi*i/127)))
def validate_protocol(s):
    require(s["arm"] in ("A","D") and s["seed"]==42 and s["threads"]==1 and s["batch_size"]==8 and s["updates"]==128
        and s["device"]=="cpu" and s["precision"]=="FP32" and s["schedule"]==SCHEDULE,"EXEC_PROTOCOL")
    require(s["optimizer"]==dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),"EXEC_OPTIMIZER")
    e=s["execution"]
    require(e["limit"] in (4,128) and e["checkpoint_every"]==1 and e["purpose"]==("PG3_V21_SHORT4" if e["limit"]==4 else "PG3_V21_PAIRED128"),"EXEC_LIMIT")
    require({k:e[k] for k in ("wall_s","rss_mib","min_available_mib","disk_gib")}==dict(wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4),"EXEC_RESOURCE_PROTOCOL")
    require(s["stop_mode"]==("legacy" if s["arm"]=="A" else "replace_full_stop"),"EXEC_STOP_MODE")
def validate_frozen_spec(s,reference):
    for key in ("arm","method_id","initialization","bindings","order","configuration","pg2_weights","weights","stop_mode","seed","threads","batch_size","updates","optimizer","schedule","precision","device"):
        require(s[key]==reference[key],"EXEC_FROZEN_SPEC_DRIFT:"+key)

def authorize_spec(s,a):
    validate_protocol(s)
    require(a.get("approved") is True and a.get("spec_sha256")==digest(s) and a.get("max_updates")==s["execution"]["limit"]
        and a.get("scope")==s["execution"]["purpose"] and bool(a.get("authorized_by")),"EXEC_AUTHORIZATION")
def wrap_pg2(z):
    zero=z["raw"].sum()*0
    return dict(pg2=z,base_loss=z["loss"],loss=z["loss"],raw=z["raw"],output=z["output"],matched=z["matched"],labels=z["labels"],goal_labels=z["goal_labels"],
        terms={k:zero for k in KEYS_V21},weights={k:0. for k in KEYS_V21},stop_mode="legacy",branch_terms={k:zero for k in ("SECTION","BNECK_MATCHED","BNECK_FALLBACK")},geometry=[],diagnostics={},assignments=[])
class Engine:
    def __init__(self,model,spec,config):
        validate_protocol(spec);self.spec=spec;self.config=config;self.model=model.train()
        require(state_sha(model)==spec["initialization"]["weights_sha256"],"EXEC_INIT")
        self.score=head_rows(model);self.objective=ObjectiveV21(model,state_sha(model),spec["weights"]) if spec["arm"]=="D" else None
        self.exposure=self.objective.assignment if self.objective else UniqueSampleFairAssignment()
        self.reference=self.objective.reference if self.objective else FrozenSafeReferenceV21(model,expected_sha=state_sha(model))
        self.opt=torch.optim.AdamW(model.parameters(),lr=lr_at(0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.context=TeacherGuidance();self.completed=0;self.monitor=dict(streak=0,keep_active=False);self.initial_matches={};self.last_audit=None;self.phase="INITIAL"
    def initial_labels(self,samples,fields):
        safe=[];matched=[]
        for s,f in zip(samples,fields):
            safe.append(self.reference._reference(s,f)["safe"])
            if s.sample_id not in self.initial_matches:
                with torch.no_grad():
                    r=self.reference.model(depth_input=s.depth_input[None],state=s.state[None]);o=TrajectoryPostprocessor()(r,s.state[None])
                    cost,_=cost_matrix(o,s,0);_,qq,_=match(cost,"B_set");mask=torch.zeros(6,dtype=torch.bool);mask[qq]=True;self.initial_matches[s.sample_id]=mask
            matched.append(self.initial_matches[s.sample_id])
        return torch.stack(safe),torch.stack(matched)
    def step(self,samples,geo):
        i=self.completed;require(i<self.spec["execution"]["limit"],"EXEC_UPDATE_LIMIT");self.phase="FORWARD";self.opt.zero_grad(set_to_none=True);self.model.train()
        try:
            z=self.objective(self.model,samples,geo,self.spec["pg2_weights"],self.context,i) if self.objective else wrap_pg2(pg2_forward(self.model,samples,geo,self.spec["method_id"],self.context,self.spec["pg2_weights"]))
            rec,vectors,total=audit_v21(self.model,z,False);self.last_audit=rec;self.phase="PRESTEP_GUARD"
            if self.objective:
                guard=step_guard_v21(rec,self.config,streak=self.monitor["streak"],keep_previously_active=self.monitor["keep_active"])
                monitor=dict(streak=guard["ratio_alarm_streak"],keep_active=guard["keep_ever_active"])
            else:
                require(all(np.isfinite(v) for v in rec["ratios"].values()),"EXEC_A_NONFINITE_RATIO")
                streak=self.monitor["streak"]+1 if rec["ratios"]["R_PG2"]>10 else 0;require(streak<2,"EXEC_A_TWO_STEP_RATIO_GT10")
                guard=dict(ratio_alarm_streak=streak,increment_budget_passed=True);monitor=dict(streak=streak,keep_active=False)
            require(bool(torch.isfinite(total).all()),"EXEC_NONFINITE_GRADIENT")
            before=torch.cat([p.detach().flatten().clone() for p in self.model.parameters()]);offset=0
            for n,p in self.model.named_parameters():
                p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
                if n.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"EXEC_SCORE_GRAD")
            require(rows_unchanged(self.model,self.score),"EXEC_PRE_SCORE_DRIFT")
            norm=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.));require(np.isfinite(norm),"EXEC_CLIP_FINITE")
            for g in self.opt.param_groups:g["lr"]=lr_at(i)
            initial_safe,initial_matched=self.initial_labels(samples,z["pg2"]["fields"])
            hb=z["labels"]["passed"].detach().clone();gb=z["goal_labels"].detach().clone();self.phase="OPTIMIZER"
            if self.objective:receipt=optimizer_step_v21(self.model,self.opt,self.exposure,i,self.score)
            else:
                self.opt.step();require(rows_unchanged(self.model,self.score) and all(bool(torch.isfinite(p).all()) for p in self.model.parameters()),"EXEC_A_PARAMETERS")
                self.exposure.last_step=i;receipt=dict(step=i,direct_supervision=False)
            self.completed=i+1;self.monitor=dict(self.monitor,**monitor);self.phase="POSTSTEP_MONITOR"
            after=pg2_forward(self.model,samples,geo,"query_minco_pg2_guided_v2_safe_v1",self.context,self.spec["pg2_weights"])
            ha=after["labels"]["passed"].detach();ga=after["goal_labels"].detach();delta=torch.cat([p.detach().flatten() for p in self.model.parameters()])-before
            changes=[]
            for b,s in enumerate(samples):
                for q in range(6):
                    if bool(hb[b,q])!=bool(ha[b,q]) or (s.sample_id=="f00_narrow3d_g5_v1" and q==3):changes.append(dict(sample_id=s.sample_id,query=q,before=bool(hb[b,q]),after=bool(ha[b,q]),initial_safe=bool(initial_safe[b,q]),initial_matched=bool(initial_matched[b,q]),reasons=after["labels"]["checks"][b][q]["reasons"]))
            counts={k:sum(r["geom_source"]==k for r in z["geometry"]) for k in ("SECTION","BNECK_MATCHED","BNECK_FALLBACK","NONE")};selected=sum(len(a["selected"]) for a in z["assignments"])
            with torch.no_grad():_,drift=self.reference.loss(after,samples,after["fields"])
            metrics=dict(physics_before=int(hb.sum()),physics_after=int(ha.sum()),goal_before=int(gb.sum()),goal_after=int(ga.sum()),oracle_before=int((hb&gb).any(-1).sum()),oracle_after=int((ha&ga).any(-1).sum()),
                pass_to_fail=int((hb&~ha).sum()),fail_to_pass=int((~hb&ha).sum()),initial_safe_degraded=int((initial_safe&~ha).sum()),initial_matched_safe_degraded=int((initial_safe&initial_matched&~ha).sum()),
                geom_source_counts=counts,geom_denominator=len(samples)*6,geom_source_fractions={k:v/(len(samples)*6) for k,v in counts.items()},selected_queries=selected,section_selected_coverage=None if not selected else counts["SECTION"]/selected,keep_drift=drift,changes=changes)
            self.phase="ACCEPTED_NOT_PUBLISHED"
            return dict(index=i,completed=self.completed,ids=[s.sample_id for s in samples],lr=lr_at(i),loss=float(z["loss"].detach()),audit=rec,guard=guard,metrics=metrics,
                gradient_norm_before_clip=norm,optimizer_delta_norm=float(delta.norm()),delta_cosine_negative={k:cosine(delta.double(),v.double().neg()) for k,v in dict(total=total,imitation=vectors["pg2/imitation"],PG2=vectors["PG2_total"],**{k:vectors[k] for k in KEYS_V21}).items()},
                exposure_receipt=receipt,model_sha256=state_sha(self.model),exposure_sha256=digest(self.exposure.state_dict()))
        except BaseException:self.exposure.rollback(i);raise
    def validate_state(self):
        require(self.exposure.last_step==self.completed-1,"EXEC_EXPOSURE_CURSOR")
        require(rows_unchanged(self.model,self.score) and all(bool(torch.isfinite(p).all()) for p in self.model.parameters()),"EXEC_MODEL_STATE")
        counts=[]
        for state in self.opt.state.values():
            require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values()),"EXEC_OPTIMIZER_NONFINITE");counts.append(int(state["step"]))
        require((not counts and self.completed==0) or (len(counts)==len(list(self.model.parameters())) and set(counts)=={self.completed}),"EXEC_OPTIMIZER_CURSOR")
        require(all(g["lr"]==lr_at(max(self.completed-1,0)) and g["weight_decay"]==0. and g["betas"]==(.9,.999) and g["eps"]==1e-8 for g in self.opt.param_groups),"EXEC_OPTIMIZER_PROTOCOL_STATE")


def publish_checkpoint(run,engine,record,parent=None):
    engine.validate_state();directory=Path(run)/"commits";directory.mkdir(exist_ok=True)
    target=directory/("step%04d"%engine.completed);require(not target.exists(),"EXEC_DUPLICATE_COMMIT")
    require(not list(directory.glob(".stage*")),"EXEC_UNRECOVERED_STAGE")
    stage=directory/(".stage_"+uuid.uuid4().hex);stage.mkdir()
    save_v21_checkpoint(stage/"checkpoint.pt",engine.model,engine.opt,engine.exposure,engine.spec)
    n=np.random.get_state()
    write(stage/"MONITOR.json",dict(completed=engine.completed,monitor=engine.monitor,numpy_rng=[n[0],n[1].tolist(),n[2],n[3],n[4]],python_rng=random.getstate()))
    write(stage/"UPDATE.json",record)
    write(stage/"COMMIT.json",dict(step=engine.completed,parent_sha256=parent,spec_sha256=digest(engine.spec),
        files={p.name:sha(p) for p in stage.iterdir()},model_sha256=state_sha(engine.model)))
    fd=os.open(str(stage),os.O_RDONLY);os.fsync(fd);os.close(fd);stage.rename(target)
    fd=os.open(str(directory),os.O_RDONLY);os.fsync(fd);os.close(fd);engine.phase="COMMITTED";return target

def inspect_chain(run,spec):
    directory=Path(run)/"commits";require(not list(directory.glob(".stage*")),"EXEC_UNRECOVERED_STAGE")
    commits=sorted(directory.glob("step*"));require(bool(commits),"EXEC_NO_CHECKPOINT");parent=None
    for i,path in enumerate(commits):
        rec=read(path/"COMMIT.json")
        require(path.name=="step%04d"%i and rec["step"]==i and rec["parent_sha256"]==parent and rec["spec_sha256"]==digest(spec),"EXEC_CHECKPOINT_CHAIN")
        require(set(rec["files"])=={"checkpoint.pt","MONITOR.json","UPDATE.json"},"EXEC_CHECKPOINT_MEMBERS")
        for name,value in rec["files"].items():require(sha(path/name)==value,"EXEC_CHECKPOINT_SHA")
        require(read(path/"MONITOR.json")["completed"]==i,"EXEC_MONITOR_CURSOR");parent=sha(path/"COMMIT.json")
    return commits[-1]

def restore_checkpoint(run,engine):
    latest=inspect_chain(run,engine.spec);m=read(latest/"MONITOR.json")
    blob=load_v21_checkpoint(latest/"checkpoint.pt",engine.model,engine.opt,engine.exposure,engine.spec)
    engine.completed=m["completed"];engine.monitor=m["monitor"]
    require(read(latest/"COMMIT.json")["model_sha256"]==blob["model_sha256"],"EXEC_CHECKPOINT_MODEL_SHA")
    n=m["numpy_rng"];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
    random.setstate(tuples(m["python_rng"]));engine.validate_state();return latest
