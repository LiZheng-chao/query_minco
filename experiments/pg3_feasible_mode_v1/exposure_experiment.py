"""One authorized CPU128 Mode-only experiment; exact legacy loss and evaluator."""
import os,time,uuid,signal,traceback,subprocess,resource
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.training.checkpoints import save_checkpoint,load_checkpoint
from query_minco.training.pg1.state_io import publish,sync_dir,presets,inspect_commit,atomic_json
from query_minco.training.pg3_feasible_mode_v1.exposure_assignment import ExposureRotation,METHOD_ID
from query_minco.training.pg3_feasible_mode_v1.objective import forward
from query_minco.training.pg3_feasible_mode_v1.spec import recipe,effective
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm
from query_minco.inference.identity import state_sha
from .common import model_from
from .fulltrain128 import load_items,lr_at,SCHEDULE,METHODS,R as HISTORY,seal
from .fulltrain128_evaluation import evaluate_set,inference_measure
from .exposure_replay import OUT,csv_write
BASE_METHOD=METHODS["B"]
RUN_ID="A42_PG3_ModeExposure_FullFrozen384_CPU128_run001"

def scripts():
    names=["exposure_experiment.py","exposure_replay.py","test_exposure_experiment.py",
           "fulltrain128.py","fulltrain128_evaluation.py","common.py"]
    return {str(Path(__file__).with_name(n).relative_to(ROOT)):sha(Path(__file__).with_name(n)) for n in names}

def committed(run,model,opt,spec,step,record,policy,parent=None):
    require(policy.last_committed_step==step-1,"EXPOSURE_SNAPSHOT_CURSOR")
    require(all(g["lr"]==lr_at(max(step-1,0)) for g in opt.param_groups),"EXPOSURE_SNAPSHOT_LR")
    folder=run/"commits";folder.mkdir(exist_ok=True)
    stage=folder/(".stage_%04d_"%step+uuid.uuid4().hex);stage.mkdir()
    receipt=save_checkpoint(stage,model,opt,spec,step,scheduler_id=SCHEDULE)
    write(stage/"ASSIGNMENT_STATE.json",policy.state_dict())
    write(stage/"UPDATE.json",record)
    write(stage/"COMMIT.json",dict(schema="pg3_exposure_commit_v1",step=step,parent_hash=parent,
        root_spec_sha256=digest(spec),files={p.name:sha(p) for p in stage.iterdir()},
        weights_sha256=receipt["weights_sha256"]))
    sync_dir(stage);target=folder/("step%04d"%step);publish(stage,target);return target

def restore(folder,model,opt,spec,policy,restore_rng=True):
    c=read(folder/"COMMIT.json");step=c["step"]
    require(c["schema"]=="pg3_exposure_commit_v1" and c["root_spec_sha256"]==digest(spec),"EXPOSURE_COMMIT_SPEC")
    require(set(c["files"])=={"checkpoint_step%04d.pt"%step,"checkpoint_step%04d.pt.json"%step,
                               "UPDATE.json","ASSIGNMENT_STATE.json"},"EXPOSURE_COMMIT_FILES")
    require(all("/" not in n and sha(folder/n)==h for n,h in c["files"].items()),"EXPOSURE_COMMIT_SHA")
    require("ASSIGNMENT_STATE.json" in read(folder/"COMMIT.json")["files"],"EXPOSURE_MISSING_HISTORY")
    blob=load_checkpoint(folder/("checkpoint_step%04d.pt"%step),model,opt,spec,restore_rng)
    policy.load_state_dict(read(folder/"ASSIGNMENT_STATE.json"))
    require(policy.last_committed_step==step-1,"EXPOSURE_RESTORE_CURSOR")
    require(blob["completed_optimizer_updates"]==step and blob["weights_sha256"]==c["weights_sha256"],
            "EXPOSURE_COMMIT_STATE")
    require(all(int(s["step"])==step for s in opt.state.values()),"EXPOSURE_OPTIMIZER_STEPS")
    return blob

def snapshot(run,folder,step):
    presets(run,folder,step)
    target=run/"checkpoints"
    os.link(str(folder/"ASSIGNMENT_STATE.json"),str(target/("assignment_state_step%04d.json"%step)))
    write(target/("BUNDLE_step%04d.json"%step),dict(commit=str(folder),commit_sha256=sha(folder/"COMMIT.json"),
        assignment_state_sha256=sha(folder/"ASSIGNMENT_STATE.json"),restore_entry="exposure_experiment.restore(commit_directory)"))

def prepare():
    gate=read(OUT/"REPLAY_RESULT.json");require(gate["gate"]["passed"],"EXPOSURE_REPLAY_NO_GAIN")
    cal=read(HISTORY/"preparation/CPU0.json")
    cal["bindings"]=gate["bindings"];cal["code"]=read(artifact(cal["bindings"]))["code"];cal["scripts"]=scripts()
    verify_code(cal["code"]);ds,geo,init,q,train,dev=load_items(cal)
    require(len(train)==384 and len(dev)==192 and sum(len(s.teacher_coeffs) for s,_,_ in train)==672,"EXPOSURE_ASSETS")
    oldspec=read(HISTORY/"runs/A42_B_FullFrozen384_CPU128_run001/RUN_SPEC.json")
    require(oldspec["method_id"]==BASE_METHOD and oldspec["new_weights"]==effective(cal["new_weights"],BASE_METHOD)
        and oldspec["old_weights"]==cal["old_weights"] and oldspec["initialization"]==init
        and oldspec["order128"]==cal["order128"] and oldspec["optimizer"]==dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.)
        and oldspec["schedule"]==SCHEDULE and oldspec["updates"]==128,"EXPOSURE_CONTROL_COMPARABILITY")
    configure(dict(seed=42,threads=1,device="cpu"));m=model_from(init);before=state_sha(m)
    # Entire shared initial-output identity, without repeating historical hard checks.
    baseline=HISTORY/"baseline"
    with np.load(baseline/"training384_step0000.npz",allow_pickle=False) as ar:
        lookup={str(s):i for i,s in enumerate(ar["ids"])};m.eval()
        from query_minco.trajectory import TrajectoryPostprocessor
        with torch.no_grad():
            for s,field,meta in train:
                raw=m(depth_input=s.depth_input[None],state=s.state[None]);tr=TrajectoryPostprocessor()(raw,s.state[None]);i=lookup[s.sample_id]
                require(np.array_equal(raw.numpy()[0],ar["raw"][i]) and np.array_equal(tr.trajectory.coeffs.numpy()[0],ar["coeffs"][i]),"EXPOSURE_STEP0_OUTPUT")
    m.train();batch=ds.load(geo.orders[42].batch(0)[0]);bank=TeacherBank(recipe());ctx=TeacherGuidance()
    rng=torch.get_rng_state()
    a=forward(m,batch,geo,BASE_METHOD,cal["old_weights"],cal["new_weights"],bank,ctx,0)
    ra,va,ga=audit(m,a,details=True)
    torch.set_rng_state(rng)
    pol=ExposureRotation();b=forward(m,batch,geo,BASE_METHOD,cal["old_weights"],cal["new_weights"],bank,ctx,0,assignment_policy=pol)
    rb,vb,gb=audit(m,b,details=True)
    require(torch.equal(a["loss"],b["loss"]) and torch.equal(ga,gb) and torch.equal(a["raw"],b["raw"]),"EXPOSURE_CPU0_PARITY")
    require(rb["physical_imitation_ratio"]<=10 and state_sha(m)==before==init["weights_sha256"],"EXPOSURE_CPU0_GUARD")
    cal.update(initialization=init,passed=True,optimizer_created=False,optimizer_steps=0,
        weights_before=before,weights_after=state_sha(m),cpu0=rb,step0_all384_bitwise=True,initial_gradient_bitwise=True)
    write(OUT/"CPU0.json",cal)
    write(OUT/"HISTORICAL_CONTROL_REUSE.json",dict(accepted=True,run=str(HISTORY/"runs/A42_B_FullFrozen384_CPU128_run001"),
        run_spec_sha256=sha(HISTORY/"runs/A42_B_FullFrozen384_CPU128_run001/RUN_SPEC.json"),
        seal=gate["historical_seals"]["B"],unchanged_legacy_package_files=True,all384_step0_bitwise=True,
        identical_optimizer_order_initialization_schedule_weights=True,only_variable="Query priority uses per-sample committed counts"))
    spec=dict(schema="pg3_mode_exposure_cpu128_v1",run_id=RUN_ID,method_id=METHOD_ID,base_loss_method=BASE_METHOD,
        code=cal["code"],experiment_sources=cal["scripts"],preparation=dict(path=str(OUT/"CPU0.json"),sha256=sha(OUT/"CPU0.json")),
        bindings=cal["bindings"],initialization=init,initialization_prior_updates=2880,
        device="cpu",precision="FP32",threads=1,seed=42,batch_size=8,updates=128,checkpoint_steps=[0,32,64,96,128],
        schedule=SCHEDULE,optimizer=oldspec["optimizer"],old_weights=cal["old_weights"],
        new_weights=effective(cal["new_weights"],BASE_METHOD),assignment=dict(name="PER_SAMPLE_SUCCESSFUL_UPDATE_COUNTS",
        capacity=2,priority=["ascending_extra_mode_count","cyclic_query_rank(q-sample_exposures)%6"],
        teacher_slot_solver="unchanged detached distance Hungarian",counter_commit="after successful optimizer update; unique sample/update"),
        order128=cal["order128"],train_index=cal["train_index"],development_index=cal["development_index"],
        conditional_control=False,ranking=False,head_projection=False,region=False,adaptive_risk=False,
        resources=oldspec["resources"],output_root=str(OUT/"runs"))
    write(OUT/"RUN_SPEC.json",spec)
    write(OUT/"AUTHORIZATION.json",dict(approved=True,authorized_by="LiZheng-chao",run_id=RUN_ID,
        spec_sha256=digest(spec),device="cpu",max_optimizer_updates=128,scope="Exposure-Aware Mode-only and necessary paired control; no extension"))
    write(OUT/"PREPARATION_SEAL.json",dict(files={p.name:sha(p) for p in [OUT/"CPU0.json",OUT/"RUN_SPEC.json",OUT/"AUTHORIZATION.json",
        OUT/"HISTORICAL_CONTROL_REUSE.json"]},status="READY_CPU128"))
    print("EXPOSURE_READY",digest(spec),rb["physical_imitation_ratio"],flush=True)

def train():
    spec=read(OUT/"RUN_SPEC.json");cal=read(artifact(spec["preparation"]));verify_code(spec["code"])
    require(scripts()==spec["experiment_sources"] and cal["passed"],"EXPOSURE_CODE_DRIFT")
    auth=read(OUT/"AUTHORIZATION.json")
    require(auth["approved"] is True and auth["spec_sha256"]==digest(spec) and auth["run_id"]==RUN_ID
            and auth["max_optimizer_updates"]==128 and spec["method_id"]==METHOD_ID,"EXPOSURE_AUTH")
    ds,geo,init,q,train_items,dev=load_items(cal);order=read(artifact(spec["order128"]))
    require(set(sum(order["batches"],[]))==set(ds.rows),"EXPOSURE_FROZEN384")
    run=Path(spec["output_root"])/RUN_ID;run.mkdir(parents=True,exist_ok=False)
    for n,obj in (("RUN_SPEC.json",spec),("AUTHORIZATION.json",auth),("QUALIFICATION.json",q)):write(run/n,obj)
    configure(dict(seed=42,threads=1,device="cpu"));model=model_from(init);require(state_sha(model)==init["weights_sha256"],"EXPOSURE_INIT")
    opt=torch.optim.AdamW(model.parameters(),lr=lr_at(0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
    pol=ExposureRotation();bank=TeacherBank(recipe());ctx=TeacherGuidance();score=head_rows(model);last=None;steps=0;streak=0;history=[];stop=False
    targets=read(HISTORY/"baseline/INITIAL_TARGETS.json");guardbase=read(HISTORY/"baseline/OPEN_DEV8.json")
    def request(*args):
        nonlocal stop
        stop=True
    oldsig=signal.signal(signal.SIGTERM,request);start=time.monotonic()
    def cosine(a,b):
        a=a.double();b=b.double();den=float(a.norm()*b.norm());return float(a@b)/den if den else None
    try:
        last=committed(run,model,opt,spec,0,dict(completed_optimizer_updates=0,shared_baseline_sha256=sha(HISTORY/"baseline/BASELINE.json")),pol)
        snapshot(run,last,0)
        for i in range(128):
            available=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
            require(not stop and time.monotonic()-start<1800 and available>=3072 and
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"EXPOSURE_RESOURCE")
            require(sum(p.stat().st_size for p in run.rglob("*") if p.is_file())<4*1024**3,"EXPOSURE_DISK")
            require(source_files()==spec["code"]["files"] and scripts()==spec["experiment_sources"] and
                    os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"EXPOSURE_SOURCE_DEVICE")
            ids,roles=geo.orders[42].batch(i);require(ids==order["batches"][i],"EXPOSURE_ORDER")
            samples=ds.load(ids);model.train();opt.zero_grad(set_to_none=True)
            z=forward(model,samples,geo,BASE_METHOD,cal["old_weights"],cal["new_weights"],bank,ctx,i,assignment_policy=pol)
            rec,vectors,total=audit(model,z,details=True);im=vectors["imitation"];ratio=rec["physical_imitation_ratio"]
            require(bool(torch.isfinite(total).all()) and np.isfinite(ratio),"EXPOSURE_NONFINITE")
            streak=streak+1 if ratio>10 else 0
            if streak>=2:
                write(run/"REJECTED_UPDATE.json",dict(attempt=i+1,ratio=ratio,ids=ids,optimizer_step_executed=False))
                raise ValueError("TWO_STEP_GRADIENT_RATIO_GT10")
            before=torch.cat([p.detach().flatten().clone() for p in model.parameters()]);offset=0
            for name,p in model.named_parameters():
                p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
                if name.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"EXPOSURE_SCORE_GRAD")
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));require(np.isfinite(norm),"EXPOSURE_CLIP")
            for group in opt.param_groups:group["lr"]=lr_at(i)
            rec.update(index=i,completed_optimizer_updates=i+1,ids=ids,roles=roles,loss=float(z["loss"].detach()),
                losses={k:float(v.detach()) for k,v in z["terms"].items()},lr=lr_at(i),physical_imitation_ratio=ratio,
                ratio_alarm_streak=streak,gradnorm=norm,assignments=z["assignments"],matched=z["matched"].tolist(),
                hard_pass=z["labels"]["passed"].tolist(),goal_pass=z["goal_labels"].tolist(),M=[len(s.teacher_coeffs) for s in samples],
                regime=[ds.rows[s.sample_id]["regime"] for s in samples],route_records=z["records"])
            arrays=dict(raw=z["raw"].detach().numpy(),inner=z["output"].parameters.inner_pts.detach().numpy(),
                coeffs=z["output"].trajectory.coeffs.detach().numpy(),durations=z["output"].parameters.durations.detach().numpy(),
                matched=z["matched"].numpy(),mode=z["per_query"]["mode"].detach().numpy())
            intents=run/"intents";intents.mkdir(exist_ok=True)
            with (intents/("step%04d_ARRAYS.npz"%(i+1))).open("xb") as f:np.savez_compressed(f,**arrays)
            atomic_json(intents/("step%04d_STARTED.json"%(i+1)),dict(parent=sha(last/"COMMIT.json"),
                arrays_sha256=sha(intents/("step%04d_ARRAYS.npz"%(i+1)))))
            opt.step();steps=i+1
            require(rows_unchanged(model,score) and all(bool(torch.isfinite(p).all()) for p in model.parameters()),"EXPOSURE_PARAMETER_CONTRACT")
            pol.commit(i)
            delta=torch.cat([p.detach().flatten() for p in model.parameters()])-before
            rec.update(adam_delta_vs_negative_imitation=cosine(delta,-im),adam_delta_vs_negative_total=cosine(delta,-total),
                adam_delta_vs_negative_mode=cosine(delta,-vectors["mode"]),imitation_mode_cosine=cosine(im,vectors["mode"]))
            alarm=None
            if steps in spec["checkpoint_steps"]:
                rec["open_dev8"]=open8(model,read(artifact(spec["bindings"]))["development"])
                try:metric_alarm(history,rec["open_dev8"],guardbase,{})
                except ValueError as e:alarm=str(e)
                rec["checkpoint_guard_failure"]=alarm
            last=committed(run,model,opt,spec,steps,rec,pol,parent=sha(last/"COMMIT.json"))
            if steps in spec["checkpoint_steps"]:
                snapshot(run,last,steps)
                for name,items in (("training384",train_items),("development192",dev)):
                    evaluate_set(model,items,run/"evaluation",name,steps,targets)
                write(run/"evaluation"/("INFERENCE_step%04d.json"%steps),inference_measure(model,train_items[0][0]))
            print("EXPOSURE_UPDATE",steps,ratio,flush=True)
            if alarm:raise ValueError(alarm)
        restored=ExposureRotation();blob=restore(last,model,opt,spec,restored,False)
        require(blob["scheduler"]==SCHEDULE and restored.state_dict()==pol.state_dict() and
                all(int(s["step"])==128 for s in opt.state.values()),"EXPOSURE_FINAL_STATE")
        write(run/"RESULT.json",dict(status="COMPLETED",optimizer_updates=128,
            final_weights_sha256=state_sha(model),unique_tasks=len(pol.state),unique_sample_updates=sum(h["exposures"] for h in pol.state.values()),
            wall_s=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    except BaseException:
        pol.abort()
        write(run/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),actual_optimizer_updates=steps,last_committed=str(last),retry=False))
        raise
    finally:
        signal.signal(signal.SIGTERM,oldsig)
        seal(run,status="COMPLETED" if (run/"RESULT.json").exists() else "FAIL",actual_optimizer_updates=steps)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","train"]);args=p.parse_args()
    prepare() if args.command=="prepare" else train()
