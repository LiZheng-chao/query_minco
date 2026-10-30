"""Independent CPU0/calibration and explicitly authorized paired A/D execution."""
import argparse, copy, csv, json, os, resource, subprocess, time, traceback
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v2.objective import forward as safe_forward
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg3_v2_no_corridor.objective import Objective,KEYS,METHOD_ID,PILOT
from query_minco.training.pg3_v2_no_corridor.audit import audit,calibrate
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_feasible_mode_v1.fulltrain128 import lr_at,load_items
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import evaluate_set,inference_measure
OUT=ROOT/"research_outputs/pg3_v2_no_corridor"
OLD=ROOT/"research_outputs/pg3_feasible_mode_v1/fulltrain128_001/preparation/CPU0.json"
CONFIG=Path(__file__).with_name("pilot.json")
PREP=OUT/"_preparation"/"cpu0_002"

def sources():
    reused=[ROOT/"experiments/pg3_feasible_mode_v1"/name for name in ("common.py","fulltrain128.py","fulltrain128_evaluation.py")]
    local=[p for p in sorted(Path(__file__).parent.glob("*")) if p.is_file()]
    return dict(package=source_files(),experiment={str(p.relative_to(ROOT)):sha(p) for p in local+reused})

def prepare():
    folder=PREP;folder.mkdir(parents=True,exist_ok=False)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","V2_CPU_ONLY")
    configure(dict(seed=42,threads=1,device="cpu"))
    old=read(OLD);binding=read(artifact(old["bindings"]))
    # A NEW binding carries the current identity. Historical bindings are immutable.
    binding["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=True)
    write(folder/"BINDINGS.json",binding);ref=dict(path=str(folder/"BINDINGS.json"),sha256=sha(folder/"BINDINGS.json"))
    ds,geo,init,q=qualify_bindings(ref,42);write(folder/"QUALIFICATION.json",q)
    m=model_from(init);m.train();before=state_sha(m);require(before==init["weights_sha256"],"V2_INIT")
    weights=dict(mode=1.,bottleneck=1.,diversity=1.,**PILOT)
    objective=Objective(m,before,weights);records=[];cfg=read(CONFIG)
    require(all(cfg[k]==v for k,v in PILOT.items()) and cfg["updates"]==128 and cfg["batch_size"]==8 and cfg["seed"]==42 and cfg["threads"]==1,"V2_CONFIG_CONTRACT")
    for i in range(cfg["calibration_batches"]):
        ids,_=geo.orders[42].batch(i); samples=ds.load(ids)
        z=objective(m,samples,geo,old["old_weights"],TeacherGuidance(),0)
        row,_,_=audit(m,z);row["ids"]=ids;records.append(row);write(folder/("gradient_batch%02d.json"%i),row);objective.assignment.discard()
        print("TRAIN_GRADIENT_AUDIT",i,flush=True)
    calibration=calibrate(records);objective.weights=calibration["weights"]
    ids,_=geo.orders[42].batch(0); samples=ds.load(ids)
    z=objective(m,samples,geo,old["old_weights"],TeacherGuidance(),0);cpu,_,_=audit(m,z);objective.assignment.discard()
    require(state_sha(m)==before and state_sha(objective.reference.model)==before,"V2_CPU0_MUTATED")
    passed=calibration["gradient_guard_passed"] and cpu["physical_imitation_ratio"] is not None and cpu["physical_imitation_ratio"]<=10
    write(folder/"GRADIENT_AUDIT.json",dict(records=records,calibration=calibration,cpu0=cpu,uses_development=False,optimizer_steps=0))
    cal=dict(old,bindings=ref,code=binding["code"],new_weights=calibration["weights"],sources=sources(),configuration=dict(path=str(CONFIG),sha256=sha(CONFIG)),
             historical_preparation=dict(path=str(OLD),sha256=sha(OLD)),calibration=calibration,cpu0=cpu,weights_before=before,weights_after=state_sha(m),optimizer_steps=0,training_status="NOT_RUN",passed=passed)
    write(folder/"CPU0.json",cal)
    for arm in ("A","D"):write(folder/("RUN_SPEC_"+arm+".json"),make_spec(arm,cal))
    print("CPU0_RESULT",dict(passed=passed,ratio=cpu["physical_imitation_ratio"],calibration=calibration,training="NOT_RUN"),flush=True)
    require(passed,"V2_CPU0_GRAD_GUARD")

def make_spec(arm,cal):
    require(arm in ("A","D"),"V2_ARM")
    return dict(schema="pg3_v2_paired_cpu128_v1",run_id="A42_"+arm+"_PG3V2_CPU128_run001",arm=arm,
        method_id="query_minco_pg2_guided_v2_safe_v1" if arm=="A" else METHOD_ID,
        initialization=cal["initialization"],bindings=cal["bindings"],sources=cal["sources"],
        order128=cal["order128"],train_index=cal["train_index"],development_index=cal["development_index"],
        old_weights=cal["old_weights"],new_weights={} if arm=="A" else cal["new_weights"],
        seed=42,threads=1,device="cpu",precision="FP32",batch_size=8,updates=128,
        optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
        schedule="fulltrain128_cosine_2e-5_1e-6_endpoint_v1",ranking=False,control=False,projection=False,
        convex_corridor=False,teacher_corridor=False,calibration_uses_development=False,
        checkpoint_steps=[0,32,64,96,128],resources=dict(wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4),output_root=str(OUT/"runs"),training_status="NOT_RUN")

def train(arm,authorization):
    cal=read(PREP/"CPU0.json");require(cal.get("passed") is True,"V2_CPU0_NOT_READY");verify_code(cal["code"]);spec=make_spec(arm,cal)
    auth=read(authorization)
    require(auth.get("approved") is True and auth.get("run_spec_sha256")==digest(spec)
        and auth.get("max_updates")==128 and auth.get("scope")=="PG3_V2_SINGLE_CPU128_RUN"
        and bool(auth.get("authorized_by")),"V2_EXPLICIT_RUN_AUTHORIZATION")
    require(sources()==spec["sources"],"V2_SOURCE_DRIFT")
    # Commit the new implementation first; do not stage unrelated exposure work.
    tracked=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True)
    require(not tracked,"V2_TRAIN_REQUIRES_COMMITTED_CHANGES")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","V2_CPU_ONLY")
    configure(dict(seed=42,threads=1,device="cpu"))
    ds,geo,init,q,items,dev=load_items(cal);m=model_from(init);m.train();before=state_sha(m)
    require(before==spec["initialization"]["weights_sha256"],"V2_INITIAL_STATE")
    objective=Objective(m,before,cal["new_weights"]) if arm=="D" else None
    run=OUT/"runs"/spec["run_id"];run.mkdir(parents=True,exist_ok=False)
    write(run/"RUN_SPEC.json",spec);write(run/"AUTHORIZATION.json",auth)
    opt=torch.optim.AdamW(m.parameters(),lr=lr_at(0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
    score=head_rows(m);order=read(artifact(cal["order128"]));streak=0;completed=0;start=time.monotonic()
    def checkpoint(step):
        p=run/"checkpoints";p.mkdir(exist_ok=True)
        # Atomic exclusive publication of optimizer+fair state in one checkpoint.
        temp=p/(".step%04d.pt"%step);target=p/("step%04d.pt"%step)
        require(not temp.exists() and not target.exists(),"V2_CHECKPOINT_EXISTS")
        torch.save(dict(state_dict=m.state_dict(),optimizer=opt.state_dict(),step=step,
            fair=None if objective is None else objective.assignment.state_dict(),run_spec_sha256=digest(spec)),temp)
        temp.rename(target)
    try:
        initial={};history=[]
        guardbase=open8(m,read(artifact(cal["bindings"]))["development"])
        write(run/"INITIAL_OPEN_DEV8.json",guardbase)
        for name,data in (("training384",items),("development192",dev)):
            _,targets=evaluate_set(m,data,run/"evaluation",name,0);initial.update(targets)
        checkpoint(0)
        for i in range(128):
            avail=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
            require(time.monotonic()-start<1800 and avail>=3072 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"V2_RESOURCE")
            require(sum(p.stat().st_size for p in run.rglob("*") if p.is_file())<4*1024**3,"V2_DISK")
            require(sources()==spec["sources"] and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"V2_RUNTIME_SOURCE_CPU")
            ids,_=geo.orders[42].batch(i);require(ids==order["batches"][i],"V2_ORDER")
            samples=ds.load(ids);m.train();opt.zero_grad(set_to_none=True)
            if objective:
                z=objective(m,samples,geo,cal["old_weights"],TeacherGuidance(),i);rec,_,total=audit(m,z)
            else:
                base=safe_forward(m,samples,geo,spec["method_id"],TeacherGuidance(),cal["old_weights"])
                z=dict(base,base=base,terms={k:base["raw"].sum()*0 for k in KEYS},weights={k:0. for k in KEYS})
                rec,_,total=audit(m,z)
            ratio=rec["physical_imitation_ratio"];require(ratio is not None,"V2_ZERO_IMITATION")
            streak=streak+1 if ratio>10 else 0;require(streak<2,"V2_GRAD_GUARD")
            offset=0
            for n,p in m.named_parameters():
                p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
                if n.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"V2_SCORE_GRAD")
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),10.);require(bool(torch.isfinite(norm)),"V2_CLIP")
            for g in opt.param_groups:g["lr"]=lr_at(i)
            opt.step();completed=i+1
            require(rows_unchanged(m,score) and all(bool(torch.isfinite(p).all()) for p in m.parameters()),"V2_PARAMETER_GUARD")
            receipt=None if objective is None else objective.assignment.commit(i)
            write(run/("step%04d.json"%completed),dict(ids=ids,lr=lr_at(i),audit=rec,fair_commit=receipt,
                assignments=z.get("assignments"),diagnostics=z.get("diagnostics"),optimizer_step_succeeded=True))
            if completed in spec["checkpoint_steps"]:
                current=open8(m,read(artifact(cal["bindings"]))["development"])
                write(run/("OPEN_DEV8_step%04d.json"%completed),current)
                checkpoint(completed)
                for name,data in (("training384",items),("development192",dev)):
                    evaluate_set(m,data,run/"evaluation",name,completed,initial)
                metric_alarm(history,current,guardbase,{})
        write(run/"INFERENCE.json",inference_measure(m,items[0][0]))
        write(run/"RESULT.json",dict(status="COMPLETED_AWAITING_PAIRED_AUDIT",optimizer_steps=completed))
    except BaseException:
        if objective:objective.assignment.discard()
        write(run/"RESULT.json",dict(status="FAILED",optimizer_steps=completed,traceback=traceback.format_exc()))
        raise
    finally:
        # A seal records contents and status, including failures; it is not a PASS.
        write(run/"RUN_SEAL.json",dict(status="FAILED" if not (run/"RESULT.json").exists() else read(run/"RESULT.json")["status"],
            actual_optimizer_steps=completed,files={str(p.relative_to(run)):sha(p) for p in run.rglob("*") if p.is_file()}))

def transition(initial,final):
    def rows(path):
        with Path(path).open() as f:data=list(csv.DictReader(f))
        keys=[(r["sample_id"],int(r["query"])) for r in data]
        require(len(keys)==len(set(keys)),"V2_DUPLICATE_EVAL_KEY")
        return dict(zip(keys,data))
    a,b=rows(initial),rows(final);require(set(a)==set(b),"V2_EVAL_PAIR_KEYS")
    truth=lambda r,k:r[k]=="True"
    safe=[k for k,r in a.items() if truth(r,"hard_pass")]
    matched=[k for k in safe if truth(a[k],"matched")]
    return dict(fail_to_pass=sum(not truth(a[k],"hard_pass") and truth(b[k],"hard_pass") for k in a),
        pass_to_fail=sum(truth(a[k],"hard_pass") and not truth(b[k],"hard_pass") for k in a),
        initial_matched_safe=len(matched),initial_matched_safe_degraded=sum(not truth(b[k],"hard_pass") for k in matched),
        initial_safe=len(safe),initial_safe_degraded=sum(not truth(b[k],"hard_pass") for k in safe))

def compare():
    arms={arm:OUT/"runs"/("A42_"+arm+"_PG3V2_CPU128_run001") for arm in ("A","D")}
    specs={arm:read(p/"RUN_SPEC.json") for arm,p in arms.items()}
    shared=("initialization","bindings","sources","order128","old_weights","seed","threads","device","precision","batch_size","updates","optimizer","schedule")
    require(all(specs["A"][k]==specs["D"][k] for k in shared),"V2_NOT_PAIRED")
    require(all(read(p/"RESULT.json")["status"]=="COMPLETED_AWAITING_PAIRED_AUDIT" for p in arms.values()),"V2_RUN_NOT_COMPLETE")
    result=dict(semantics="A/D combination effect; geometric signatures are NOT homotopy; Development192 is NOT blind test",datasets={})
    for dataset in ("training384","development192"):
        files={arm:{step:p/"evaluation"/(dataset+"_step%04d.csv"%step) for step in (0,128)} for arm,p in arms.items()}
        require(sha(files["A"][0])==sha(files["D"][0]),"V2_INITIAL_PREDICTIONS_DIFFER")
        result["datasets"][dataset]=dict(A=transition(files["A"][0],files["A"][128]),D=transition(files["D"][0],files["D"][128]),
            A_to_D=transition(files["A"][128],files["D"][128]),
            summaries={arm:read(p/"evaluation"/(dataset+"_step0128_SUMMARY.json")) for arm,p in arms.items()})
    ck=torch.load(arms["D"]/"checkpoints/step0128.pt",map_location="cpu",weights_only=False)
    history=ck["fair"]["history"];eligible=sum(r["eligible"] for rows in history.values() for r in rows);direct=sum(r["direct"] for rows in history.values() for r in rows)
    result["exposure"]=dict(eligible_occurrences=eligible,direct_occurrences=direct,rate=None if not eligible else direct/eligible,
                            per_sample_query=history)
    result["inference"]={arm:read(p/"INFERENCE.json") for arm,p in arms.items()}
    result["sources"]={arm:dict(path=str(p/"RUN_SPEC.json"),sha256=sha(p/"RUN_SPEC.json")) for arm,p in arms.items()}
    folder=OUT/"summary";folder.mkdir(exist_ok=True);write(folder/"PAIRED_AD.json",result)

def main():
    global PREP
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","A","D","compare"]);p.add_argument("--authorization",type=Path);p.add_argument("--preparation-id",default="cpu0_002");a=p.parse_args()
    require(a.preparation_id and all(c.isalnum() or c in "_-" for c in a.preparation_id),"V2_PREPARATION_ID")
    PREP=OUT/"_preparation"/a.preparation_id
    if a.action=="prepare":prepare()
    elif a.action=="compare":compare()
    else:
        require(a.authorization is not None,"V2_AUTHORIZATION_FILE_REQUIRED");train(a.action,a.authorization)
if __name__=="__main__":main()
