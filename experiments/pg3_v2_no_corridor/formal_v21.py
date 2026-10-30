"""Independent paired128 executor: frozen formulas, explicit authorization, strict stops."""
import argparse,copy,json,os,resource,subprocess,time,traceback
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,verify_code
from query_minco.training.initialization import configure
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_v2_no_corridor.executor_v21 import Engine,authorize_spec,publish_checkpoint,inspect_chain,restore_checkpoint,validate_frozen_spec
from experiments.pg3_feasible_mode_v1.fulltrain128 import load_items
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import evaluate_set,inference_measure
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm
from .qualification_v21 import snapshot,ref
OLD=Path("/home/lizheng/query_minco_mainline/research_outputs/pg3_v2_1_no_corridor/_preparation/final_head002")

def bundle(preparation):
    p=Path(preparation);q=read(p/"QUALIFICATION.json");require(q["status"]=="QUALIFIED","FORMAL_NOT_QUALIFIED")
    verify_code(q["code"]);require(snapshot()==q["sources"],"FORMAL_SOURCE_DRIFT")
    for key in ("data","cpu0","tests","gradient_audit","shadow_probe"):artifact(q[key])
    prep=read(p/"PREPARATION.json");specs={a:read(artifact(q["runspecs"][a])) for a in ("A","D")}
    frozen=read(OLD/"RUN_SPEC_D.json")
    require(specs["D"]["weights"]==frozen["weights"] and specs["D"]["pg2_weights"]==frozen["pg2_weights"],"FORMAL_FROZEN_WEIGHT_DRIFT")
    require(sha(artifact(prep["configuration"]))==sha(artifact(frozen["configuration"])),"FORMAL_FROZEN_RECIPE_DRIFT")
    return prep,specs,q

def plan(preparation,output):
    prep,specs,q=bundle(preparation);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=out.parent
    for a in ("A","D"):
        for suffix,limit,purpose in (("formal",128,"PG3_V21_PAIRED128"),("short_u",4,"PG3_V21_SHORT4"),("short_r",4,"PG3_V21_SHORT4")):
            s=copy.deepcopy(specs[a]);s["schema"]="pg3_v21_paired128_executor_v1";s["run_id"]="A42_"+a+"_PG3V21_"+suffix+"_run001"
            s["output_root"]=str(base/("runs" if limit==128 else "short"));s["preparation"]=str(Path(preparation).resolve())
            s["execution"]=dict(limit=limit,purpose=purpose,checkpoint_every=1,wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4)
            write(out/(a+"_"+suffix+".json"),s)

def resources(run,start,spec):
    r=spec["execution"];available=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
    require(time.monotonic()-start<r["wall_s"] and available>=r["min_available_mib"] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=r["rss_mib"],"FORMAL_RESOURCE_LIMIT")
    require(sum(p.stat().st_size for p in run.rglob("*") if p.is_file())<r["disk_gib"]*1024**3,"FORMAL_DISK_LIMIT")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"FORMAL_CPU_CONTRACT")

def run(spec_path,authorization,resume=False,pause_after=None):
    s=read(spec_path);auth=read(authorization);authorize_spec(s,auth);prep,qualified,q=bundle(s["preparation"]);validate_frozen_spec(s,qualified[s["arm"]])
    require(s["sources"]==q["sources"] and s["code"]==q["code"],"FORMAL_SPEC_SOURCE")
    limit=s["execution"]["limit"];root=Path(s["output_root"]);run=root/s["run_id"]
    require(pause_after is None or (limit==4 and 0<pause_after<limit),"FORMAL_PAUSE_SCOPE")
    if limit==128:
        short=read(root.parent/"short/SHORT_PAIR_RESULT.json");require(short["status"]=="PASS" and short["code_commit"]==s["code"]["git_commit"],"FORMAL_SHORT_PREFLIGHT_FAILED")
    if not resume:run.mkdir(parents=True,exist_ok=False);write(run/"RUN_SPEC.json",s);write(run/"AUTHORIZATION.json",auth)
    else:
        require(limit==4 and read(run/"RUN_SPEC.json")==s,"FORMAL_RESUME_SCOPE")
        previous=sorted(run.glob("PROCESS_*_RESULT.json"));require(previous and read(previous[-1])["status"]=="PAUSED_FOR_RESUME_TEST","FORMAL_FAILED_RUN_NOT_RESUMABLE")
    phase=len(list(run.glob("PROCESS_*_START.json")));write(run/("PROCESS_%02d_START.json"%phase),dict(argv=__import__("sys").argv,resume=resume))
    configure(dict(seed=42,threads=1,device="cpu"));old=read(artifact(prep["historical_cpu0"]));cal=dict(old,bindings=prep["bindings"])
    ds,geo,init,qualification,train,dev=load_items(cal)
    from experiments.pg3_feasible_mode_v1.common import model_from
    engine=Engine(model_from(init),s,read(artifact(prep["configuration"])));order=read(artifact(s["order"]));start=time.monotonic();last=None;status="FAILED";error=None
    try:
        if resume:last=restore_checkpoint(run,engine)
        else:
            if limit==128:
                targets={}
                for name,items in (("training384",train),("development192",dev)):
                    _,t=evaluate_set(engine.model,items,run/"evaluation",name,0);targets.update(t)
                write(run/"INITIAL_TARGETS.json",targets);write(run/"INITIAL_OPEN_DEV8.json",open8(engine.model,read(artifact(prep["bindings"]))["development"]))
            last=publish_checkpoint(run,engine,dict(completed=0,initial_model_sha256=state_sha(engine.model)))
        while engine.completed<limit:
            resources(run,start,s);require(snapshot()==s["sources"],"FORMAL_RUNTIME_SOURCE_DRIFT")
            i=engine.completed;ids,roles=geo.orders[42].batch(i);require(ids==order["batches"][i],"FORMAL_ORDER_DRIFT")
            rec=engine.step(ds.load(ids),geo);rec["roles"]=roles;alarm=None
            if limit==128 and engine.completed in (32,64,96,128):
                cur=open8(engine.model,read(artifact(prep["bindings"]))["development"]);history=engine.monitor.setdefault("evaluation_alarm_history",[])
                try:metric_alarm(history,cur,read(run/"INITIAL_OPEN_DEV8.json"),{})
                except ValueError as ex:alarm=str(ex)
                rec.update(open_dev8=cur,checkpoint_alarm=alarm)
            last=publish_checkpoint(run,engine,rec,sha(last/"COMMIT.json"))
            print("EXEC_UPDATE",s["arm"],engine.completed,rec["audit"]["ratios"],"P2F",rec["metrics"]["pass_to_fail"],flush=True)
            if limit==128 and engine.completed in (32,64,96,128):
                for name,items in (("training384",train),("development192",dev)):evaluate_set(engine.model,items,run/"evaluation",name,engine.completed,read(run/"INITIAL_TARGETS.json"))
                write(run/("INFERENCE_step%04d.json"%engine.completed),inference_measure(engine.model,train[0][0]))
            if alarm:raise ValueError(alarm)
            if pause_after is not None and engine.completed==pause_after:status="PAUSED_FOR_RESUME_TEST";break
        else:status="COMPLETED"
        inspect_chain(run,s)
    except BaseException:
        error=traceback.format_exc();engine.exposure.rollback(engine.completed)
        write(run/("REJECTED_PROCESS_%02d.json"%phase),dict(phase=engine.phase,completed_in_process=engine.completed,last_committed=None if last is None else str(last),
            audit=engine.last_audit,error=error,optimizer_state_steps=[int(v["step"]) for v in engine.opt.state.values()]))
    result=dict(status=status,completed_updates=engine.completed,formal_updates=engine.completed if limit==128 else 0,diagnostic_updates=engine.completed if limit==4 else 0,
        code_commit=s["code"]["git_commit"],model_sha256=state_sha(engine.model),exposure_sha256=digest(engine.exposure.state_dict()),last_committed=None if last is None else ref(last/"COMMIT.json"),wall_s=time.monotonic()-start,error=error)
    write(run/("PROCESS_%02d_RESULT.json"%phase),result)
    if status=="FAILED":raise RuntimeError(error)
    return result

def state_fingerprint(path):
    import hashlib
    blob=torch.load(path,map_location="cpu",weights_only=True);h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit({k:blob[k] for k in ("model","optimizer","exposure","torch_rng")});return h.hexdigest()

def compare_short(base):
    base=Path(base);rows={};passed=True
    for arm in ("A","D"):
        paths={suffix:base/("A42_"+arm+"_PG3V21_"+suffix+"_run001") for suffix in ("short_u","short_r")}
        specs={k:read(p/"RUN_SPEC.json") for k,p in paths.items()};latest={k:inspect_chain(p,specs[k]) for k,p in paths.items()}
        results={k:read(sorted(p.glob("PROCESS_*_RESULT.json"))[-1]) for k,p in paths.items()}
        good=all(r["status"]=="COMPLETED" and r["completed_updates"]==4 for r in results.values())
        fingerprints={k:state_fingerprint(p/"checkpoint.pt") for k,p in latest.items()};good=good and len(set(fingerprints.values()))==1
        updates={k:[read(p/"commits"/("step%04d"%i)/"UPDATE.json") for i in range(1,read(latest[k]/"COMMIT.json")["step"]+1)] for k,p in paths.items()}
        good=good and all(updates["short_u"][i]["ids"]==updates["short_r"][i]["ids"] and updates["short_u"][i]["exposure_sha256"]==updates["short_r"][i]["exposure_sha256"] and updates["short_u"][i]["model_sha256"]==updates["short_r"][i]["model_sha256"] for i in range(4))
        rows[arm]=dict(passed=good,results=results,fingerprints=fingerprints,first_step=updates["short_u"][0]["metrics"] if updates["short_u"] else None,steps=[u["metrics"] for u in updates["short_u"]]);passed=passed and good
    write(base/"SHORT_PAIR_RESULT.json",dict(status="PASS" if passed else "FAILED",code_commit=specs["short_u"]["code"]["git_commit"],arms=rows))
    require(passed,"FORMAL_SHORT_RESUME_MISMATCH")

def summarize(base):
    import csv
    base=Path(base);result=dict(formal={},paired_eval="NOT_RUN",training="NOT_RUN",forest="NOT_RUN",online_control="NOT_RUN",flight="NOT_RUN")
    paths={a:base/"runs"/("A42_"+a+"_PG3V21_formal_run001") for a in ("A","D")}
    for a,p in paths.items():
        records=sorted(p.glob("PROCESS_*_RESULT.json")) if p.exists() else []
        if not records:result["formal"][a]=dict(status="NOT_RUN");continue
        r=read(records[-1]);result["formal"][a]=r;result["training"]="EXECUTED"
        result["formal"][a]["evaluations"]={}
        for dataset in ("training384","development192"):
            summaries=sorted((p/"evaluation").glob(dataset+"*_SUMMARY.json"))
            if summaries:result["formal"][a]["evaluations"][dataset]=read(summaries[-1])
    if all(result["formal"][a]["status"]=="COMPLETED" and result["formal"][a]["completed_updates"]==128 for a in paths):
        from .run import transition
        result["paired_eval"]="COMPLETED";result["paired"]={}
        for dataset in ("training384","development192"):
            files={a:{i:p/"evaluation"/(dataset+"_step%04d.csv"%i) for i in (0,128)} for a,p in paths.items()}
            require(sha(files["A"][0])==sha(files["D"][0]),"FORMAL_UNPAIRED_INITIAL_PREDICTIONS")
            result["paired"][dataset]=dict(A=transition(files["A"][0],files["A"][128]),D=transition(files["D"][0],files["D"][128]),A_to_D=transition(files["A"][128],files["D"][128]))
    short=base/"short/SHORT_PAIR_RESULT.json"
    result["short"]=read(short) if short.exists() else dict(status="NOT_COMPLETED")
    result["single_step_comparison"]=read(base/"stage0/PAIRED_STEP1.json")["arms"]
    folder=base/"summary";folder.mkdir(exist_ok=True);write(folder/"PAIRED128_RESULT.json",result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["plan","run","compare-short","summarize"]);p.add_argument("--preparation");p.add_argument("--output",type=Path);p.add_argument("--spec");p.add_argument("--authorization");p.add_argument("--resume",action="store_true");p.add_argument("--pause-after",type=int);a=p.parse_args()
    if a.action=="plan":plan(a.preparation,a.output)
    elif a.action=="compare-short":compare_short(a.output)
    elif a.action=="summarize":summarize(a.output)
    else:run(a.spec,a.authorization,a.resume,a.pause_after)
if __name__=="__main__":main()
