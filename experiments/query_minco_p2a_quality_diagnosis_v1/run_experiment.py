"""P2-A bounded offline diagnosis and manual-only larger experiment."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,csv
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
EXP=BASE/"experiments/query_minco_p2a_quality_diagnosis_v1"
OUT=BASE/"research_outputs/query_minco_p2a_quality_diagnosis_v1"
FREEZE=OUT/"protocol/diagnostic_freeze_001"
PILOT=OUT/"protocol/pilot_freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as stream:json.dump(x,stream,indent=2,allow_nan=False);stream.write("\n")
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open("x",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def guard(phase,stage):
    if sys.executable!=PYTHON:raise ValueError("PYTHON_IDENTITY")
    e=read(OUT/"protocol/entry_001/ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("PROTECTED_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for k,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][k]:raise ValueError("GIT_DRIFT "+label)
    if stage!="main" and time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
    if phase=="formal":
        if (FREEZE/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
        for p,h in read(FREEZE/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise ValueError("SOURCE_DRIFT "+p)
        for p,h in read(FREEZE/"INPUT_HASHES.json").items():
            if sha(p)!=h:raise ValueError("INPUT_DRIFT "+p)
        if stage in ("pilot","main"):
            spec=read(PILOT/"RUNSPEC.json")
            if sha(OUT/"runs/acceptance_calibration_001/RESULT.json")!=spec["calibration_RESULT_sha256"]:raise ValueError("CALIBRATION_DRIFT")
            if sha(FREEZE/"SOURCE_FREEZE.json")!=spec["source_freeze_sha256"]:raise ValueError("PILOT_SOURCE_DRIFT")
def worker_tests(dest):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    old=BASE/"research_outputs/query_minco_p1a3_validation_v1/runs/acceptance_regression_001/PYTEST_COMMAND.json"
    args=[a for a in read(old)["args"] if not a.startswith(("--basetemp=","--junitxml="))]
    args+=[str(EXP/"test_quality.py"),"--basetemp="+str(dest/"tmp_tests"),"--junitxml="+str(dest/"TEST_RESULTS.xml")]
    write(dest/"PYTEST_COMMAND.json",dict(args=args,original=str(old),original_sha256=sha(old)))
    class Collect:
        def pytest_collection_finish(self,session):write(dest/"COLLECTION.json",dict(selected=len(session.items),nodeids=[i.nodeid for i in session.items]))
    code=int(pytest.main(args,plugins=[Collect()]))
    write(dest/"RESULT.json",dict(status="PASS" if code==0 else "STOP",exit_code=code,network_updates=0))
    if code:raise SystemExit(code)
def worker_gate0(dest):
    import torch
    from experiments.query_minco_p2a_quality_diagnosis_v1.diagnosis import paired_reanalysis
    torch.set_num_threads(1);pairs,events,result=paired_reanalysis()
    write(dest/"PAIRED.json",pairs);write(dest/"EVENTS.json",events);csvout(dest/"P2A_MAIN_REANALYSIS.csv",pairs);csvout(dest/"EVENTS.csv",events);write(dest/"RESULT.json",result)
def worker_gate1(dest):
    import torch
    from experiments.query_minco_p2a_quality_diagnosis_v1.diagnosis import gradient_update_records
    torch.set_num_threads(1);rows=gradient_update_records()
    write(dest/"GRADIENT_RECORDS.json",rows);csvout(dest/"P2A_GRADIENT_VS_UPDATE.csv",rows)
    write(dest/"RESULT.json",dict(status="PASS",points=len(rows),optimizer_updates=0,network_updates=0))
def worker_calibration(dest):
    import torch,numpy as np
    from experiments.query_minco_p2a_quality_diagnosis_v1.quality import f,parts,CALIBRATION_ALPHA
    torch.set_num_threads(1);records=[]
    for sample in ("train_free_roi_1101_p0_near","train_free_roi_1101_p0_far","train_sparse_1102_p0_near"):
        batch,source=f.load_source(sample)
        for seed in (None,20261109,20261110):
            u=f.initial("B0_nominal",batch.state,seed).requires_grad_();r=f.p1.objective("B0")(f.p1.raw_slots(u),batch)
            gp=torch.autograd.grad(.025*r.metrics["P"].mean(),u,retain_graph=True)[0]
            gc=torch.autograd.grad(.1*(r.metrics["C_shape"]+r.metrics["S"]).mean(),u)[0]
            pn=float(gp.double().norm());cn=float(gc.double().norm())
            if not np.isfinite([pn,cn]).all():raise ValueError("NONFINITE_CALIBRATION")
            records.append(dict(sample=sample,scene=source["scene_id"],seed=seed,progress_norm=pn,shape_plus_S_norm=cn,ratio=pn/max(cn,1e-12),informative=pn>1e-6,source=source))
    scenes=sorted({r["scene"] for r in records})
    medians={s:float(np.median([r["ratio"] for r in records if r["scene"]==s and r["informative"]])) for s in scenes if any(r["informative"] for r in records if r["scene"]==s)}
    if not medians:raise ValueError("NO_INFORMATIVE_CALIBRATION")
    eligible=[alpha for alpha in CALIBRATION_ALPHA if all(alpha*r>=1 for r in medians.values())]
    alpha=min(eligible) if eligible else max(CALIBRATION_ALPHA)
    write(dest/"CALIBRATION_RECORDS.json",records)
    write(dest/"RESULT.json",dict(status="PASS",alpha=alpha,calibration_target_met=bool(eligible),per_scene_median_base_ratio=medians,records=9,excluded_near_zero_progress=sum(not r["informative"] for r in records),gate_or_witness_used=False,coefficient_selected_by="pre-registered train-only gradient ratios; no success tuning"))
def worker_pilot(dest,manual,phase):
    import torch
    from experiments.query_minco_p2a_quality_diagnosis_v1.quality import f,ARMS,PILOT_CASES,PILOT_SEEDS,scalar,parts,accept
    torch.set_num_threads(1);spec=read(PILOT/"RUNSPEC.json");alpha=spec["alpha"];budget=64 if manual else 8
    seeds=[None]+list(range(20261109,20261114)) if manual else PILOT_SEEDS
    rows=[];proposals=0;applied=0;current=None
    try:
        for case in PILOT_CASES:
            batch,source=f.load_source(f.CASE_SOURCES[case]);contract=f.native_contract(batch,case) if case.startswith("native") else None
            for seed in seeds:
                for arm in ARMS:
                    guard(phase,"main" if manual else "pilot")
                    current=case+"__"+str(seed)+"__"+arm;trial=dest/current;trial.mkdir()
                    u=f.initial("B0_nominal",batch.state,seed).requires_grad_();obj=f.p1.objective("B0")
                    opt=torch.optim.AdamW([u],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
                    write(trial/"IDENTITY.json",dict(case=case,seed=seed,arm=arm,alpha=alpha,initial_u=u.detach().tolist(),source=source,decoder="B0",witness_consumed=False,optimizer_only_15D_leaf=True))
                    firstH=None;firstT=None;everchannels=set();initial=None;rejects=0
                    with (trial/"STEPS.jsonl").open("x") as log:
                        for step in range(budget+1):
                            r=obj(f.p1.raw_slots(u),batch);loss=scalar(r,arm,alpha)
                            grad=torch.autograd.grad(loss,u,retain_graph=True)[0]
                            if not torch.isfinite(grad).all() or not torch.isfinite(loss):raise ValueError("NONFINITE")
                            rec=f.p1.summarize(r,batch,u,grad,arm="B0")
                            task=f.crossing_success(r,batch,contract,rec["hard"]) if contract else dict(OriginalHardAll=rec["hard"]["passed"],TaskSuccess=rec["hard"]["passed"],successful_channels=[])
                            v,q=parts(r,alpha);rec.update(step=step,experiment_arm=arm,alpha=alpha,optimization_scalar=float(loss),q_alpha=float(q),Task=task)
                            rec["weighted_raw_component_gradients"]={}
                            for name,factor in (("C_viol",1.),("F",1.),("G_task_train",1.),("C_shape",.1),("P",.025*(1 if arm=="L0" else alpha)),("S",.1)):
                                cg=torch.autograd.grad(factor*r.metrics[name].mean(),u,retain_graph=True)[0]
                                rec["weighted_raw_component_gradients"][name]=cg.detach().tolist()
                            if step==0:initial=dict(Hard=rec["hard"]["passed"],TaskSuccess=task["TaskSuccess"],progress=rec["hard"]["goal"]["distance_reduction_m"],loss0=rec["loss"])
                            if firstH is None and rec["hard"]["passed"]:firstH=step
                            if firstT is None and task["TaskSuccess"]:firstT=step
                            if task["TaskSuccess"]:everchannels.update(task["successful_channels"])
                            if step in (0,8,16,32,64,budget):write(trial/("STATE_"+str(step)+".json"),rec)
                            if step<budget:
                                opt.zero_grad(set_to_none=True);u.grad=grad;norm=torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True)
                                before=u.detach().clone();opt.step();proposals+=1
                                proposed=obj(f.p1.raw_slots(u),batch);pr=f.p1.summarize(proposed,batch,u,arm="B0")
                                pt=f.crossing_success(proposed,batch,contract,pr["hard"]) if contract else dict(TaskSuccess=pr["hard"]["passed"],successful_channels=[])
                                _,pq=parts(proposed,alpha)
                                ok,reason=accept(arm,rec["hard"]["passed"],pr["hard"]["passed"],float(q),float(pq),pr["UNKNOWN_count"])
                                proposal_u=u.detach().clone()
                                if not ok:
                                    with torch.no_grad():u.copy_(before)
                                    rejects+=1
                                delta=u.detach()-before
                                applied+=int(bool(delta.any()))
                                rec.update(proposal_u=proposal_u.tolist(),proposed_Hard=pr["hard"]["passed"],proposed_TaskSuccess=pt["TaskSuccess"],proposal_full_record=pr,accepted=ok,filter_reason=reason,preclip_norm=float(norm),proposed_du=(proposal_u-before).tolist(),actual_du=delta.tolist(),actual_update_norm_FP64=float(delta.double().norm()),gradient_vs_actual_du_cosine=None if not bool(delta.any()) else float(torch.nn.functional.cosine_similarity(-grad.double(),delta.double(),dim=0)),Adam_m=opt.state[u]["exp_avg"].tolist(),Adam_v=opt.state[u]["exp_avg_sq"].tolist(),moment_state_retained_on_rejection=True)
                            log.write(json.dumps(rec,allow_nan=False)+"\n");log.flush()
                    row=dict(case=case,arm=arm,seed=seed,steps=budget,initial=initial,ever_Hard=firstH is not None,first_Hard_step=firstH,ever_TaskSuccess=firstT is not None,first_TaskSuccess_step=firstT,final_Hard=rec["hard"]["passed"],final_TaskSuccess=task["TaskSuccess"],final_progress=rec["hard"]["goal"]["distance_reduction_m"],final_goal_error=rec["hard"]["goal"]["endpoint_goal_error_m"],final_maxima=rec["hard"]["dynamic_polynomial_maxima"],final_loss0=rec["loss"],final_optimization_scalar=rec["optimization_scalar"],final_q_alpha=rec["q_alpha"],final_UNKNOWN=rec["UNKNOWN_count"],final_path_length=rec["path_length_m"],final_certified_surface_clearance_lower_bound=rec["certified_surface_clearance_lower_bound_m"],final_jerk_quality=rec["metrics"]["S"],final_reasons=rec["hard"]["reasons"],channels=sorted(everchannels),rejected_steps=rejects,quality_status="JOINT_FEASIBLE" if task["TaskSuccess"] else "NA_NO_JOINT_SUCCESS")
                    rows.append(row);write(trial/"RESULT.json",row);print(json.dumps(row),flush=True)
        write(dest/"PILOT_ROWS.json",rows);csvout(dest/"P2A_PILOT_RESULTS.csv",rows)
        write(dest/"RESULT.json",dict(status="PASS",trials=len(rows),AdamW_proposals=proposals,applied_nonzero_moves=applied,steps_per_trial=budget,network_updates=0,alpha=alpha,manual_main=manual))
    except BaseException:
        write(dest/"PARTIAL_ROWS.json",rows);write(dest/"RESULT.json",dict(status="STOP",current_trial=current,completed_trials=len(rows),AdamW_proposals=proposals,applied_nonzero_moves=applied,error=traceback.format_exc()));raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["tests","gate0","gate1","calibration","pilot","main"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--worker",action="store_true");ap.add_argument("--manual-main",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    if a.stage=="main" and (not a.manual_main or a.phase!="formal"):raise ValueError("MANUAL_MAIN_ONLY")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id;sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase,a.stage)
        if a.stage=="tests":worker_tests(dest)
        elif a.stage=="gate0":worker_gate0(dest)
        elif a.stage=="gate1":worker_gate1(dest)
        elif a.stage=="calibration":worker_calibration(dest)
        else:worker_pilot(dest,a.stage=="main",a.phase)
        return
    dest.mkdir();write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    write(dest/"SOURCE_SNAPSHOT.json",{str(p.relative_to(BASE)):dict(sha256=sha(p),content=p.read_text()) for p in EXP.glob("*") if p.suffix in (".md",".py")})
    try:
        guard(a.phase,a.stage)
        dependencies={"gate1":["gate0","tests"],"calibration":["gate0","gate1","tests"],"pilot":["gate0","gate1","tests","calibration"],"main":["gate0","gate1","tests","calibration","pilot"]}.get(a.stage,[])
        for stage in dependencies:
            if read(OUT/"runs"/("acceptance_"+stage+"_001")/"RESULT.json")["status"]!="PASS":raise ValueError("GATE_REQUIRED "+stage)
        from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
        write(dest/"PARENT_DEPENDENCIES.json",bootstrap())
        spec=importlib.util.spec_from_file_location("_accepted_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(dest/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]+(["--manual-main"] if a.manual_main else [])
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else FREEZE/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,14400 if a.stage=="main" else 600)
        guard(a.phase,a.stage);write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc();(dest/"ENTRY_stderr.log").open("x").write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else FREEZE/"STOP.json"
        if not stop.exists():write(stop,dict(status="STOP",stage=a.stage,error=error))
        raise
if __name__=="__main__":main()
