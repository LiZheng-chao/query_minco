"""P1-A3 independent fixed-interpreter launcher. Large experiment manual only."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,csv
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
EXP=BASE/"experiments/query_minco_p1a3_validation_v1"
OUT=BASE/"research_outputs/query_minco_p1a3_validation_v1"
FREEZE=OUT/"protocol/freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
OLD=BASE/"research_outputs/query_minco_p1a2_falsification_v1"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as stream:json.dump(x,stream,indent=2,allow_nan=False);stream.write("\n")
def guard(phase,manual=False):
    e=read(OUT/"protocol/entry_001/ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("PROTECTED_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for k,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][k]:raise ValueError("GIT_DRIFT "+label)
    if sys.executable!=PYTHON:raise ValueError("PYTHON_IDENTITY")
    if not manual and time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
    if phase=="formal":
        if (FREEZE/"STOP.json").exists():raise ValueError("P1A3_PERSISTENT_STOP")
        for p,h in read(FREEZE/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise ValueError("SOURCE_DRIFT "+p)
        for p,h in read(FREEZE/"INPUT_HASHES.json").items():
            if sha(p)!=h:raise ValueError("INPUT_DRIFT "+p)
def worker_regression(dest):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    old=BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1/runs/acceptance_tests_001/PYTEST_COMMAND.json"
    original=read(old)["args"]
    selected=[a for a in original if not a.startswith(("--basetemp=","--junitxml="))]
    selected+=["experiments/query_minco_p1a_representation_v1/test_representation.py","experiments/query_minco_p1a2_falsification_v1/test_falsification.py",str(EXP/"test_alignment.py")]
    args=selected+["--basetemp="+str(dest/"tmp_tests"),"--junitxml="+str(dest/"TEST_RESULTS.xml")]
    write(dest/"PYTEST_COMMAND.json",dict(args=args,original_command_path=str(old),original_command_sha256=sha(old),nonsmooth_contract="unchanged counterexample called by existing expected-GradcheckError test; smooth/branch tests mandatory"))
    class Collector:
        def pytest_collection_finish(self,session):
            write(dest/"COLLECTION.json",dict(selected=len(session.items),nodeids=[x.nodeid for x in session.items]))
    code=int(pytest.main(args,plugins=[Collector()]))
    write(dest/"RESULT.json",dict(status="PASS" if code==0 else "P1A3_GATE_R_STOP",exit_code=code,optimizer_updates=0,network_updates=0))
    if code:raise SystemExit(code)
def worker_alignment(dest):
    import torch,numpy as np
    from experiments.query_minco_p1a3_validation_v1 import alignment as a
    torch.set_num_threads(1);records=[]
    for case,batch,contract,theta,meta in a.frozen_sources():
        rec,r=a.diagnose(batch,contract,theta,kind=meta["kind"],identity=meta["identity"],u=meta.get("u"),which=meta.get("which"))
        rec.update(case=case,source=meta["source"])
        if "old_record" in meta:
            old=meta["old_record"];coeff=old["coefficients"]
            diff=float(np.abs(np.asarray(rec["coefficients"])-np.asarray(coeff)).max())
            if diff>2e-4:raise ValueError("FROZEN_COEFFICIENT_DRIFT")
            if rec["OriginalHardAll"]!=old["hard"]["passed"]:raise ValueError("FROZEN_HARD_DRIFT")
            oldtask=old["task"] if "task" in old else old["Task"]
            if rec["TaskSuccess"]!=oldtask["TaskSuccess"]:raise ValueError("FROZEN_TASK_DRIFT")
            if "loss" in old and abs(rec["actual_Phase_A_loss"]-old["loss"])>.003+.003*abs(old["loss"]):raise ValueError("FROZEN_LOSS_DRIFT")
            rec["frozen_comparison"]=dict(max_coefficient_abs=diff,old_Hard=old["hard"]["passed"],old_TaskSuccess=oldtask["TaskSuccess"])
        records.append(rec)
        write(dest/("RECORD_"+str(len(records)).zfill(3)+".json"),rec)
    write(dest/"ALIGNMENT_RECORDS.json",records)
    evidence=[]
    for case in sorted({r["case"] for r in records}):
        pre=[r for r in records if r["case"]==case and r["OriginalHardAll"] and not r["TaskSuccess"] and not r["witness_read_only_evaluation"]]
        crossed=[r for r in records if r["case"]==case and r["TaskSuccess"] and r["witness_read_only_evaluation"]]
        pairs=[dict(prewall=p["identity"],crossing=q["identity"],prewall_L=p["actual_Phase_A_loss"],crossing_L=q["actual_Phase_A_loss"],crossing_minus_prewall=q["actual_Phase_A_loss"]-p["actual_Phase_A_loss"]) for p in pre for q in crossed]
        evidence.append(dict(case=case,prewall_count=len(pre),qualified_crossing_count=len(crossed),pairs=pairs,observed_ranking_inversion=any(p["crossing_minus_prewall"]>0 for p in pairs)))
    write(dest/"RANKING_COMPARISONS.json",evidence)
    write(dest/"RESULT.json",dict(status="PASS",records=len(records),qualification_evaluation_only=sum(r["witness_read_only_evaluation"] for r in records),optimizer_updates=0,network_updates=0,observed_inversions=[dict(case=x["case"],inversion=x["observed_ranking_inversion"]) for x in evidence]))
def worker_main(dest,phase):
    # Optimization path imports ONLY original analytic initialization/objective and current guard.
    # Never imports alignment.py, witness candidates, qualification.json or successful trajectories.
    import torch
    from experiments.query_minco_p1a2_falsification_v1 import falsification as f
    torch.set_num_threads(1);rows=[];updates=0;current=None
    try:
        for case,sample in f.CASE_SOURCES.items():
            batch,source=f.load_source(sample);contract=f.native_contract(batch,case) if case.startswith("native") else None
            controls,aliases=f.canonical_controls(batch.state)
            for seed in [None]+list(range(20261109,20261114)):
                for control in controls:
                    guard(phase,True);current=case+"__"+str(seed)+"__"+control;trial=dest/current;trial.mkdir()
                    u=f.initial(control,batch.state,seed).requires_grad_();which=f.arm(control);obj=f.p1.objective(which)
                    opt=torch.optim.AdamW([u],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
                    write(trial/"IDENTITY.json",dict(case=case,source=source,control=control,arm=which,seed=seed,initial_u=u.detach().tolist(),aliases=aliases,network_loaded=False,qualification_witness_consumed=False))
                    first_hard=None;first_task=None;first_cross=None;ever_channels=set();start=time.monotonic();initial=None
                    with (trial/"STEPS.jsonl").open("x") as log:
                        for step in range(65):
                            r=obj(f.p1.raw_slots(u),batch);g=torch.autograd.grad(r.loss,u)[0]
                            if not torch.isfinite(g).all() or not torch.isfinite(r.loss):raise ValueError("NONFINITE")
                            rec=f.p1.summarize(r,batch,u,g,arm=which)
                            task=f.crossing_success(r,batch,contract,rec["hard"]) if contract else dict(OriginalHardAll=rec["hard"]["passed"],TaskSuccess=rec["hard"]["passed"],successful_channels=[])
                            J=torch.autograd.functional.jacobian(lambda z:f.actual_theta(z,batch.state,which),u).detach().double();sv=torch.linalg.svdvals(J)
                            rec.update(step=step,Task=task,control=control,seed=seed,theta_Jacobian_singular_values=sv.tolist(),theta_Jacobian_rank=int((sv>1e-7).sum()),theta_Jacobian_condition=float(sv[0]/sv[-1]) if float(sv[-1])>1e-7 else None)
                            if step==0:initial=dict(Hard=rec["hard"]["passed"],TaskSuccess=task["TaskSuccess"],loss=rec["loss"])
                            if first_hard is None and rec["hard"]["passed"]:first_hard=step
                            if first_task is None and task["TaskSuccess"]:first_task=step
                            if first_cross is None and task.get("certified_gate_crossing",False):first_cross=step
                            if task["TaskSuccess"]:ever_channels.update(task["successful_channels"])
                            if step in (0,8,16,32,64):write(trial/("STEP_"+str(step)+".json"),rec)
                            if step<64:
                                opt.zero_grad(set_to_none=True);u.grad=g
                                norm=torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True);before=u.detach().clone();opt.step();updates+=1
                                rec.update(preclip_norm=float(norm),update_norm_FP64=float((u.detach()-before).double().norm()))
                            log.write(json.dumps(rec,allow_nan=False)+"\n");log.flush()
                    row=dict(case=case,sample_id=sample,control=control,seed=seed,steps=64,elapsed_s=time.monotonic()-start,initial=initial,first_Hard_step=first_hard,ever_Hard=first_hard is not None,first_TaskSuccess_step=first_task,ever_TaskSuccess=first_task is not None,first_certified_crossing_step=first_cross,final_Hard=rec["hard"]["passed"],final_TaskSuccess=task["TaskSuccess"],final_loss=rec["loss"],final_channels=task["successful_channels"],ever_successful_channels=sorted(ever_channels),aliases=aliases)
                    rows.append(row);write(trial/"RESULT.json",row);write(trial/"COMPLETION.json",dict(status="PASS",leaf_updates=64));print(json.dumps(row),flush=True)
        write(dest/"PAIRED_ROWS.json",rows);write(dest/"RESULT.json",dict(status="PASS",trials=len(rows),trajectory_parameter_updates=updates,network_updates=0))
    except BaseException:
        write(dest/"PARTIAL_ROWS.json",rows);write(dest/"RESULT.json",dict(status="STOP",trials_completed=len(rows),current_trial=current,trajectory_parameter_updates=updates,network_updates=0,error=traceback.format_exc()));raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["regression","alignment","main"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--manual-main",action="store_true");ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    if a.stage=="main" and (not a.manual_main or a.phase!="formal"):raise ValueError("MANUAL_MAIN_ONLY")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id;sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase,a.stage=="main")
        if a.stage=="regression":worker_regression(dest)
        elif a.stage=="alignment":worker_alignment(dest)
        else:worker_main(dest,a.phase)
        return
    dest.mkdir();write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    write(dest/"SOURCE_SNAPSHOT.json",{str(p.relative_to(BASE)):dict(sha256=sha(p),content=p.read_text()) for p in EXP.glob("*") if p.suffix in (".md",".py")})
    try:
        guard(a.phase,a.stage=="main")
        if a.stage in ("alignment","main"):
            if read(OUT/"runs/acceptance_regression_001/RESULT.json")["status"]!="PASS":raise ValueError("GATE_R_REQUIRED")
        if a.stage=="main" and read(OUT/"runs/acceptance_alignment_001/RESULT.json")["status"]!="PASS":raise ValueError("GATE_Q_REQUIRED")
        # Parent identity + imports validated under the same interpreter, before launching children.
        from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
        write(dest/"PARENT_DEPENDENCIES.json",bootstrap())
        spec=importlib.util.spec_from_file_location("_accepted_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(dest/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]+(["--manual-main"] if a.manual_main else [])
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else FREEZE/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,14400 if a.stage=="main" else 600)
        guard(a.phase,a.stage=="main");write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc();(dest/"ENTRY_stderr.log").open("x").write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="P1A3_GATE_R_STOP" if a.stage=="regression" else "STOP",exit_code=1,error=error))
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else FREEZE/"STOP.json"
        if not stop.exists():write(stop,dict(status="STOP",stage=a.stage,error=error))
        raise
if __name__=="__main__":main()
