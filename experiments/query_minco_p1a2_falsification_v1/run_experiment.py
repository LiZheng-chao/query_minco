"""Exclusive monitored P1-A2 diagnostic launcher. Main experiment is manual only."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
EXP=BASE/"experiments/query_minco_p1a2_falsification_v1"
OUT=BASE/"research_outputs/query_minco_p1a2_falsification_v1"
FREEZE=OUT/"protocol/freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard(phase,manual=False):
    e=read(OUT/"protocol/entry_001/ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("PROTECTED_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for k,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][k]:raise ValueError("GIT_DRIFT "+label)
    if not manual and time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
    if phase=="formal":
        if (FREEZE/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
        for p,h in read(FREEZE/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise ValueError("SOURCE_DRIFT "+p)
        for p,h in read(FREEZE/"INPUT_HASHES.json").items():
            if sha(p)!=h:raise ValueError("INPUT_DRIFT "+p)
def worker_tests(dest):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"tmp_tests"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),str(EXP/"test_falsification.py")]
    write(dest/"PYTEST_COMMAND.json",dict(args=args))
    code=int(pytest.main(args));write(dest/"RESULT.json",dict(status="PASS" if code==0 else "FAIL",pytest_exit=code))
    if code:raise SystemExit(code)
def worker_a(dest):
    import torch,numpy as np
    from experiments.query_minco_p1a2_falsification_v1 import falsification as f
    torch.set_num_threads(1);rows=[]
    for sample in f.p1.PILOT:
        batch,source=f.load_source(sample);state=batch.state;records={}
        for control in f.CONTROLS:
            u=f.initial(control,state).requires_grad_();which=f.arm(control)
            result=f.p1.objective(which)(f.p1.raw_slots(u),batch)
            grad=torch.autograd.grad(result.loss,u)[0]
            J=torch.autograd.functional.jacobian(lambda z:f.actual_theta(z,state,which),u).detach().double().numpy()
            S=np.linalg.svd(J,compute_uv=False)
            rec=f.p1.summarize(result,batch,u,grad,arm=which)
            rec.update(control=control,sample_id=sample,source=source,theta_Jacobian=J.tolist(),singular_values=S.tolist(),rank=int((S>1e-7).sum()),condition=float(S[0]/S[-1]),raw_gradient_norm_FP64=float(grad.double().norm()),network_updates=0)
            records[control]=rec
        comparisons=[]
        for ref in ("B0","B1"):
            a=records["A_matched_"+ref+"_ref"];b=records[ref+"_nominal"]
            th=float(np.abs(np.asarray(a["theta"])-np.asarray(b["theta"])).max())
            # Curves compared using actual production coefficients on a common time grid.
            ua=f.initial("A_matched_"+ref+"_ref",state);ub=f.initial(ref+"_nominal",state)
            dp=float((f.p1.samples_vector(ua,state,"A")-f.p1.samples_vector(ub,state,ref)).abs().max())
            dl=abs(a["loss"]-b["loss"]);ha=a["hard"];hb=b["hard"]
            ok=th<=3e-6 and dp<=2e-4 and dl<=.003+.003*abs(b["loss"]) and ha["passed"]==hb["passed"] and sorted(ha["reasons"])==sorted(hb["reasons"])
            va=np.asarray(ha["dynamic_polynomial_maxima"]);vb=np.asarray(hb["dynamic_polynomial_maxima"])
            ok=bool(ok and np.all(np.abs(va-vb)<=.001+.0001*np.abs(vb)))
            comparisons.append(dict(reference=ref,theta_max_abs=th,curve_max_abs_m=dp,loss_abs=dl,matched=ok))
        item=dict(sample_id=sample,records=records,comparisons=comparisons)
        rows.append(item);write(dest/(sample+".json"),item)
        if not all(x["matched"] for x in comparisons):raise ValueError("MATCHED_INITIALIZATION")
    write(dest/"A_MATCHED_INITIALIZATION.json",rows)
    write(dest/"RESULT.json",dict(status="PASS",tasks=len(rows),controls=5,comparisons=10,optimizer_steps=0,network_updates=0))
def worker_b(dest):
    import torch
    from experiments.query_minco_p1a2_falsification_v1 import falsification as f
    from experiments.query_minco_p1a2_falsification_v1.qualification import qualification_candidates
    torch.set_num_threads(1);items=[]
    for case,sample in f.CASE_SOURCES.items():
        if not case.startswith("native"):continue
        batch,source=f.load_source(sample);contract=f.native_contract(batch,case)
        out=qualification_candidates(batch,contract)
        out.update(case_id=case,source=source,contract=contract)
        write(dest/(case+".json"),out);items.append(out)
    write(dest/"REAL_GATE_TASKS.json",items)
    # Explicitly audit the old final Hard-valid wall trajectories, preserving old status.
    old=BASE/"research_outputs/query_minco_p1a_representation_v1/runs/pilot_001"
    audit=[]
    import numpy as np
    from types import SimpleNamespace
    from query_minco.trajectory.minco_adapter import MincoAdapter
    for sample,case in (("train_single_slit_1104_p0_near","native_slit_cross_far_v1"),("train_two_portals_1103_p0_far","native_two_portals_cross_far_v1")):
        batch,_=f.load_source(sample);contract=f.native_contract(batch,case)
        for init in ("nominal","noise"):
            for which in ("B0","B1"):
                path=old/(sample+"__"+init+"__"+which)/"FINAL.json"
                rec=read(path);u=torch.tensor(rec["u"],dtype=torch.float32)
                result=f.p1.objective(which)(f.p1.raw_slots(u),batch)
                outcome=f.crossing_success(result,batch,contract)
                audit.append(dict(source_path=str(path),source_sha256=sha(path),sample_id=sample,arm=which,initialization=init,old_hard=rec["hard"]["passed"],task=outcome))
    write(dest/"OLD_TASK_CROSSING_AUDIT.json",audit)
    good=all(x["status"]=="FEASIBLE_CROSSING_CERTIFIED" for x in items)
    write(dest/"RESULT.json",dict(status="PASS" if good else "TASK_NOT_QUALIFIED",tasks=2,candidates=sum(len(x["candidates"]) for x in items),old_final_trajectories=len(audit),optimizer_steps=0,network_updates=0,impossible_counter=dict(required_progress_m=6.5,max_speed_m_s=3,duration_upper_s=2.00000006,status="INFEASIBLE_BY_SPEED_TIME")))
    if not good:raise ValueError("GATE_B_TASK_NOT_QUALIFIED")
def worker_opt(dest,manual,phase):
    import torch
    from experiments.query_minco_p1a2_falsification_v1 import falsification as f
    torch.set_num_threads(1);rows=[];updates=0
    seeds=[None]+list(range(20261109,20261114)) if manual else [None]
    budget=64 if manual else 4
    for case,sample in f.CASE_SOURCES.items():
        batch,source=f.load_source(sample);contract=f.native_contract(batch,case) if case.startswith("native") else None
        controls,aliases=f.canonical_controls(batch.state)
        for seed in seeds:
            for control in controls:
                guard(phase,manual);name=case+"__"+str(seed)+"__"+control
                trial=dest/name;trial.mkdir()
                u=f.initial(control,batch.state,seed).requires_grad_();which=f.arm(control);obj=f.p1.objective(which)
                opt=torch.optim.AdamW([u],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
                write(trial/"IDENTITY.json",dict(case=case,source=source,control=control,arm=which,seed=seed,initial_u=u.detach().tolist(),aliases=aliases,network_loaded=False,qualification_witness_consumed=False))
                first_hard=None;first_task=None;trial_started=time.monotonic();initial_hard=None;initial_task=None
                with (trial/"STEPS.jsonl").open("x") as log:
                    for step in range(budget+1):
                        r=obj(f.p1.raw_slots(u),batch);g=torch.autograd.grad(r.loss,u)[0]
                        if not torch.isfinite(g).all() or not torch.isfinite(r.loss):raise ValueError("NONFINITE")
                        rec=f.p1.summarize(r,batch,u,g,arm=which)
                        task=f.crossing_success(r,batch,contract,rec["hard"]) if contract else dict(OriginalHardAll=rec["hard"]["passed"],TaskSuccess=rec["hard"]["passed"],successful_channels=[])
                        J=torch.autograd.functional.jacobian(lambda z:f.actual_theta(z,batch.state,which),u).detach().double()
                        sv=torch.linalg.svdvals(J)
                        rec.update(step=step,Task=task,control=control,seed=seed,theta_Jacobian_singular_values=sv.tolist(),theta_Jacobian_rank=int((sv>1e-7).sum()),theta_Jacobian_condition=float(sv[0]/sv[-1]) if float(sv[-1])>1e-7 else None)
                        if step==0:initial_hard=rec["hard"]["passed"];initial_task=task["TaskSuccess"]
                        if first_hard is None and rec["hard"]["passed"]:first_hard=step
                        if first_task is None and task["TaskSuccess"]:first_task=step
                        if step in (0,8,16,32,64,budget):write(trial/("STEP_"+str(step)+".json"),rec)
                        if step<budget:
                            opt.zero_grad(set_to_none=True);u.grad=g
                            norm=torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True);before=u.detach().clone();opt.step();updates+=1
                            rec.update(preclip_norm=float(norm),update_norm_FP64=float((u.detach()-before).double().norm()))
                        log.write(json.dumps(rec,allow_nan=False)+"\n");log.flush()
                row=dict(case=case,sample_id=sample,control=control,seed=seed,steps=budget,elapsed_s=time.monotonic()-trial_started,initial_Hard=initial_hard,initial_TaskSuccess=initial_task,ever_Hard=first_hard is not None,ever_TaskSuccess=first_task is not None,final_Hard=rec["hard"]["passed"],final_TaskSuccess=rec["Task"]["TaskSuccess"],first_Hard_step=first_hard,first_TaskSuccess_step=first_task,final_loss=rec["loss"],channels=rec["Task"]["successful_channels"],aliases=aliases)
                rows.append(row);write(trial/"RESULT.json",row);print(json.dumps(row),flush=True)
    write(dest/"PAIRED_ROWS.json",rows)
    write(dest/"RESULT.json",dict(status="PASS",trials=len(rows),updates=updates,network_updates=0,main_manual=manual,steps_per_trial=budget))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["tests","gate_a","gate_b","smoke","main"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--manual-main",action="store_true");ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    if a.stage=="main" and not a.manual_main:raise ValueError("USER_MANUAL_COMMAND_REQUIRED")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id;sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase,a.stage=="main")
        try:
            if a.stage=="tests":worker_tests(dest)
            elif a.stage=="gate_a":worker_a(dest)
            elif a.stage=="gate_b":worker_b(dest)
            else:worker_opt(dest,a.stage=="main",a.phase)
        except BaseException:
            write(dest/"WORKER_FAILURE.json",dict(error=traceback.format_exc()));raise
        return
    dest.mkdir()
    write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    write(dest/"SOURCE_SNAPSHOT.json",{str(p.relative_to(BASE)):dict(sha256=sha(p),content=p.read_text()) for p in EXP.glob("*") if p.suffix in (".py",".md")})
    try:
        guard(a.phase,a.stage=="main")
        if sys.executable!=PYTHON:raise ValueError("PYTHON_IDENTITY")
        if a.stage in ("smoke","main"):
            for stage in ("tests","gate_a","gate_b"):
                if read(OUT/"runs"/("acceptance_"+stage+"_001")/"RESULT.json")["status"]!="PASS":raise ValueError("GATES_REQUIRED "+stage)
        spec=importlib.util.spec_from_file_location("_accepted_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(dest/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]+(["--manual-main"] if a.manual_main else [])
        stop=dest/"DEV_STOP.json" if a.phase=="dev" or a.stage=="main" else FREEZE/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,14400 if a.stage=="main" else 600)
        guard(a.phase,a.stage=="main");write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc();(dest/"ENTRY_stderr.log").open("x").write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        stop=dest/"DEV_STOP.json" if a.phase=="dev" or a.stage=="main" else FREEZE/"STOP.json"
        if not stop.exists():write(stop,dict(error=error))
        raise
if __name__=="__main__":main()
