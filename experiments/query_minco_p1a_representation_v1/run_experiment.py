"""P1-A bounded offline direct-u experiments; never network training."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/query_minco_p1a_representation_v1"
EXP=BASE/"experiments/query_minco_p1a_representation_v1"
ENTRY=OUT/"protocol/entry_001"
FREEZE=OUT/"protocol/freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
CODE=[str(p.relative_to(BASE)) for p in EXP.glob("*") if p.suffix in (".py",".md")]
P0=BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard(phase,external=False):
    e=read(ENTRY/"ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("PROTECTED_DRIFT "+p)
    for p,h in read(P0/"_preparation/freeze_p0c_001/INPUT_HASHES.json").items():
        if sha(p)!=h:raise ValueError("P0C_INPUT_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for k,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][k]:raise ValueError("GIT_DRIFT")
    if not external and time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
    if phase=="formal":
        if (FREEZE/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
        for p,h in read(FREEZE/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise ValueError("SOURCE_DRIFT "+p)
def bindings():
    return read(P0/"_preparation/bindings_p0c_001/BINDING_INDEX.json")
def load(row):
    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
    return load_e1_bound(row["p0c_manifest_path"],row["p0c_manifest_sha256"])
def worker_tests(dest):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"tmp_tests"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),str(EXP/"test_representation.py")]
    write(dest/"PYTEST_COMMAND.json",dict(args=args))
    code=int(pytest.main(args));write(dest/"RESULT.json",dict(status="PASS" if code==0 else "FAIL",pytest_exit=code,network_updates=0))
    if code:raise SystemExit(code)
def worker_audit(dest):
    import numpy as np,torch
    from experiments.query_minco_p1a_representation_v1.representation import (ARMS,PILOT,D_VALUES,bias,theta_math,physical_theta,PrototypeDecoder,raw_slots,samples_vector,independent_samples,objective,reference)
    torch.set_num_threads(1)
    def spectrum(J):
        s=np.linalg.svd(J,compute_uv=False);rank=int((s>1e-7).sum())
        return dict(rank=rank,singular_values=s.tolist(),condition=float(s[0]/s[-1]) if rank==15 else None,rank_threshold=1e-7)
    items=[]
    for name in PILOT:
        row=next(r for r in bindings() if r["sample_id"]==name);batch=load(row);st=batch.state
        for arm in ARMS:
            u=torch.zeros(15,requires_grad=True)
            Jt=torch.autograd.functional.jacobian(lambda x:physical_theta(PrototypeDecoder(arm)(raw_slots(x),st))[0,0],u).detach().double().numpy()
            Jp=torch.autograd.functional.jacobian(lambda x:samples_vector(x,st,arm),u,vectorize=True).detach().double().numpy()
            h=1e-5;eye=np.eye(15)
            fd=np.stack([(independent_samples(eye[i]*h,st,arm)-independent_samples(-eye[i]*h,st,arm))/(2*h) for i in range(15)],-1)
            if not np.all(np.abs(Jp-fd)<=.003+.003*np.abs(fd)):raise ValueError("PRIMARY_AD_FD")
            hf=.002
            f32=np.stack([((samples_vector(torch.tensor(eye[i]*hf,dtype=torch.float32),st,arm)-samples_vector(torch.tensor(-eye[i]*hf,dtype=torch.float32),st,arm))/(2*hf)).detach().double().numpy() for i in range(15)],-1)
            saturated=[]
            for x in [0.,4.,8.,50.]:
                q=torch.full((15,),x,requires_grad=True)
                jj=torch.autograd.functional.jacobian(lambda z:physical_theta(PrototypeDecoder(arm)(raw_slots(z),st))[0,0],q).detach().double().numpy()
                saturated.append(dict(logit=x,**spectrum(jj)))
            items.append(dict(sample_id=name,arm=arm,theta=physical_theta(PrototypeDecoder(arm)(raw_slots(u),st))[0,0].detach().tolist(),
                theta_units_spectrum=spectrum(Jt),theta_dimensionless_spectrum=spectrum(Jt/np.array(D_VALUES)[:,None]),
                position_curve_spectrum=spectrum(Jp),position_curve_dimensionless_spectrum=spectrum(Jp/6),
                AD_FP32_vs_FD_FP64_max_abs=float(np.abs(Jp-fd).max()),primary_pass=True,
                FP32_h002_failed_entries=int((np.abs(Jp-f32)>(.003+.003*np.abs(f32))).sum()),FP32_h002_max_abs=float(np.abs(Jp-f32).max()),saturation=saturated))
    write(dest/"JACOBIAN_AUDIT.json",items)
    # Matched physical initialization control: two equivalent coordinate origins, not pilot performance.
    batch=load(next(r for r in bindings() if r["sample_id"]==PILOT[0]));st=batch.state
    u0=torch.linspace(-.05,.05,15);u1=(u0.double()+bias(st,"B0")-bias(st,"B1")).float()
    params=[u0.clone().requires_grad_(),u1.clone().requires_grad_()]
    opts=[torch.optim.AdamW([u],lr=.01,weight_decay=0,foreach=False) for u in params]
    objs=[objective(a) for a in ["B0","B1"]];matches=[]
    for step in range(4):
        rs=[o(raw_slots(u),batch) for o,u in zip(objs,params)]
        dt=float((physical_theta(rs[0].parameters)-physical_theta(rs[1].parameters)).abs().max())
        dp=float((samples_vector(params[0],st,"B0")-samples_vector(params[1],st,"B1")).abs().max())
        dl=abs(float(rs[0].loss)-float(rs[1].loss))
        matches.append(dict(step=step,max_theta_difference=dt,max_sample_difference=dp,loss_difference=dl))
        if dt>3e-6 or dp>2e-4 or dl>.003+.003*abs(float(rs[0].loss)):raise ValueError("MATCHED_INIT_CONTROL")
        if step<3:
            for opt,u,r in zip(opts,params,rs):
                opt.zero_grad(set_to_none=True);r.loss.backward()
                torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True);opt.step()
    write(dest/"MATCHED_INITIALIZATION_CONTROL.json",dict(status="PASS",records=matches,trajectory_parameter_updates=6,network_updates=0))
    write(dest/"RESULT.json",dict(status="PASS",Jacobian_cases=len(items),arms=3,effective_dimensions=15,network_updates=0,matched_initialization_control="PASS"))
def worker_pilot(dest,scope,phase):
    import numpy as np,torch
    from unittest.mock import patch
    from experiments.query_minco_p1a_representation_v1.representation import ARMS,PILOT,objective,raw_slots,summarize
    torch.set_num_threads(1)
    selected=[r for r in bindings() if r["sample_id"] in PILOT] if scope=="pilot" else [r for r in bindings() if r["split"]=="train" and r["data_eligible"]]
    selected.sort(key=lambda r:PILOT.index(r["sample_id"]) if scope=="pilot" else [x["sample_id"] for x in bindings()].index(r["sample_id"]))
    summary=[];updates=0;current=None
    try:
        for row in selected:
            guard(phase,external=scope!="pilot");batch=load(row)
            for init in ["nominal","noise"]:
                initvec=torch.zeros(15) if init=="nominal" else torch.randn(15,generator=torch.Generator().manual_seed(20261109))*.05
                for arm in ARMS:
                    current=row["sample_id"]+"__"+init+"__"+arm;trial=dest/current;trial.mkdir()
                    write(trial/"IDENTITY.json",dict(sample_id=row["sample_id"],source=row,arm=arm,initialization=init,seed=20261109,initial_u=initvec.tolist(),effective_dof=15,network_loaded=False))
                    u=initvec.clone().requires_grad_();obj=objective(arm)
                    opt=torch.optim.AdamW([u],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
                    if len(opt.param_groups)!=1 or opt.param_groups[0]["params"]!=[u] or u.numel()!=15:raise ValueError("OPTIMIZER_SCOPE")
                    first=None;initial=None;last=None;start=time.monotonic()
                    with (trial/"STEPS.jsonl").open("x") as log,patch.object(torch,"load",side_effect=RuntimeError("CHECKPOINT_FORBIDDEN")):
                        for step in range(17):
                            result=obj(raw_slots(u),batch)
                            grad=torch.autograd.grad(result.loss,u)[0]
                            if not torch.isfinite(grad).all() or not torch.isfinite(result.loss):raise ValueError("NONFINITE_DIRECT_OPTIMIZATION")
                            rec=summarize(result,batch,u,grad,arm=arm);rec.update(step=step,arm=arm,network_updates=0)
                            if first is None and rec["hard"]["passed"]:first=step
                            if initial is None:initial=rec
                            last=rec
                            if step<16:
                                opt.zero_grad(set_to_none=True);u.grad=grad
                                clipped=torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True)
                                before=u.detach().clone();opt.step();updates+=1
                                rec["preclip_norm"]=float(clipped);rec["update_norm_FP64"]=float((u.detach()-before).double().norm())
                            else:rec["update_executed"]=False
                            log.write(json.dumps(rec,allow_nan=False)+"\n");log.flush()
                    write(trial/"INITIAL.json",initial);write(trial/"FINAL.json",last)
                    line=dict(sample_id=row["sample_id"],scene_id=row["scene_id"],arm=arm,initialization=init,seed=20261109,steps=16,
                        initial_Hard=initial["hard"]["passed"],final_Hard=last["hard"]["passed"],first_Hard_step=first,initial_loss=initial["loss"],final_loss=last["loss"],
                        final_UNKNOWN=last["UNKNOWN_count"],final_known=last["known_count"],final_goal_error=last["hard"]["goal"]["endpoint_goal_error_m"],
                        final_progress=last["hard"]["goal"]["distance_reduction_m"],final_max_v=last["hard"]["dynamic_polynomial_maxima"][0],
                        final_max_a=last["hard"]["dynamic_polynomial_maxima"][1],final_max_j=last["hard"]["dynamic_polynomial_maxima"][2],
                        final_reasons=";".join(last["hard"]["reasons"]),final_path_length=last["path_length_m"],final_sample_clearance=last["known_sample_clearance_m"],
                        final_jerk_quality=last["metrics"].get("S"),final_loss_components=last["metrics"],channel_coverage=last["channel_coverage"],elapsed_s=time.monotonic()-start)
                    summary.append(line);write(trial/"RESULT.json",line)
                    print(json.dumps({k:line[k] for k in ["sample_id","arm","initialization","initial_Hard","final_Hard","first_Hard_step","final_loss"]}),flush=True)
        write(dest/"PILOT_ROWS.json",summary)
        write(dest/"RESULT.json",dict(status="PASS",scope=scope,trials=len(summary),trajectory_parameter_updates=updates,network_updates=0,current_trial=None))
    except BaseException:
        write(dest/"PARTIAL_ROWS.json",summary);write(dest/"RESULT.json",dict(status="STOP",current_trial=current,trials_completed=len(summary),trajectory_parameter_updates=updates,network_updates=0,error=traceback.format_exc()));raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["tests","audit","pilot","expanded"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--manual-expanded",action="store_true");ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    if a.stage=="expanded" and not a.manual_expanded:raise ValueError("EXPLICIT_USER_COMMAND_REQUIRED")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id;sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase,external=a.stage=="expanded")
        try:
            if a.stage=="tests":worker_tests(dest)
            elif a.stage=="audit":worker_audit(dest)
            else:worker_pilot(dest,"pilot" if a.stage=="pilot" else "train13",a.phase)
        except BaseException:
            write(dest/"WORKER_FAILURE.json",dict(error=traceback.format_exc()));raise
        return
    dest.mkdir();write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    write(dest/"SOURCE_SNAPSHOT.json",{p:dict(sha256=sha(BASE/p),content=(BASE/p).read_text()) for p in CODE})
    try:
        guard(a.phase,external=a.stage=="expanded")
        if sys.executable!=PYTHON:raise ValueError("PYTHON_IDENTITY")
        if a.phase=="formal" and a.stage in ("pilot","expanded"):
            for p in ["acceptance_tests_001","acceptance_audit_001"]:
                if read(OUT/"runs"/p/"RESULT.json")["status"]!="PASS":raise ValueError("GATE_A_REQUIRED")
        s=importlib.util.spec_from_file_location("_accepted_full_tree_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(dest/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]+(["--manual-expanded"] if a.manual_expanded else [])
        stop=dest/"DEV_STOP.json" if a.phase=="dev" or a.stage=="expanded" else FREEZE/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,7200 if a.stage=="expanded" else 600)
        guard(a.phase,external=a.stage=="expanded");write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc()
        with (dest/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="FAIL" if a.phase=="dev" else "STOP",exit_code=1,error=error))
        if a.phase=="formal" and a.stage!="expanded" and not (FREEZE/"STOP.json").exists():write(FREEZE/"STOP.json",dict(error=error))
        raise
if __name__=="__main__":main()
