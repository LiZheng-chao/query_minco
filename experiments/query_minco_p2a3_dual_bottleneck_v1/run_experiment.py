"""P2-A3 fixed-interpreter audit runner; Slit/Portal optimization manual only."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,csv
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
EXP=BASE/"experiments/query_minco_p2a3_dual_bottleneck_v1"
OUT=BASE/"research_outputs/query_minco_p2a3_dual_bottleneck_v1"
AF=OUT/"protocol/audit_freeze_001";BF=OUT/"protocol/probes_freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as stream:json.dump(x,stream,indent=2,allow_nan=False);stream.write("\n")
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open("x",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def guard(phase,stage):
    if sys.executable!=PYTHON:raise ValueError("PYTHON_IDENTITY")
    e=read(OUT/"protocol/entry_001/ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("PROTECTED_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for k,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][k]:raise ValueError("GIT_DRIFT "+label)
    if stage not in ("slit","portal") and time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
    if phase=="formal":
        if (AF/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
        freezes=[AF]+([BF] if stage in ("probe_tests","initial_audit","slit","portal") else [])
        for root in freezes:
            for p,h in read(root/"SOURCE_FREEZE.json").items():
                if sha(BASE/p)!=h:raise ValueError("SOURCE_DRIFT "+p)
            for p,h in read(root/"INPUT_HASHES.json").items():
                if sha(p)!=h:raise ValueError("INPUT_DRIFT "+p)
def worker_tests(dest,probes=False):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    paths=[str(EXP/"test_audit.py"),"experiments/query_minco_p2a_quality_diagnosis_v1/test_quality.py"]
    if probes:paths+=[str(EXP/"test_probes.py")]
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"tmp_tests"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),*paths]
    write(dest/"PYTEST_COMMAND.json",dict(args=args))
    class Collector:
        def pytest_collection_finish(self,session):write(dest/"COLLECTION.json",dict(selected=len(session.items),nodeids=[x.nodeid for x in session.items]))
    code=int(pytest.main(args,plugins=[Collector()]))
    write(dest/"RESULT.json",dict(status="PASS" if code==0 else "STOP",exit_code=code,optimizer_updates=0,network_updates=0))
    if code:raise SystemExit(code)
def worker_audit(dest):
    import torch
    from experiments.query_minco_p2a3_dual_bottleneck_v1.audit import collect,evaluate_selected
    torch.set_num_threads(1);data,states=collect();baselines,validations=evaluate_selected(data)
    csvout(dest/"ALL_STATES.csv",states);csvout(dest/"P2A3_CHECKPOINT_BASELINES.csv",baselines)
    write(dest/"BASELINES.json",baselines);write(dest/"FRESH_VALIDATIONS.json",validations)
    stats=[]
    for case in ("free_near","native_slit_cross_far_v1","native_two_portals_cross_far_v1"):
        for arm in ("L0","L1","L2"):
            ds=[v["rows"] for k,v in data.items() if k[0]==case and k[2]==arm]
            stats.append(dict(case=case,arm=arm,trials=len(ds),first_TaskSuccess=[next((r["step"] for r in rows if r["Task"]["TaskSuccess"]),None) for rows in ds],
                Hard32=sum(r[32]["hard"]["passed"] for r in ds),Task32=sum(r[32]["Task"]["TaskSuccess"] for r in ds),Hard64=sum(r[64]["hard"]["passed"] for r in ds),Task64=sum(r[64]["Task"]["TaskSuccess"] for r in ds),rejects=sum(not r["accepted"] for rows in ds for r in rows[:64]),final_reasons=[r[64]["hard"]["reasons"] for r in ds],mean_S64=sum(r[64]["metrics"]["S"] for r in ds)/len(ds)))
    write(dest/"MAIN_COUNTS.json",stats)
    lateral=[]
    for arm in ("L0","L1","L2"):
        rows=data[("native_two_portals_cross_far_v1",None,arm)]["rows"];ix=[1,4,7,10,13]
        lateral.append(dict(arm=arm,states=65,max_abs_u_y=max(abs(r["u"][i]) for r in rows for i in ix),max_abs_gradient_y=max(abs(r["gradient"][i]) for r in rows for i in ix),max_abs_endpoint_y=max(abs(r["endpoint"][1]) for r in rows),Hard=sum(r["hard"]["passed"] for r in rows),TaskSuccess=sum(r["Task"]["TaskSuccess"] for r in rows)))
    write(dest/"PORTAL_SYMMETRY.json",lateral)
    write(dest/"RESULT.json",dict(status="PASS",source_trials=54,states=3510,checkpoint_rows=216,final_rows=54,quality_rows=216,baselines=len(baselines),fresh_unique_objective_Hard_API_calls=len(validations),logical_selected_counter_pairs=len(baselines),selection_extra_Hard_calls=0,historical_Hard_APIs_per_source_trial=129,fresh_calls_by_source_arm={arm:sum(r["key"][2]==arm for r in validations) for arm in ("L1","L2")},optimizer_updates=0,network_updates=0))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["tests","audit","probe_tests","initial_audit","slit","portal"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--worker",action="store_true");ap.add_argument("--manual",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    if a.stage in ("slit","portal") and (not a.manual or a.phase!="formal"):raise ValueError("MANUAL_OPTIMIZATION_ONLY")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id;sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase,a.stage)
        if a.stage in ("tests","probe_tests"):worker_tests(dest,a.stage=="probe_tests")
        elif a.stage=="audit":worker_audit(dest)
        else:
            from experiments.query_minco_p2a3_dual_bottleneck_v1 import probes
            if a.stage=="initial_audit":probes.initial_audit(dest,write,csvout)
            else:probes.optimize(a.stage,dest,write,csvout,lambda:guard(a.phase,a.stage))
        return
    dest.mkdir();write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    write(dest/"SOURCE_SNAPSHOT.json",{str(p.relative_to(BASE)):dict(sha256=sha(p),content=p.read_text()) for p in EXP.glob("*") if p.suffix in (".py",".md")})
    try:
        guard(a.phase,a.stage)
        need={"audit":["tests"],"probe_tests":["audit"],"initial_audit":["audit","probe_tests"],"slit":["audit","probe_tests","initial_audit"],"portal":["audit","probe_tests","initial_audit"]}.get(a.stage,[])
        for stage in need:
            if read(OUT/"runs"/("acceptance_"+stage+"_001")/"RESULT.json")["status"]!="PASS":raise ValueError("GATE_REQUIRED "+stage)
        from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
        write(dest/"PARENT_DEPENDENCIES.json",bootstrap())
        spec=importlib.util.spec_from_file_location("_accepted_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(dest/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]+(["--manual"] if a.manual else [])
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else AF/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,14400 if a.stage in ("slit","portal") else 600)
        guard(a.phase,a.stage);write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc();(dest/"ENTRY_stderr.log").open("x").write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else AF/"STOP.json"
        if not stop.exists():write(stop,dict(status="STOP",stage=a.stage,error=error))
        raise
if __name__=="__main__":main()
