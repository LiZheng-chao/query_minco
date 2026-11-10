"""P0 design precheck launcher; no training proxy or optimizer is enabled."""
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_p0_roi_proxy_v1"
PREP=OUT/"_preparation/freeze_001"
RUN=OUT/"runs/math_precheck_seed20261104_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()},**read(PREP/"INPUT_HASHES.json")}.items():
        if sha(p)!=h:raise RuntimeError("SOURCE_OR_DATA_SHA_DRIFT "+p)
    if time.time()-e["started_unix"]>=3600:raise RuntimeError("TOTAL_BUDGET")
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for key,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][key]:
                raise RuntimeError("GIT_DRIFT "+label+" "+key)
def load_helpers():
    p=BASE/"tests/test_physics_v12_p0_roi_proxy.py"
    s=importlib.util.spec_from_file_location("_p0_design_counterexample",p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inventory():
    frozen=read(PREP/"TRAIN13_AND_EXCLUSIONS.json")
    manifest=read(BASE/"research_outputs/physics_only_v1_2_e1_r2c_dataset_v1/data/DATASET_MANIFEST.json")
    rows=frozen["all28"]
    if len(rows)!=28 or [r["sample_id"] for r in rows]!=[r["sample_id"] for r in manifest["samples"]]:
        raise RuntimeError("ALL28_IDENTITY_OR_ORDER")
    summary=[];verified={}
    for r in rows:
        for field,hfield in (("source_manifest_path","source_manifest_sha256"),("hard_evidence_path","hard_evidence_sha256")):
            if r.get(field):
                if sha(r[field])!=r[hfield]:raise RuntimeError("TASK_IDENTITY "+r["sample_id"])
                verified[r[field]]=r[hfield]
        m=read(r["source_manifest_path"])
        for p,h in m["files_sha256"].items():
            if p not in verified:
                if sha(p)!=h:raise RuntimeError("BOUND_INPUT_SHA "+p)
                verified[p]=h
        eligible=r["source"]=="SOURCE_QUALIFIED" and r["reachability"]=="FEASIBLE_CERTIFIED" and r["data_eligible"]
        summary.append(dict(sample_id=r["sample_id"],scene_id=r["scene_id"],split=r["split"],
            source=r["source"],reachability=r["reachability"],data_eligible=r["data_eligible"],
            source_manifest_path=r["source_manifest_path"],source_manifest_sha256=r["source_manifest_sha256"],
            geometry_sha256=r.get("geometry_sha256"),camera_sha256=r.get("camera_sha256"),state_sha256=r.get("state_sha256"),
            hard_evidence_path=r.get("hard_evidence_path"),hard_evidence_sha256=r.get("hard_evidence_sha256"),
            P0_source_byte_audit="PASS",qualification_origin="frozen R2d; not recertified",
            allowed_action="Train zero-update only after all P0 design gates" if eligible and r["split"]=="train" else "Dev read-only; no parameter selection" if eligible else "ISOLATED; not used for training",
            exclusion_reason="" if eligible else r["reachability"]))
    with (OUT/"summary/freeze_001/P0_ALL28_TASK_INVENTORY.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    write(RUN/"ALL28_SOURCE_HASHES.json",verified)
    counts={split:{s:sum(r["split"]==split and r["reachability"]==s for r in rows) for s in ("FEASIBLE_CERTIFIED","REACHABILITY_UNKNOWN","INFEASIBLE_CERTIFIED")} for split in ("train","dev")}
    if counts!={"train":{"FEASIBLE_CERTIFIED":13,"REACHABILITY_UNKNOWN":2,"INFEASIBLE_CERTIFIED":1},"dev":{"FEASIBLE_CERTIFIED":8,"REACHABILITY_UNKNOWN":2,"INFEASIBLE_CERTIFIED":2}}:
        raise RuntimeError("QUALIFICATION_COUNTS")
    write(RUN/"ALL28_RESULT.json",dict(status="PASS_STATIC_SOURCE_REUSE",counts=counts,total=28,
        source_qualified=sum(r["source"]=="SOURCE_QUALIFIED" for r in rows),new_Hard_certifications=0))
    return frozen

def worker():
    sys.path.insert(0,str(BASE));guard()
    if sys.executable!=PYTHON:raise RuntimeError("INTERPRETER_IDENTITY")
    RUN.mkdir()
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    dependencies=bootstrap();write(RUN/"DEPENDENCIES.json",dependencies)
    frozen=inventory()
    import pytest
    from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import monitor_preflight
    write(RUN/"MONITOR_PREFLIGHT.json",monitor_preflight())
    args=["-q","-p","no:cacheprovider","--basetemp="+str(RUN/"pytest_tmp"),
          "--junitxml="+str(RUN/"TEST_RESULTS.xml"),"tests/test_physics_v12_p0_roi_proxy.py"]
    write(RUN/"PYTEST_COMMAND.json",dict(args=args,scope="counterexample reproduction only"))
    code=int(pytest.main(args));write(RUN/"PYTEST_EXIT.json",dict(exit_code=code))
    if code:raise RuntimeError("COUNTEREXAMPLE_TESTS_FAILED")
    helpers=load_helpers()
    syn=helpers.synthetic_counterexample();write(RUN/"SYNTHETIC_BOUNDARY_COUNTEREXAMPLE.json",syn)
    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
    row=frozen["Train13"][0]
    batch=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"])
    real=helpers.real_floor_counterexample(batch);real["source_identity"]=row
    write(RUN/"REAL_SOURCE_FLOOR_COUNTEREXAMPLE.json",real)
    if not (syn["inside_C_viol"]>syn["outside_C_viol"] and real["inside_C_viol"]>real["outside_C_viol"] and real["strict_outside_error"]=="UNKNOWN_OUTSIDE_CERTIFIED_ROI"):
        raise RuntimeError("EXPECTED_DESIGN_COUNTEREXAMPLE_NOT_REPRODUCED")
    write(RUN/"RESULT.json",dict(status="STOP_MATH_CONTRACT",exit_code=2,
        reason="ROI-only distance reset lowers Collision cost across UNKNOWN and cannot continuously match arbitrary known boundary traces",
        synthetic=syn,real_source=real,qualified_proxy=False,model_forwards=0,Train13_completed=0,
        optimizer_updates=0,Hard_new_results=0,legacy_regressions="NOT_RUN_AFTER_MATH_STOP"))
    print(json.dumps(dict(status="STOP_MATH_CONTRACT",synthetic_inside_C=syn["inside_C_viol"],synthetic_outside_C=syn["outside_C_viol"],
        real_inside_C=real["inside_C_viol"],real_outside_C=real["outside_C_viol"])),flush=True)
    raise SystemExit(2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if a.worker:
        try:worker()
        except SystemExit:raise
        except BaseException:
            if RUN.exists() and not (RUN/"FAILURE.json").exists():write(RUN/"FAILURE.json",dict(status="STOP",error=traceback.format_exc()))
            raise
        return
    folder=OUT/"audit/freeze_001/entry_math_precheck_001";folder.mkdir()
    write(folder/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    sys.path.insert(0,str(BASE))
    try:
        guard()
        if sys.executable!=PYTHON:raise RuntimeError("FIXED_INTERPRETER")
        s=importlib.util.spec_from_file_location("_accepted_complete_tree_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(folder/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(folder/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--worker"]
        old.launch(cmd,folder/"driver",env,PREP/"STOP.json",OUT,min(300,3600-(time.time()-read(PREP/"ENTRY.json")["started_unix"])))
        raise RuntimeError("DESIGN_DIAGNOSTIC_CANNOT_AUTHORIZE_TRAINING")
    except BaseException:
        error=traceback.format_exc()
        with (folder/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(folder/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        if not (PREP/"STOP.json").exists():write(PREP/"STOP.json",dict(status="STOP",error=error))
        raise
if __name__=="__main__":main()
