"""P0b thin staged diagnostic launcher. No model or production proxy is imported."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_p0b_roi_proxy_v1"
PREP=OUT/"_preparation/p0b_seed20261104_001"
SUMMARY=OUT/"summary/p0b_seed20261104_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()},**read(PREP/"INPUT_HASHES.json")}.items():
        if sha(p)!=h:raise RuntimeError("SHA_DRIFT "+p)
    if time.time()-e["started_unix"]>3600:raise RuntimeError("TOTAL_BUDGET")
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for key,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][key]:raise RuntimeError("GIT_DRIFT")
def helpers():
    s=importlib.util.spec_from_file_location("_p0b_tests",BASE/"tests/test_physics_v12_p0b_roi_proxy.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def inventory(dest):
    frozen=read(PREP/"TRAIN13_AND_EXCLUSIONS.json")
    rows=frozen["all28"];verified={};records=[]
    r2d=read(BASE/"research_outputs/physics_only_v1_2_e1_r2c_dataset_v1/data/DATASET_MANIFEST.json")
    if len(rows)!=28 or [r["sample_id"] for r in rows]!=[r["sample_id"] for r in r2d["samples"]]:raise RuntimeError("DATASET28")
    for r in rows:
        for pathfield,hfield in (("source_manifest_path","source_manifest_sha256"),("hard_evidence_path","hard_evidence_sha256")):
            if r.get(pathfield):
                p=r[pathfield];h=r[hfield]
                if sha(p)!=h:raise RuntimeError("TASK_IDENTITY "+p)
                verified[p]=h
        for p,h in read(r["source_manifest_path"])["files_sha256"].items():
            if p not in verified:
                if sha(p)!=h:raise RuntimeError("DATA_BYTE_DRIFT "+p)
                verified[p]=h
        eligible=r["source"]=="SOURCE_QUALIFIED" and r["reachability"]=="FEASIBLE_CERTIFIED" and r["data_eligible"]
        records.append(dict(sample_id=r["sample_id"],scene_id=r["scene_id"],split=r["split"],source=r["source"],reachability=r["reachability"],data_eligible=r["data_eligible"],
            manifest_path=r["source_manifest_path"],manifest_sha256=r["source_manifest_sha256"],hard_evidence_path=r.get("hard_evidence_path"),hard_evidence_sha256=r.get("hard_evidence_sha256"),
            geometry_sha256=r.get("geometry_sha256"),camera_sha256=r.get("camera_sha256"),state_sha256=r.get("state_sha256"),source_rehash="PASS",
            exclusion_reason="" if eligible else r["reachability"],action="Train zero-update only after prerequisite gates" if eligible and r["split"]=="train" else "Dev readonly only" if eligible else "ISOLATED; unchanged"))
    with (SUMMARY/"P0B_ALL28_TASK_INVENTORY.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(dest/"ALL28_HASHES.json",verified)
    write(dest/"ALL28_RESULT.json",dict(total=28,source_qualified=sum(r["source"]=="SOURCE_QUALIFIED" for r in rows),Train=sum(r["split"]=="train" for r in rows),Dev=sum(r["split"]=="dev" for r in rows),Train_feasible=13,Dev_feasible=8,isolated=7,new_certifications=0,verified_files=len(verified)))
def worker(stage):
    sys.path.insert(0,str(BASE));guard()
    dest=OUT/"runs"/("gate_"+stage+"_seed20261104_001");dest.mkdir()
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap())
    if stage=="a":inventory(dest)
    elif read(OUT/"runs/gate_a_seed20261104_001/RESULT.json")["status"]!="TEST_FIX_PASS_AND_ROI_ONLY_COUNTEREXAMPLE_PASS":
        raise RuntimeError("GATE_A_REQUIRED")
    import pytest
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),
          "tests/test_physics_v12_p0b_roi_proxy.py","-k","test_"+stage.upper()+"_"]
    write(dest/"PYTEST_COMMAND.json",dict(args=args,scope="test fixture" if stage=="a" else "projection math, no production adapter"))
    code=int(pytest.main(args));write(dest/"PYTEST_EXIT.json",dict(exit_code=code))
    if code:
        reason="P0B_TEST_FIX_STOP" if stage=="a" else "P0B_MATH_STOP"
        write(dest/"RESULT.json",dict(status=reason,exit_code=code,model_forwards=0,optimizer_updates=0,production_adapter=False))
        raise SystemExit(code)
    if stage=="a":
        h=helpers()
        from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
        row=read(PREP/"TRAIN13_AND_EXCLUSIONS.json")["Train13"][0]
        b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"])
        write(dest/"COUNTEREXAMPLES.json",dict(synthetic=[h.A_report(n) for n in [2,192]],real_source_floor=[h.floor_report(b,n) for n in [2,192]],source=row))
    write(dest/"RESULT.json",dict(status="TEST_FIX_PASS_AND_ROI_ONLY_COUNTEREXAMPLE_PASS" if stage=="a" else "MATH_TESTS_PASS_REQUIRES_INJECTION_REVIEW",model_forwards=0,optimizer_updates=0,production_adapter=False))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["a","b"],required=True);ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if a.worker:
        try:worker(a.stage)
        except SystemExit:raise
        except BaseException:
            dest=OUT/"runs"/("gate_"+a.stage+"_seed20261104_001")
            if dest.exists():write(dest/"FAILURE.json",dict(status="STOP",error=traceback.format_exc()))
            raise
        return
    folder=OUT/"audit/p0b_seed20261104_001"/("entry_gate_"+a.stage+"_001");folder.mkdir()
    write(folder/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    sys.path.insert(0,str(BASE))
    try:
        guard()
        if sys.executable!=PYTHON:raise RuntimeError("INTERPRETER_IDENTITY")
        if a.stage=="b" and read(OUT/"runs/gate_a_seed20261104_001/RESULT.json")["status"]!="TEST_FIX_PASS_AND_ROI_ONLY_COUNTEREXAMPLE_PASS":raise RuntimeError("GATE_A_REQUIRED")
        s=importlib.util.spec_from_file_location("_accepted_full_tree_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(folder/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(folder/"tmp").mkdir()
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--stage",a.stage,"--worker"]
        m.launch(cmd,folder/"driver",env,PREP/"STOP.json",OUT,min(300,3600-(time.time()-read(PREP/"ENTRY.json")["started_unix"])))
        guard();write(folder/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc()
        with (folder/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(folder/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        if not (PREP/"STOP.json").exists():write(PREP/"STOP.json",dict(status="STOP",error=error))
        raise
if __name__=="__main__":main()
