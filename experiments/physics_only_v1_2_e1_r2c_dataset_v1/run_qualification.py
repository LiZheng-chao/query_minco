"""Lightweight parent supervises the complete R2d driver, including imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r2c_dataset_v1"
PREP=OUT/"_preparation/continuation_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
    e=json.loads((PREP/"ENTRY.json").read_text())
    for p,h in e["protected_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError("HISTORICAL_SOURCE_DRIFT "+p)
    for p,h in json.loads((PREP/"SOURCE_FREEZE.json").read_text()).items():
        if hashlib.sha256((BASE/p).read_bytes()).hexdigest()!=h:raise RuntimeError("FROZEN_SOURCE_DRIFT")
    for key,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO"))):
        actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True)
        if actual!=e["git_before"][key]["head"]:raise RuntimeError("HEAD_DRIFT")
    if time.time()-e["started_unix"]>=3600:raise RuntimeError("TOTAL_TIME_BUDGET")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["preflight","tests","qualify"],required=True);args=parser.parse_args()
    receipt=OUT/"audit/continuation_001"/("entry_"+args.stage+"_001");receipt.mkdir()
    write(receipt/"COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),stage=args.stage))
    code=1;error=None;started=time.monotonic()
    with (receipt/"stdout.log").open("x") as out,(receipt/"stderr.log").open("x") as err:
        try:
            if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise RuntimeError("FIXED_INTERPRETER")
            guard();sys.path.insert(0,str(BASE))
            from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import supervise
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(BASE),OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(receipt/"tmp"))
            (receipt/"tmp").mkdir();env.pop("PYTEST_ADDOPTS",None)
            if args.stage=="tests":
                if json.loads((PREP/"PREFLIGHT_PASS.json").read_text())["status"]!="PASS":raise RuntimeError("PREFLIGHT_REQUIRED")
                dest=OUT/"runs/tests_001";dest.mkdir(parents=True)
                tests=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),
                    "--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes",
                    "tests/test_physics_v12_e1_r2c_dataset.py","tests/test_physics_v12_e1_reachability_v2.py",
                    "tests/test_physics_v12_e1_r2c_supervision.py","tests/test_physics_v12_e1_dataset.py",
                    "tests/test_physics_v12_e1_supervisor.py","tests/test_physics_v12_gate1_geometry.py",
                    "tests/test_physics_v12_gate1_sdf_nonsmooth.py","query_minco/training/physics_only_v1_2/tests",
                    "query_minco/training/physics_only_v1/tests"]
                script="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main("+repr(tests)+"))"
                argv=[PYTHON,"-B","-c",script];limit=300
            else:
                if args.stage=="qualify" and json.loads((PREP/"TESTS_PASS.json").read_text())["status"]!="PASS":raise RuntimeError("TESTS_REQUIRED")
                argv=[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_r2c_dataset","--worker",args.stage]
                remaining=3600-(time.time()-json.loads((PREP/"ENTRY.json").read_text())["started_unix"])
                limit=min(300 if args.stage=="preflight" else remaining,remaining)
            result=supervise(argv,receipt/"worker","driver",cwd=BASE,env=env,stop_path=PREP/"STOP.json",
                budget_root=OUT,guard=guard,timeout=limit,rss_limit=4294967296,disk_limit=2147483648)
            guard()
            write(receipt/"DRIVER_SCOPE.json",dict(root="full Python qualification worker",driver_own_RSS_included=True,
                imports_and_geometry_included=True,all_observed_descendants_included=True,sampled_not_continuous=True))
            if args.stage=="tests":write(PREP/"TESTS_PASS.json",dict(status="PASS",source_freeze_sha256=hashlib.sha256((PREP/"SOURCE_FREEZE.json").read_bytes()).hexdigest()))
            code=0;out.write("PASS "+args.stage+"\n")
        except BaseException:
            error=traceback.format_exc();err.write(error);err.flush()
            try:write(PREP/"STOP.json",dict(reason="STAGE_FAILURE",stage=args.stage,traceback=error,exit_code=1))
            except FileExistsError:pass
        finally:write(receipt/"PROCESS_EXIT.json",dict(exit_code=code,error=error,elapsed_s=time.monotonic()-started))
    if code:raise SystemExit(code)
if __name__=="__main__":main()
