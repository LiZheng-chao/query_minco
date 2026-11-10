"""Thin launcher: immutable receipts, CPU resource supervision, no training by default."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from query_minco.training.physics_only_v1_2_gate1.short_train import OUT,check_authorization,write,sha

PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"


def supervised(case,argv,timeout=300,disk_limit=128*1024**2):
    dest=OUT/case;dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir();tmp=dest/"tmp";tmp.mkdir()
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(tmp),TEMP=str(tmp),TMP=str(tmp))
    env.pop("PYTEST_ADDOPTS",None)
    sp=importlib.util.spec_from_file_location("budget_helpers",ROOT/"experiments/physics_only_v1_2_math_contract_v1/run_math_acceptance.py")
    helpers=importlib.util.module_from_spec(sp);sp.loader.exec_module(helpers)
    write(dest/"COMMAND.json",dict(argv=argv,cwd=str(ROOT),env={k:env[k] for k in (
        "PYTHONPATH","CUDA_VISIBLE_DEVICES","PYTHONDONTWRITEBYTECODE","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","TMPDIR")},
        timeout_seconds=timeout,rss_limit_bytes=4*1024**3,disk_limit_bytes=disk_limit))
    child=None;peak=0;stop=None;error=None;code=None;start=time.monotonic()
    with (dest/"stdout.log").open("xb") as stdout,(dest/"stderr.log").open("xb") as stderr:
        try:
            child=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,helpers.process_tree_rss(child.pid))
                if time.monotonic()-start>timeout:stop="TIME_BUDGET"
                if peak>4*1024**3:stop="RSS_BUDGET"
                if helpers.disk_bytes(OUT)>disk_limit:stop="DISK_BUDGET"
                if stop:os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=child.wait()
        except BaseException:
            error=traceback.format_exc()
            if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    write(dest/"PROCESS_EXIT.json",dict(exit_code=code,stop_reason=stop,supervisor_error=error,
        elapsed_seconds=time.monotonic()-start,sampled_peak_tree_RSS_bytes=peak,
        mode="planning/audit" if case.startswith("runs/") else "authorized_training_supervisor"))
    print(json.dumps(dict(case=case,exit_code=code,stop=stop,error=error,peak_RSS_bytes=peak)),flush=True)
    if code!=0 or stop or error:raise RuntimeError("STOP "+case)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("audit","audit-corrected","tests","train"))
    p.add_argument("--spec");p.add_argument("--approval");args=p.parse_args()
    module="query_minco.training.physics_only_v1_2_gate1.short_train"
    if args.stage=="train":
        spec=check_authorization(args.spec,args.approval) # must precede any run or optimizer
        supervised("audit/plan_001/train_launch_001",
            [PYTHON,"-B","-m",module,"--stage","train","--spec",args.spec,"--approval",args.approval],
            timeout=spec["resources"]["wall_seconds"],disk_limit=spec["resources"]["output_bytes"])
    elif args.stage in ("audit","audit-corrected"):
        case="runs/init_audit_001" if args.stage=="audit" else "runs/init_audit_002"
        supervised(case,[PYTHON,"-B","-m",module,"--stage","audit","--output-run",str(OUT/case)])
    else:
        case="runs/plan_tests_001"
        selected=["tests/test_physics_v12_short_train_contract.py",
                  "query_minco/training/physics_only_v1_2/tests",
                  "query_minco/training/physics_only_v1/tests"]
        pytest_args=["-q","-p","no:cacheprovider","--basetemp="+str(OUT/case/"pytest_tmp"),
                     "--junitxml="+str(OUT/case/"TEST_RESULTS.xml"),*selected]
        code="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main("+repr(pytest_args)+"))"
        supervised(case,[PYTHON,"-B","-c",code])
if __name__=="__main__":main()
