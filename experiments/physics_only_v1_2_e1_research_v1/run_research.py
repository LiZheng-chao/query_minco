"""Thin exclusive E1 read-only launch with receipts and resource budgets."""
import argparse
import hashlib
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
from query_minco.training.physics_only_v1_2_gate1.e1_research import OUT,write,protect

PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("diagnostic","checkpoint","tests"))
    parser.add_argument("--review-entry",choices=("entry_001","r1_001"),default="entry_001")
    parser.add_argument("--case",choices=("checkpoint_gradients_002","tests_002"))
    args=parser.parse_args()
    if args.case and (args.stage,args.case) not in (
        ("checkpoint","checkpoint_gradients_002"),("tests","tests_002")):
        raise ValueError("RUN_STAGE_CASE_MISMATCH")
    if args.review_entry=="r1_001" and not args.case:raise ValueError("FRESH_R1_CASE_REQUIRED")
    os.environ["QUERY_MINCO_E1_REVIEW_ENTRY"]=args.review_entry
    protect()
    case=dict(diagnostic="diagnostic_001",checkpoint="checkpoint_gradients_001",tests="tests_001")[args.stage]
    case=args.case or case
    dest=OUT/"runs"/case;dest.parent.mkdir(exist_ok=True);dest.mkdir();tmp=dest/"tmp";tmp.mkdir()
    if args.stage=="tests":
        tests=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),
               "--junitxml="+str(dest/"TEST_RESULTS.xml"),"tests/test_physics_v12_e1_research.py",
               "query_minco/training/physics_only_v1_2/tests","query_minco/training/physics_only_v1/tests"]
        code="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main("+repr(tests)+"))"
        argv=[PYTHON,"-B","-c",code]
    else:argv=[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_research","--stage",args.stage,"--run",str(dest)]
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(tmp),TEMP=str(tmp),TMP=str(tmp))
    env.pop("PYTEST_ADDOPTS",None)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
      ROOT/"query_minco/training/physics_only_v1_2_gate1/e1_research.py",Path(__file__).resolve(),ROOT/"tests/test_physics_v12_e1_research.py"]}
    write(dest/"COMMAND.json",dict(argv=argv,cwd=str(ROOT),env={k:env[k] for k in (
        "PYTHONPATH","PYTHONDONTWRITEBYTECODE","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","TMPDIR","QUERY_MINCO_E1_REVIEW_ENTRY")},
        source_sha256=source,timeout_seconds=300,RSS_bytes=4*1024**3,output_bytes=128*1024**2))
    sp=importlib.util.spec_from_file_location("budget",ROOT/"experiments/physics_only_v1_2_math_contract_v1/run_math_acceptance.py")
    helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
    child=None;code=None;stop=None;error=None;peak=0;started=time.monotonic()
    with (dest/"stdout.log").open("xb") as stdout,(dest/"stderr.log").open("xb") as stderr:
        try:
            child=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,helper.process_tree_rss(child.pid))
                if time.monotonic()-started>300:stop="TIME_BUDGET"
                if peak>4*1024**3:stop="RSS_BUDGET"
                if helper.disk_bytes(OUT)>128*1024**2:stop="OUTPUT_BUDGET"
                if stop:os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=child.wait()
        except BaseException:
            error=traceback.format_exc()
            if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    receipt=dict(exit_code=code,stop=stop,supervisor_error=error,elapsed_s=time.monotonic()-started,
                 sampled_peak_RSS_bytes=peak,optimizer_constructed=False,optimizer_updates=0,new_data_generated=0)
    write(dest/"PROCESS_EXIT.json",receipt);print(json.dumps(dict(case=case,**receipt)),flush=True)
    if code!=0 or stop or error:raise RuntimeError("E1_STOP "+case)
    protect()

if __name__=="__main__":main()
