"""Stdlib-only parent launches the full qualification worker under RSS supervision."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r2c_audit_v1"
PREP=OUT/"_preparation/review_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2);f.write("\n")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("tests","audit"));args=parser.parse_args()
    receipt=OUT/"audit/review_001"/("entry_"+args.stage+"_001");receipt.mkdir()
    write(receipt/"COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),stage=args.stage))
    code=1;error=None;start=time.monotonic()
    with (receipt/"stdout.log").open("x") as stdout,(receipt/"stderr.log").open("x") as stderr:
        try:
            if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise ValueError("FIXED_INTERPRETER")
            sys.path.insert(0,str(BASE))
            from query_minco.training.physics_only_v1_2_gate1.e1_r2c_supervision import supervised_worker,guard_inputs
            guard=lambda:guard_inputs(OUT)
            guard()
            dest=OUT/"runs"/("tests_001" if args.stage=="tests" else "audit_001")
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(BASE),CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(receipt/"tmp"))
            (receipt/"tmp").mkdir();env.pop("PYTEST_ADDOPTS",None)
            if args.stage=="tests":
                dest.mkdir(parents=True)
                tests=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),
                    "--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes",
                    "tests/test_physics_v12_e1_reachability_v2.py","tests/test_physics_v12_e1_r2c_supervision.py",
                    "tests/test_physics_v12_e1_dataset.py","tests/test_physics_v12_e1_supervisor.py",
                    "tests/test_physics_v12_gate1_geometry.py","tests/test_physics_v12_gate1_sdf_nonsmooth.py",
                    "query_minco/training/physics_only_v1_2/tests","query_minco/training/physics_only_v1/tests"]
                script="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main("+repr(tests)+"))"
                cmd=[PYTHON,"-B","-c",script];limit=300
            else:
                if json.loads((PREP/"TESTS_PASS.json").read_text())["status"]!="PASS":raise ValueError("TESTS_REQUIRED")
                cmd=[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_r2c_audit","--worker","audit"];limit=900
            # Audit worker creates its own exclusive Run; outer receipts live under audit.
            supervised_worker(cmd,receipt/"worker",cwd=BASE,env=env,stop_path=PREP/"STOP.json",
                budget_root=OUT,guard=guard,timeout=limit)
            guard()
            if args.stage=="tests":write(PREP/"TESTS_PASS.json",dict(status="PASS"))
            else:write(PREP/"AUDIT_PASS.json",dict(status="PASS"))
            code=0;stdout.write("PASS "+args.stage+"\n")
        except BaseException:
            error=traceback.format_exc();stderr.write(error);stderr.flush()
            try:write(PREP/"STOP.json",dict(stage=args.stage,reason="STAGE_FAILURE",traceback=error))
            except FileExistsError:pass
        finally:write(receipt/"PROCESS_EXIT.json",dict(exit_code=code,error=error,elapsed_s=time.monotonic()-start))
    if code:raise SystemExit(code)
if __name__=="__main__":main()
