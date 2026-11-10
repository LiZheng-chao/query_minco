"""R2b thin entry: early receipts, explicit stages, no training or retries."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_dataset_v2"
PREP=OUT/"_preparation/qualification_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def write(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("preflight","tests","qualify"));args=parser.parse_args()
    folder=OUT/"audit/qualification_001"/("entry_"+args.stage+"_001");folder.mkdir()
    write(folder/"COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),stage=args.stage,identity="e1_r2b_data_qualification_v1"))
    code=1;error=None;q=None;started=time.monotonic()
    with (folder/"stdout.log").open("x") as stdout,(folder/"stderr.log").open("x") as stderr:
        try:
            if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
            if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise RuntimeError("FIXED_INTERPRETER")
            os.environ["QUERY_MINCO_E1_DATA_REVIEW"]="r2b_001";os.environ["QUERY_MINCO_E1_DATA_STAGE"]=args.stage
            os.environ["PYTHONDONTWRITEBYTECODE"]="1";sys.path.insert(0,str(BASE))
            from query_minco.training.physics_only_v1_2_gate1 import e1_dataset_qualification as q
            q.protect()
            if not (PREP/"SOURCE_FREEZE.json").is_file():raise RuntimeError("NEW_SOURCE_FREEZE_REQUIRED")
            if args.stage=="preflight":
                run=OUT/"runs/build_001";run.mkdir(parents=True)
                script="import sys,importlib,json;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');mods={n:importlib.import_module(n) for n in ['numpy','scipy','torch','pytest','query_minco']};print(json.dumps({n:{'file':m.__file__,'version':getattr(m,'__version__',None)} for n,m in mods.items()},indent=2));from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import monitor_preflight;print(json.dumps(monitor_preflight()))"
                q.command([PYTHON,"-B","-c",script],run,"dependencies")
                approved=q.proposal()["accepted_binary"]
                q.command(["ldd",approved["path"]],run,"ldd")
                if "not found" in (run/"ldd_stdout.log").read_text():raise RuntimeError("DYNAMIC_LIBRARY_MISSING")
                write(run/"BINARY.json",dict(path=approved["path"],sha256=approved["sha256"],reused_unchanged_R2a_binary=True,
                       original_build_receipt=approved,python_driver_source_sha256=q.read(PREP/"SOURCE_FREEZE.json")))
                samples=[q.gpu_sample() for _ in range(3)]
                write(PREP/"GPU_MONITOR_PREFLIGHT.json",dict(status="PASS",samples=samples,under_load="NOT_RUN"))
                q.protect();write(PREP/"PREFLIGHT_PASS.json",dict(status="PASS"))
            elif args.stage=="tests":
                if q.read(PREP/"PREFLIGHT_PASS.json")["status"]!="PASS":raise RuntimeError("PREFLIGHT_REQUIRED")
                q.tests()
                from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import gpu_callback_acceptance
                gpu_callback_acceptance(OUT/"runs/tests_001",PYTHON)
                # Exact retained counterexample classification remains mandatory.
                import xml.etree.ElementTree as E
                cases=list(E.parse(OUT/"runs/tests_001/TEST_RESULTS.xml").iter("testcase"))
                if len(cases)!=194 or any(t.find("failure") is not None or t.find("error") is not None for t in cases):raise RuntimeError("REGRESSION_COUNTS")
                if not any(t.attrib["name"]=="test_original_failure_unchanged_and_reproduced" for t in cases):raise RuntimeError("COUNTEREXAMPLE_CONTRACT_REQUIRED")
                q.protect();write(PREP/"TESTS_PASS.json",dict(status="PASS",passed=194,ordinary_deselected=1,gpu_callback_tests=3))
            else:
                q.require_data_gate()
                q.qualify()
            q.protect();code=0;stdout.write("PASS "+args.stage+"\n")
        except BaseException:
            error=traceback.format_exc();stderr.write(error);stderr.flush()
            try:write(PREP/"STOP.json",dict(reason="STAGE_FAILURE",stage=args.stage,exit_code=1,traceback=error))
            except FileExistsError:pass
        finally:
            write(folder/"PROCESS_EXIT.json",dict(exit_code=code,elapsed_s=time.monotonic()-started,error=error,stage=args.stage))
    if code:raise SystemExit(code)
if __name__=="__main__":main()
