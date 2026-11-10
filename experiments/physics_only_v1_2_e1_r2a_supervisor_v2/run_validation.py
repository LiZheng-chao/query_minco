"""Stdlib bootstrap receipts precede all project/optional imports."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r2a_supervisor_v2"
PREP=OUT/"_preparation/validation_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"

def write(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2);f.write("\n")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("preflight","tests","build"))
    args=parser.parse_args()
    # Every stage gets an exclusive bootstrap receipt even if project import fails.
    folder=OUT/"audit/validation_001"/("entry_"+args.stage+"_001");folder.mkdir()
    write(folder/"COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),stage=args.stage,identity="e1_r2a_supervisor_v2"))
    started=time.monotonic();code=1;error=None
    with (folder/"stdout.log").open("x") as stdout,(folder/"stderr.log").open("x") as stderr:
        try:
            if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
            if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise RuntimeError("FIXED_INTERPRETER_REQUIRED")
            os.environ["QUERY_MINCO_E1_DATA_REVIEW"]="r2a_002"
            os.environ["PYTHONDONTWRITEBYTECODE"]="1"
            sys.path.insert(0,str(BASE))
            from query_minco.training.physics_only_v1_2_gate1 import e1_dataset_qualification as q
            q.protect()
            if not (PREP/"SOURCE_FREEZE.json").is_file():raise RuntimeError("NEW_SOURCE_FREEZE_REQUIRED")
            if args.stage=="preflight":
                run=OUT/"runs/preflight_001";run.mkdir(parents=True)
                script="import sys,json,importlib;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');names=['numpy','scipy','torch','pytest','query_minco'];mods={n:importlib.import_module(n) for n in names};print(json.dumps({n:{'file':m.__file__,'version':getattr(m,'__version__',None)} for n,m in mods.items()},indent=2));from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import monitor_preflight;print(json.dumps(monitor_preflight()))"
                q.command([PYTHON,"-B","-c",script],run,"dependencies")
                for name,cmd in [("cxx",["/usr/bin/g++-9","--version"]),("nvcc",["/usr/local/cuda-11.8/bin/nvcc","--version"]),("nm",["nm","--version"]),("ldd",["ldd","--version"])]:
                    q.command(cmd,run,name)
                q.protect();write(run/"RESULT.json",dict(status="PASS",dependencies="fixed interpreter imports verified",environment_installed_or_modified=False))
            else:
                if json.loads((OUT/"runs/preflight_001/RESULT.json").read_text())["status"]!="PASS":raise RuntimeError("DEPENDENCY_PREFLIGHT_REQUIRED")
                if args.stage=="tests":
                    q.tests()
                    write(OUT/"runs/tests_001/RESULT.json",dict(status="PASS",source_freeze=q.read(PREP/"SOURCE_FREEZE.json")))
                else:
                    if json.loads((OUT/"runs/tests_001/RESULT.json").read_text())["status"]!="PASS":raise RuntimeError("TESTS_PASS_REQUIRED")
                    q.build()
                    art=OUT/"runs/build_001/artifacts"
                    q.command(["nm","-C",art/"e1_renderer_bridge"],OUT/"runs/build_001","abi_symbols")
                    # ABI dry invocation intentionally has no valid mode/cloud and exits before CUDA.
                    from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import supervise
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib",PYTHONDONTWRITEBYTECODE="1")
                    supervise([art/"e1_renderer_bridge"],OUT/"runs/build_001","abi_reject_noargs",cwd=BASE,env=env,
                              stop_path=PREP/"STOP.json",budget_root=OUT,guard=q.protect,expected_exit=(1,))
                    symbols=(OUT/"runs/build_001/abi_symbols_stdout.log").read_text()
                    for symbol in (" e1_export","raycast::GridMap::GridMap","raycast::renderDepthImage"):
                        if symbol not in symbols:raise RuntimeError("ABI_SYMBOL_MISSING "+symbol)
                    if "not found" in (OUT/"runs/build_001/ldd_stdout.log").read_text():raise RuntimeError("DYNAMIC_LIBRARY_MISSING")
                    write(OUT/"runs/build_001/RESULT.json",dict(status="PASS",ABI="symbols+link+negative argc; CUDA workload NOT_RUN",binary=q.read(OUT/"runs/build_001/BINARY.json")))
            q.protect();code=0
            stdout.write("PASS "+args.stage+"\n")
        except BaseException:
            error=traceback.format_exc();stderr.write(error);stderr.flush()
            try:write(PREP/"STOP.json",dict(reason="BOOTSTRAP_OR_STAGE_FAILURE",stage=args.stage,exit_code=1,traceback=error))
            except FileExistsError:pass
        finally:
            write(folder/"PROCESS_EXIT.json",dict(exit_code=code,elapsed_s=time.monotonic()-started,stage=args.stage,error=error))
    if code:raise SystemExit(code)
if __name__=="__main__":main()
