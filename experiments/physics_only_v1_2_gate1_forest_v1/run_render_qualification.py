"""Exactly one authorized CUDA forest export, then separate fail-closed CPU checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research_outputs/physics_only_v1_2_gate1_forest_v1"
s=importlib.util.spec_from_file_location("gate1_receipts",Path(__file__).with_name("run_gate1.py"))
runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner)
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
SAMPLE="train_sparse_1001_001"


def write(path,x):
    with path.open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("render","qualify","tests","cpu0"),required=True)
    p.add_argument("--attempt",type=int,choices=(1,2),default=1)
    args=p.parse_args()
    binary=json.loads((OUT/"runs/build_002/BINARY.json").read_text())
    if hashlib.sha256(Path(binary["path"]).read_bytes()).hexdigest()!=binary["sha256"]:
        raise RuntimeError("BINARY_SHA_DRIFT")
    if args.stage=="render":
        assert not (OUT/"data"/SAMPLE).exists()
        case="render_sparse_1001_%03d"%args.attempt
        assert not (OUT/"runs"/case).exists()
        old=json.loads((OUT/"_preparation/ENTRY.json").read_text())
        protected=dict(old["protected_sha256"])
        for path in OUT.rglob("*"):
            if path.is_file():protected[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        build_sources=json.loads((OUT/"_preparation/BUILD_002_SOURCE_HASHES.json").read_text())
        for path,sha in build_sources.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
            protected[path]=sha
        protected[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        child_env={"LD_LIBRARY_PATH":"/opt/ros/noetic/lib:/usr/local/cuda-11.8/lib64:"+os.environ.get("LD_LIBRARY_PATH","")}
        write(OUT/"_preparation"/("RENDER_%03d_ENTRY.json"%args.attempt),dict(
            query_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            query_status=subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],text=True),
            yopo_head=old["yopo_head"],yopo_status=old["yopo_status"],protected_sha256=protected,
            binary=binary,requested_cuda_invocations=1,scene_seed=1001,scene_case="sparse",child_extra_env=child_env,
            reference_body="V12_REFERENCE_SPHERE_0p12_NOT_AIRFRAME_CERTIFICATE",
            optimizer_updates=0,episodes=0))
        runner.command(case,[binary["path"],"/home/lizheng/YOPO",
                       str(OUT/"data"/SAMPLE),"sparse","1001"],extra_env=child_env)
        write(OUT/"runs"/case/"EXPORTED_SHA256.json",
              {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (OUT/"data"/SAMPLE).iterdir() if path.is_file()})
    elif args.stage=="tests":
        code="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','query_minco/training/physics_only_v1_2_gate1/tests','query_minco/training/physics_only_v1_2/tests']))"
        runner.command("gate1_tests_001",[PYTHON,"-B","-c",code],extra_env={"CUDA_VISIBLE_DEVICES":""})
    else:
        case="qualify_sparse_1001_001" if args.stage=="qualify" else "gate1_cpu0_001"
        runner.command(case,[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.preflight",
                       "--stage",args.stage,"--sample",SAMPLE,"--output-root",str(OUT)],
                       extra_env={"CUDA_VISIBLE_DEVICES":""})

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("STOP",repr(e),file=sys.stderr);raise SystemExit(1)
