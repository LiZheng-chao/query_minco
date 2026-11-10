"""Exclusive offline Gate1 receipts. No ROS node or overwrite-style generator."""
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_outputs/physics_only_v1_2_gate1_forest_v1"
CONFIG = Path(__file__).resolve().with_name("gate1_config.json")
PYTHON = "/home/lizheng/anaconda3/envs/yopo/bin/python"
spec = importlib.util.spec_from_file_location(
    "v12_resource_helpers", ROOT/"experiments/physics_only_v1_2_math_contract_v1/run_math_acceptance.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def command(case, argv, *, extra_env=None, limit_s=300):
    dest = OUT/"runs"/case
    dest.mkdir()
    temp = dest/"tmp";temp.mkdir()
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp), ROS_LOG_DIR=str(temp/"ros_logs"))
    if extra_env:env.update(extra_env)
    write(dest/"COMMAND.json",dict(argv=argv,cwd=str(ROOT),
          env={k:env[k] for k in ("PYTHONPATH","PYTHONDONTWRITEBYTECODE","CUDA_VISIBLE_DEVICES",
             "OMP_NUM_THREADS","TMPDIR","ROS_LOG_DIR")},timeout_s=limit_s,
          rss_bytes_limit=4*1024**3,total_disk_bytes_limit=512*1024**2))
    start=time.monotonic();peak=0;stop=None;code=None;error=None
    with (dest/"stdout.log").open("xb") as stdout,(dest/"stderr.log").open("xb") as stderr:
        child=None
        try:
            child=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,helpers.process_tree_rss(child.pid))
                if time.monotonic()-start>limit_s:stop="TIME_BUDGET"
                if peak>4*1024**3:stop="RSS_BUDGET"
                if helpers.disk_bytes(OUT)>512*1024**2:stop="DISK_BUDGET"
                if stop:
                    os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=child.wait()
        except BaseException:
            error=traceback.format_exc()
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    receipt=dict(exit_code=code,stop_reason=stop,supervisor_error=error,elapsed_s=time.monotonic()-start,
                 sampled_peak_tree_rss_bytes=peak,disk_bytes=helpers.disk_bytes(OUT),
                 optimizer_updates=0,episodes=0)
    write(dest/"PROCESS_EXIT.json",receipt)
    print(json.dumps(dict(case=case,**receipt)),flush=True)
    if code!=0 or stop or error:
        raise RuntimeError("STOP: "+case)
    return dest


def renderer(config):
    source=Path(config["yopo_root"])/"Simulator/src"
    sources=[source/"src"/"maps.cpp",source/"src"/"perlinnoise.cpp"]+sorted((source/"src/simsense").glob("*.cu"))
    readonly=[source/"src/sensor_simulator.cu",source/"config/config.yaml",source/"pointcloud/tree.ply"]+sources
    readonly += [ROOT/"query_minco/training/physics_only_v1_2_gate1/renderer_bridge.cu", CONFIG, Path(__file__).resolve()]
    readonly+=list((source/"include").rglob("*.h"))+list((source/"include").rglob("*.hpp"))+list((source/"include").rglob("*.cuh"))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in readonly}
    write(OUT/"_preparation"/"RENDERER_SOURCE_HASHES.json",hashes)
    binary=OUT/"_preparation"/"renderer_bridge"
    assert not binary.exists()
    flags=["-I"+str(source/"include"),"-I"+str(source/"src"),
           "-I/usr/include/opencv4","-I/usr/include/pcl-1.10","-I/usr/include/eigen3",
           "-I/opt/ros/noetic/include"]
    argv=["/usr/local/cuda-11.8/bin/nvcc","-std=c++17","-O2","-arch=sm_75",
          "-ccbin","/usr/bin/g++-9",*flags,str(ROOT/"query_minco/training/physics_only_v1_2_gate1/renderer_bridge.cu"),
          *map(str,sources),"-o",str(binary),"-L/opt/ros/noetic/lib","-lopencv_core","-lopencv_imgproc",
          "-lopencv_calib3d","-lpcl_common","-lpcl_io","-lyaml-cpp","-lrosconsole","-lrostime",
          "-lcpp_common","-lroscpp","-lcurand"]
    command("build_001",argv)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    write(OUT/"_preparation"/"RENDERER_BINARY.json",dict(path=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
          access_visibility_only="GridMap private->public in qualification translation unit; original map/raycast math unchanged",
          no_ros_nodes=True))
    for row in config["scenes"]:
        command("render_"+row["sample_id"],[str(binary),config["yopo_root"],
               str(OUT/"data"/row["sample_id"]),row["case"],str(row["seed"])])


def build_only(config, case):
    """Compile/link only. Never execute the resulting renderer binary."""
    if case != "build_002":
        raise ValueError("only the explicitly authorized build_002 case is supported")
    dest = OUT/"runs"/case
    if dest.exists():
        raise ValueError("build case exists")
    source = Path(config["yopo_root"])/"Simulator/src"
    module = ROOT/"query_minco/training/physics_only_v1_2_gate1"
    artifacts = dest/"artifacts"
    includes = ["-I"+str(source/"include"), "-I"+str(source/"src"),
                "-I/usr/include/opencv4", "-I/usr/include/pcl-1.10",
                "-I/usr/include/eigen3", "-I/opt/ros/noetic/include"]
    cuda_sources = [module/"renderer_bridge.cu"] + sorted((source/"src/simsense").glob("*.cu"))
    host_sources = [module/"renderer_host.cpp", source/"src/maps.cpp", source/"src/perlinnoise.cpp"]
    plan, objects = [], []
    for src in cuda_sources:
        obj = artifacts/(src.stem+".cuda.o"); objects.append(obj)
        plan.append(["/usr/local/cuda-11.8/bin/nvcc", "-std=c++17", "-O2", "-arch=sm_75",
                     "-ccbin", "/usr/bin/g++-9", *includes, "-c", str(src), "-o", str(obj)])
    for src in host_sources:
        obj = artifacts/(src.stem+".host.o"); objects.append(obj)
        plan.append(["/usr/bin/g++-9", "-std=c++17", "-O2", *includes,
                     "-c", str(src), "-o", str(obj)])
    binary = artifacts/"renderer_bridge"
    plan.append(["/usr/bin/g++-9", *map(str,objects), "-o", str(binary),
                 "-L/usr/local/cuda-11.8/lib64", "-Wl,-rpath,/usr/local/cuda-11.8/lib64",
                 "-L/opt/ros/noetic/lib", "-Wl,-rpath,/opt/ros/noetic/lib",
                 "-lopencv_core", "-lopencv_imgproc", "-lopencv_calib3d",
                 "-lpcl_common", "-lpcl_io", "-lyaml-cpp", "-lrosconsole", "-lrostime",
                 "-lcpp_common", "-lroscpp", "-lcurand", "-lcudart"])
    readonly = cuda_sources + host_sources + [
        source/"src/sensor_simulator.cu", source/"config/config.yaml",
        source/"pointcloud/tree.ply", CONFIG, Path(__file__).resolve()]
    for pattern in ("*.h","*.hpp","*.cuh"):
        readonly += list((source/"include").rglob(pattern))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in readonly}
    write(OUT/"_preparation"/"BUILD_002_SOURCE_HASHES.json", hashes)
    # The child dispatches only compilers/linker; never the compiled binary.
    body = """for index,argv in enumerate(plan):
 print('BUILD_STEP',index,argv,flush=True)
 started=time.monotonic()
 result=subprocess.run(argv)
 f.write(json.dumps(dict(step=index,argv=argv,exit_code=result.returncode,elapsed_s=time.monotonic()-started))+chr(10));f.flush()
 if result.returncode: sys.exit(result.returncode)
"""
    script = (
        "import json,subprocess,sys,time;from pathlib import Path;"
        "plan="+repr(plan)+";"
        "Path("+repr(str(artifacts))+").mkdir();"
        "f=Path("+repr(str(dest/"STEP_EXITS.jsonl"))+").open('x');"
        "exec("+repr(body)+")")
    command(case, [sys.executable, "-B", "-c", script])
    if not all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in hashes.items()):
        raise RuntimeError("source drift during build")
    write(dest/"BINARY.json", dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
          build_only=True, binary_executed=False, renderer_qualified=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("build","renderer","qualify","tests","profile"))
    p.add_argument("--case")
    args=p.parse_args()
    config=json.loads(CONFIG.read_text())
    if args.stage=="build":build_only(config,args.case)
    elif args.stage=="renderer":renderer(config)
    elif args.stage in ("qualify","profile"):
        command(args.stage+"_001",[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.preflight",
                 "--stage",args.stage,"--config",str(CONFIG),"--output-root",str(OUT)],
                extra_env={"CUDA_VISIBLE_DEVICES":""})
    else:
        code="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','query_minco/training/physics_only_v1_2_gate1/tests','query_minco/training/physics_only_v1_2/tests']))"
        command("tests_001",[PYTHON,"-B","-c",code],extra_env={"CUDA_VISIBLE_DEVICES":""})

if __name__=="__main__":
    try:main()
    except Exception as e:
        print(str(e),file=sys.stderr);raise SystemExit(1)
