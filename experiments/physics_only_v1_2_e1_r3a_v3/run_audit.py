"""Thin versioned launch; reuse validated full-parent RSS supervisor only."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v3"
PREP=OUT/"_preparation/freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()},**read(PREP/"INPUT_HASHES.json")}.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError("SHA_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO"))):
        for key,cmd in (("head",["rev-parse","HEAD"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][key]:raise ValueError("GIT_DRIFT")
    if time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["tests","audit"],required=True);stage=ap.parse_args().stage
    folder=OUT/"audit/freeze_001"/("entry_"+stage+"_001");folder.mkdir()
    write(folder/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd())))
    try:
        guard()
        if sys.executable!=PYTHON:raise ValueError("FIXED_INTERPRETER")
        if stage=="audit" and read(PREP/"TESTS_PASS.json")["status"]!="PASS":raise ValueError("TESTS_REQUIRED")
        spec=importlib.util.spec_from_file_location("_frozen_r3a_v2_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        sys.path.insert(0,str(BASE))
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(folder/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(folder/"tmp").mkdir()
        cmd=[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_r3a_v3","--worker",stage]
        limit=min(600 if stage=="tests" else 1800,3600-(time.time()-read(PREP/"ENTRY.json")["started_unix"]))
        old.launch(cmd,folder/"driver",env,PREP/"STOP.json",OUT,limit)
        guard();write(folder/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
        if stage=="tests":write(PREP/"TESTS_PASS.json",dict(status="PASS"))
    except BaseException:
        error=traceback.format_exc()
        with (folder/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(folder/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        if not (PREP/"STOP.json").exists():write(PREP/"STOP.json",dict(stage=stage,error=error))
        raise
if __name__=="__main__":main()
