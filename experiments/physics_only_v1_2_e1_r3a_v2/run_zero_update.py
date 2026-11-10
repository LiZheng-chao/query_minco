"""R3a-v2 stdlib launcher; RSS includes this parent and every observed descendant."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v2"
PREP=OUT/"_preparation/freeze_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    e=read(PREP/"ENTRY.json")
    if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()}}.items():
        if sha(p)!=h:raise RuntimeError("SHA_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO"))):
        if subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True)!=e["git_before"][label]["head"]:raise RuntimeError("HEAD_DRIFT")
        for key,cmd in (("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][key]:raise RuntimeError("TRACKED_DIFF_DRIFT")
    if time.time()-e["started_unix"]>3600:raise RuntimeError("TOTAL_BUDGET")
def launch(argv,folder,env,stop_path,budget_root,timeout=600,rss_limit=4294967296):
    from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import tree_sample,terminate,disk_bytes,stop_once,monitor_preflight
    folder=Path(folder);folder.mkdir(parents=True)
    write(folder/"COMMAND.json",dict(argv=argv,cwd=str(BASE),env={k:v for k,v in env.items() if k in ("PYTHONPATH","CUDA_VISIBLE_DEVICES","PYTHONDONTWRITEBYTECODE","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTEST_DISABLE_PLUGIN_AUTOLOAD","TMPDIR")},timeout_s=timeout,RSS_limit=rss_limit,disk_limit=536870912,RSS_root=os.getpid(),scope="launcher itself + worker imports + all observed descendants",sample_interval_s=.05))
    proc=None;code=None;reason=None;err=None;child_seen={};seen={};peak=0;ownpeak=0;start=time.monotonic();left=[]
    with (folder/"stdout.log").open("x") as out,(folder/"stderr.log").open("x") as stderr:
        try:
            if Path(stop_path).exists():raise RuntimeError("PERSISTENT_STOP")
            monitor_preflight()
            proc=subprocess.Popen(argv,cwd=BASE,env=env,stdout=out,stderr=stderr,start_new_session=True)
            while True:
                rss,live=tree_sample(os.getpid(),seen)
                own=live[os.getpid()]["rss"];peak=max(peak,rss);ownpeak=max(ownpeak,own)
                tree_sample(proc.pid,child_seen)
                if rss>rss_limit:reason="COMPLETE_TREE_RSS_BUDGET"
                if time.monotonic()-start>timeout:reason="WALLTIME_BUDGET"
                if disk_bytes(budget_root)>536870912:reason="DISK_BUDGET"
                code=proc.poll()
                if reason:break
                if code is not None:
                    if code:reason="CHILD_EXIT"
                    break
                time.sleep(.05)
        except BaseException:
            reason=reason or "SUPERVISOR_EXCEPTION";err=traceback.format_exc();stderr.write(err);stderr.flush()
        finally:
            if proc:
                left=terminate(proc.pid,child_seen);code=proc.wait()
            if left:reason="LIVE_DESCENDANTS_AFTER_CLEANUP"
            status="PASS" if reason is None and code==0 else "STOP"
            receipt=dict(status=status,exit_code=code,reason=reason,error=err,child_started=proc is not None,elapsed_s=time.monotonic()-start,sampled_peak_full_tree_RSS_bytes=peak,sampled_peak_launcher_RSS_bytes=ownpeak,launcher_PID=os.getpid(),observed_processes=seen,live_descendants_after_cleanup=left,sampled_not_continuous=True)
            write(folder/"EXIT.json",receipt)
            if status!="PASS":stop_once(stop_path,receipt)
    if status!="PASS":raise RuntimeError("SUPERVISOR_STOP "+str(reason))
    return receipt
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("preflight","tests","zero"),required=True);stage=ap.parse_args().stage
    receipt=OUT/"audit/freeze_001"/("entry_"+stage+"_001");receipt.mkdir()
    write(receipt/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),executable=sys.executable))
    try:
        sys.path.insert(0,str(BASE));guard()
        if sys.executable!=PYTHON:raise RuntimeError("FIXED_INTERPRETER")
        for needed in (("DEPENDENCIES_PASS.json",) if stage=="tests" else ("DEPENDENCIES_PASS.json","TESTS_PASS.json") if stage=="zero" else ()):
            if read(PREP/needed)["status"]!="PASS":raise RuntimeError("PREREQUISITE "+needed)
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(receipt/"tmp"))
        env.pop("PYTEST_ADDOPTS",None);(receipt/"tmp").mkdir()
        cmd=[PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2","--worker",stage]
        timeout=min(1800 if stage=="zero" else 600,3600-(time.time()-read(PREP/"ENTRY.json")["started_unix"]))
        launch(cmd,receipt/"driver",env,PREP/"STOP.json",OUT,timeout)
        guard()
        write(receipt/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
        if stage!="zero":write(PREP/("DEPENDENCIES_PASS.json" if stage=="preflight" else "TESTS_PASS.json"),dict(status="PASS"))
    except BaseException:
        error=traceback.format_exc()
        with (receipt/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(receipt/"ENTRY_EXIT.json",dict(status="STOP",exit_code=1,error=error))
        if not (PREP/"STOP.json").exists():write(PREP/"STOP.json",dict(stage=stage,error=error))
        raise
if __name__=="__main__":main()
