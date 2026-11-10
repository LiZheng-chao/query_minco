"""Exclusive R3a entry; stdlib receipts precede third-party imports."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3_zero_update_v1"
PREP=OUT/"_preparation/r3a_001"
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()}}.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError("SHA_DRIFT "+p)
    for repo,head in (("/home/lizheng/query_minco_mainline","7a75e0bd000c1c7dbabe881dce7b44ad4668d5e6"),("/home/lizheng/YOPO","d5e1e60f83cc991602f142d9b6b96204a24d6b9f")):
        if subprocess.check_output(["git","-C",repo,"rev-parse","HEAD"],text=True).strip()!=head:raise RuntimeError("HEAD_DRIFT")
    if time.time()-e["started_unix"]>1800:raise RuntimeError("TOTAL_BUDGET")
def main():
    a=argparse.ArgumentParser();a.add_argument("--stage",choices=["preflight","tests","zero"],required=True);s=a.parse_args().stage
    receipt=OUT/"audit/r3a_001"/("entry_"+s+"_001");receipt.mkdir()
    write(receipt/"COMMAND.json",dict(argv=sys.argv,python=sys.executable))
    try:
        guard()
        if sys.executable!="/home/lizheng/anaconda3/envs/yopo/bin/python":raise RuntimeError("PYTHON_IDENTITY")
        sys.path.insert(0,str(BASE))
        from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import supervise
        spec=read(PREP/"RUN_SPEC.json")
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",TMPDIR=str(PREP/"tmp"))
        if s=="tests":
            dest=OUT/"runs/tests_001";dest.mkdir(parents=True)
            cmd=[sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",*spec["test_paths"],"--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes","--basetemp="+str(dest/"pytest_tmp")]
        else:cmd=[sys.executable,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_r3_zero_update","--worker",s]
        timeout=min(1200 if s=="zero" else 300,1800-(time.time()-read(PREP/"ENTRY.json")["started_unix"]))
        r=supervise(cmd,receipt,"worker",cwd=BASE,env=env,stop_path=PREP/"STOP.json",budget_root=OUT,timeout=timeout,rss_limit=4*1024**3,disk_limit=512*1024**2,guard=None)
        guard();write(receipt/"EXIT.json",dict(exit_code=0,status="PASS",worker=r))
    except BaseException:
        error=traceback.format_exc()
        with (receipt/"stderr.log").open("x") as f:f.write(error)
        write(receipt/"EXIT.json",dict(exit_code=1,status="STOP",error=error))
        if not (PREP/"STOP.json").exists():write(PREP/"STOP.json",dict(stage=s,error=error))
        raise
if __name__=="__main__":main()
