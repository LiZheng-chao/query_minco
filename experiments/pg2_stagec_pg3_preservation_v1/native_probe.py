"""Reuse sealed original-YOPO diagnostic runner on a new isolated master. No Query commands."""
from pathlib import Path
import sys,hashlib,json,subprocess
ROOT=Path('/home/lizheng/query_minco_mainline')
OLD=ROOT/'research_outputs/pg3_minimal_stageabc_v1/SYSTEM'
BASE=ROOT/'research_outputs/pg2_stagec_pg3_preservation_v1/SYSTEM'
def run(kind):
    if kind not in ('moving','watchdog'):raise ValueError('DIAGNOSTIC_ONLY')
    BASE.mkdir(exist_ok=True);out=BASE/(kind+'001');out.mkdir(exist_ok=False)
    old=OLD/('run_moving_shadow.py' if kind=='moving' else 'run_native_watchdog_probe.py')
    s=old.read_text();s=s.replace("root=Path(__file__).resolve().parent;out=root/'"+('moving001' if kind=='moving' else 'watchdog001')+"';out.mkdir(exist_ok=False);port="+('11343' if kind=='moving' else '11344'),"root=Path("+repr(str(OLD))+");out=Path("+repr(str(out))+ ");port="+('11401' if kind=='moving' else '11402'))
    s=s.replace(' from cleanup_owned_ros import cleanup',' sys.path.insert(0,str(root))\n from cleanup_owned_ros import cleanup')
    observer=ROOT/'experiments/pg2_stagec_pg3_preservation_v1/lease_observer.py'
    s=s.replace("('yopo',[", "('lease',['/home/lizheng/anaconda3/envs/yopo/bin/python','-B','-s','-u',"+repr(str(observer))+",str(out)],Path('/home/lizheng/query_minco_mainline')),('yopo',[")
    generated=out/'EXECUTED_RUNNER.py';generated.write_text(s)
    (out/'PROVENANCE.json').write_text(json.dumps(dict(original_path=str(old),original_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),changes='new output, isolated port, explicit immutable helper import; original modules unchanged',Query_control=False),indent=2))
    rc=subprocess.call([sys.executable,'-B',str(generated)])
    if rc:raise RuntimeError('RUNNER_FAILED:'+str(rc))
if __name__=='__main__':run(sys.argv[1])
