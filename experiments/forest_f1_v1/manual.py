"""F1 explicit contract. No physical permission is implied by offline replay."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/forest_f1_20261118_v2'
sys.path.insert(0,str(ROOT))
F0=ROOT/'research_outputs/f0_controller_consumption_20261118_v1/runs/F0_20261118_104715_1163/RESULT.json'
def historical_f0(path=F0):
    d=json.loads(Path(path).read_text())
    if d.get('stage')!='F0' or d.get('status')!='PASS' or d.get('execution_evidence',{}).get('controller_plan_lineage')!='PROVEN':raise ValueError('F0_HISTORICAL_QUALIFICATION')
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),scope='HISTORICAL_BIRTH_BALL_SHORT_MOTION_ONLY_NOT_NEW_BINDING_QUALIFICATION')
def qualification(root,binding):
    p=Path(root)/'F1_QUALIFICATION.json'
    if not p.exists():raise PermissionError('F1_SENSOR_ROUTE_EXECUTION_QUALIFICATION_PENDING')
    d=json.loads(p.read_text())
    if d.get('contract')!='forest_f1_observed_space_v1' or d.get('status')!='PASS' or d.get('source_binding')!=binding:raise PermissionError('F1_QUALIFICATION_IDENTITY')
    if d.get('historical_f0')!=historical_f0():raise PermissionError('F1_HISTORICAL_F0_IDENTITY')
    if d.get('outside_birth_replay_pass') is not True or d.get('consumption_tests_pass') is not True:raise PermissionError('F1_NONPHYSICAL_QUALIFICATION')
    replay=Path(d['replay_result_path'])
    if hashlib.sha256(replay.read_bytes()).hexdigest()!=d['replay_result_sha256']:raise PermissionError('F1_REPLAY_IDENTITY')
    rr=json.loads(replay.read_text())
    if rr.get('certified_outside_birth',0)<1 or rr.get('valid_command_fraction',0)<=0 or rr.get('longest_no_valid_command_s',1)>.2:raise PermissionError('F1_REPLAY_ROUTE_OR_CONTINUITY_PENDING')
    return d

def main():
    historical_f0()
    binding=json.loads((OUT/'F1_SOURCE_BINDING.json').read_text())
    for p,h in binding.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('F1_SOURCE_DRIFT '+p)
    stage=sys.argv[sys.argv.index('--stage')+1] if '--stage' in sys.argv else 'F1'
    if stage=='F1':
        try:qualification(OUT/'runtime',binding)
        except PermissionError as error:
            if '--check-only' not in sys.argv:raise
            print('PREFLIGHT_SOURCE_PASS_PREREQUISITE_PENDING_NO_ROS_NO_ARM',str(error));return
    from experiments.stop_latency_fix_v1.manual import main as shared
    shared(config_root=OUT/'runtime',binding_name='SOURCE_BINDING.json',audit_build=ROOT/'research_outputs/f0_controller_consumption_20261118_v1')
if __name__=='__main__':main()
