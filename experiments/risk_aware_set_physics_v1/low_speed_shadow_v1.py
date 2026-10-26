"""Independent preparation and retained numerical assertions for reference-driven Shadow."""
import argparse,collections,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from query_minco.inference.shadow import sha,require,FROZEN,FrozenPair
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.ros_shadow_low_speed import topology_check
from experiments.risk_aware_set_physics_v1.low_speed_support_v1 import joint
OUT=ROOT/'research_outputs/rsps_forest_shadow_v1'
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def prepare():
    prep=OUT/'low_speed_preparation001';run=OUT/'low_speed_shadow001';host=OUT/'low_speed_host001'
    for p in [prep,run,host]:p.mkdir()
    for p in [run,host]:
        for d in ['logs','runtime','roslog']:(p/d).mkdir()
    (host/'logs/host_run.py').write_bytes((ROOT/'experiments/risk_aware_set_physics_v1/low_speed_host_v1.py').read_bytes())
    # Preserve the already validated sensor geometry, map and exact ray lattice.
    (prep/'sensor.yaml').write_bytes((OUT/'forest85s_preparation001/sensor.yaml').read_bytes())
    (run/'sensor.yaml').write_bytes((prep/'sensor.yaml').read_bytes())
    b=json.loads((OUT/'forest85s_shadow001/BINDING.json').read_text())
    for p,h in b['files'].items():require(sha(p)==h,'PRIOR_BINDING_DRIFT '+p)
    additions=[ROOT/'query_minco/inference'/n for n in ['ros_shadow_low_speed.py','shadow_camera_low_speed.py','low_speed_reference.py','online_selector.py']]
    additions += [Path(__file__),ROOT/'experiments/risk_aware_set_physics_v1/low_speed_commands_v1.sh',
        ROOT/'experiments/risk_aware_set_physics_v1/LOW_SPEED_SHADOW_PREREG_v1.json',prep/'sensor.yaml',
        host/'logs/host_run.py',ROOT/'experiments/risk_aware_set_physics_v1/low_speed_support_v1.py']
    b['files'].update({str(p):sha(p) for p in additions});b['schema']='controlled_low_speed_sensor_v1'
    for p in [prep/'EXECUTION_BINDING.json',run/'BINDING.json']:dump(p,b)
    prereg=ROOT/'experiments/risk_aware_set_physics_v1/LOW_SPEED_SHADOW_PREREG_v1.json'
    dump(run/'RUN_SPEC.json',dict(schema=b['schema'],models=['C1','E3'],global_goal_W=[30,0,2],initial_origin_W=[0,0,2],
        local_goal_m=2.4,frozen_models_sha=sha(FROZEN),sensor_config_sha=sha(run/'sensor.yaml'),
        host=str(host),prereg=dict(path=str(prereg),sha256=sha(prereg)),query_control=False,training=False,
        run=str(run),classification='CONTROLLED_REFERENCE_SENSOR_INTEGRATION_NOT_YOPO_AUTONOMY'))
    dump(run/'ENVIRONMENT.json',dict(executable='/home/lizheng/anaconda3/envs/yopo/bin/python',device='cpu',ROS_MASTER_URI='http://127.0.0.1:11311'))
    dump(run/'COMMAND.json',dict(argv=['bash',str(ROOT/'experiments/risk_aware_set_physics_v1/low_speed_commands_v1.sh'),'start',run.name]))
def check(run):
    spec=json.loads((run/'RUN_SPEC.json').read_text());host=Path(spec['host'])
    for n,h in json.loads((run/'BINDING.json').read_text())['files'].items():require(sha(n)==h,'BOUND_FILE_DRIFT '+n)
    require(sha(run/'sensor.yaml')==spec['sensor_config_sha'],'SENSOR_CONFIG_DRIFT')
    ev=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    allframes=[x for x in ev if x['type']=='FRAME']
    decision=json.loads((host/'PILOT_DECISION.json').read_text())
    camera=[json.loads(l) for l in (host/'CAMERA_TRACE.jsonl').read_text().splitlines()]
    require(not any(x.get('type')=='FATAL' for x in camera),'CAMERA_FATAL')
    phases=[x.get('type') for x in camera if 'type' in x]
    require(phases[-4:]==['QUIESCE_BEGIN','CALLBACKS_DRAINED','PUBLISHERS_CLOSED','EXIT'],'CAMERA_CALLBACK_DRAIN_EXIT')
    bystamp={x['stamp_ns']:x for x in camera if 'stamp_ns' in x}
    from query_minco.inference.shadow import decode_image
    from query_minco.inference.shadow_camera30 import extract
    import hashlib
    attempts=[x for x in ev if x['type'] in ('FRAME','REJECT')]
    require(len(list(run.glob('attempt_*_input.npz')))==len(attempts),'RAW_ATTEMPT_COUNT')
    for e in attempts:
        art=e.get('input_artifact') or e['attempt']['input_artifact']
        with np.load(run/art,allow_pickle=False) as z:
            ns=int(z['stamp_ns']);require(ns==int(z['camera_stamp_ns'])==int(z['odom_stamp_ns'])==e['stamp_ns'],'SYNC')
            encoded=decode_image(z['depth_bytes'].tobytes(),str(z['encoding']),int(z['width']),int(z['height']),int(z['step']),bool(z['is_bigendian']))
            require(hashlib.sha256(encoded.tobytes()).hexdigest()==bystamp[ns]['query_encoded_sha'],'SOURCE_PROVENANCE')
    for x in camera:
        if 'proof' in x:
            with np.load(host/x['proof']) as z:
                q,y=extract(z['full']);require(np.array_equal(q,z['query'],equal_nan=True) and np.array_equal(y,z['yopo'],equal_nan=True),'EXACT_RAY_PROOF')
    frames=[x for x in allframes if x['stamp']>=(decision.get('full_start_stamp') or float('inf'))]
    pair=FrozenPair(spec['frozen_models_sha']);repeat=0;fingerprints=set()
    start=json.loads((run/'START.json').read_text());require(start['models']==pair.bindings and not start['training'] and not start['control'],'FROZEN_PAIR_IDENTITY')
    for x in allframes:
        require(x['selected_query'] is None and x['online_safe_status'].startswith('UNKNOWN'),'NO_CONTROL_SELECTION')
        require(0<=x['input_age_s']<=.25 and 0<=x['output_age_s']<=.25,'AGE')
        with np.load(run/('frame_%06d.npz'%x['index'])) as z:
            InferenceAdapter.inputs(z['depth_input'][None],z['state'][None],None)
            require(np.array_equal(z['global_goal_W'],[30,0,2]),'GOAL')
            require(abs(np.linalg.norm(z['state'][6:9])-2.4)<1e-5,'LOCAL_GOAL')
            require(float(z['stamp'])==x['stamp'] and int(z['stamp_ns'])==x['stamp_ns'],'STAMP')
            require(np.linalg.norm(z['velocity_W'])>=.05,'DYNAMIC')
            require(all(np.isfinite(z[k]).all() for k in z.files),'FINITE')
            rebuilt=pair.forward(z['depth_input'],z['state'])
            for label,data in rebuilt.items():
                require(z[label+'_raw'].shape==(6,21) and z[label+'_coefficients'].shape==(6,3,3,6),'SHAPE')
                d=z[label+'_durations'];require((d>=.12-1e-6).all() and (d.sum(-1)>=.8-1e-6).all() and (d.sum(-1)<=2.2+1e-6).all(),'HORIZON')
                require(z[label+'_solve_valid'].all(),'SOLVE')
                require(np.max(np.abs(z[label+'_head_pva']-np.stack([np.zeros(3),z['state'][:3],z['state'][3:6]])))<1e-5,'HEAD_PVA')
                for k,v in data.items():require(np.array_equal(v,z[label+'_'+k]),'CPU_REBUILD '+label+' '+k)
                repeat+=6
            if x in frames:fingerprints.add(hashlib.sha256(z['depth_input'].tobytes()).hexdigest())
    top=[x for x in ev if x['type']=='TOPOLOGY'];require(len(top)>=2,'TOPOLOGY_EVIDENCE')
    for x in top:topology_check([(t,['/query_minco_shadow']) for t in x['shadow_publishers']]+list(x['control_publishers'].items())+[('/sim/odom',x['odom_publishers'])])
    result=json.loads((run/'RESULT.json').read_text())
    exit=json.loads((run/'PROCESS_EXIT.json').read_text())
    require(result['status']=='RUNTIME_COMPLETE' and not result['control'] and result['counters']['accepted']==len(allframes)<=300,'RESULT_COUNT')
    require(exit['bridge_exit']==0 and not exit['resource_violation'] and not exit['sensor_exited_early'],'EXIT')
    require(json.loads((run/'LAUNCHER_EXIT.json').read_text())['returncode']==0,'LAUNCHER')
    stamps=[x['stamp_ns'] for x in frames]
    span=(stamps[-1]-stamps[0])/1e9 if len(stamps)>1 else 0;gap=float(max(np.diff(stamps))/1e9) if len(stamps)>1 else None
    failures=[]
    if not decision['passed']:failures.append('PILOT_FAILED_FULL_NOT_RUN')
    if len(frames)<50:failures.append('FRAMES_LT50')
    if span<9:failures.append('SPAN_LT9')
    if gap is None or gap>.3:failures.append('MAX_GAP_GT0P3')
    if len(fingerprints)<3:failures.append('DYNAMIC_DEPTH')
    r=dict(F2='FAIL' if failures else 'PASS_CONTROLLED_REFERENCE_SENSOR_ONLY',failures=failures,pilot=decision,
        all_frames=len(allframes),full_frames=len(frames),full_span_s=span,full_max_gap_s=gap,distinct_full_depths=len(fingerprints),
        rebuilt_candidates=repeat,models_p50_p95_ms={l:(1000*np.quantile([x['model_generation_s'][l] for x in allframes],[.5,.95])).tolist() for l in ['C1','E3']},
        output_age_p50_p95_ms=(1000*np.quantile([x['output_age_s'] for x in allframes],[.5,.95])).tolist(),
        rejects=dict(collections.Counter(x['reason'] for x in ev if x['type']=='REJECT')),control=False,autonomous_forest_F2='NOT_QUALIFIED',F3='BLOCKED')
    dump(run/'CHECK_RESULT.json',r);print(json.dumps(r,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check']);p.add_argument('--run',default=str(OUT/'low_speed_shadow001'));a=p.parse_args()
    prepare() if a.action=='prepare' else check(Path(a.run))
if __name__=='__main__':main()
