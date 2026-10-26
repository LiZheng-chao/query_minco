"""Independent 30m preparation/check; preserves original numerical assertions."""
import argparse,collections,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from query_minco.inference.shadow import sha,require,FROZEN
from query_minco.inference.shadow_camera30 import extract
OUT=ROOT/'research_outputs/rsps_forest_shadow_v1';PREP=OUT/'forest30m_preparation001'
spec=importlib.util.spec_from_file_location('legacy_shadow_check',ROOT/'experiments/risk_aware_set_physics_v1/forest_shadow_v1.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def prepare(run):
    b=legacy.verify_binding()
    require(not PREP.exists() and not run.exists(),'EXCLUSIVE_PREPARATION_AND_RUN')
    PREP.mkdir()
    source=yaml.safe_load(legacy.CONFIG.read_text())
    source.update(render_lidar=False,render_stereo=False,render_depth=True,depth_fps=33,
        depth_topic='/forest30/native_depth',camera_info_topic='/forest30/native_camera_info')
    source['camera'].update(fx=400.,fy=400.,cx=400.,cy=254.,image_width=800,image_height=512,pitch=0.,max_depth_dist=20.,normalize_depth=False)
    source['tree_file']=str(legacy.SIMROOT/'src/pointcloud/tree.ply');source['ply_file']=str(legacy.SIMROOT/'src/pointcloud/forest.ply')
    (PREP/'sensor.yaml').write_text(yaml.safe_dump(source,sort_keys=False))
    additions=[ROOT/'query_minco/inference/shadow_camera30.py',ROOT/'query_minco/inference/ros_shadow30.py',Path(__file__),
        ROOT/'experiments/risk_aware_set_physics_v1/forest30m_commands_v1.sh',
        ROOT/'experiments/risk_aware_set_physics_v1/FOREST30M_SHADOW_PROTOCOL_v1.md',
        ROOT/'experiments/risk_aware_set_physics_v1/test_forest30m_shadow_v1.py',
        Path('/home/lizheng/YOPO/Simulator/src/src/sensor_simulator.cu'),
        Path('/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/launch/simulator_attitude_control.launch'),PREP/'sensor.yaml']
    b['files'].update({str(p.resolve()):sha(p) for p in additions});b['schema']='forest30m_exact_ray_lattice_v1'
    dump(PREP/'EXECUTION_BINDING.json',b)
    run.mkdir()
    for n in ['logs','runtime','roslog']:(run/n).mkdir()
    (run/'sensor.yaml').write_bytes((PREP/'sensor.yaml').read_bytes())
    dump(run/'BINDING.json',b)
    dump(run/'RUN_SPEC.json',dict(schema=b['schema'],models=['C1','E3'],run=str(run),camera=dict(K=[[100.,0,79.5],[0,100.,63.5],[0,0,1]]),
        global_goal_W=[30,0,2],initial_origin_W=[0,0,2],local_goal_m=2.4,frozen_models_sha=sha(FROZEN),
        sensor_config_sha=sha(run/'sensor.yaml'),native_config_sha=sha(legacy.CONFIG),max_wall_seconds=180,bridge_seconds=120,max_frames=300,
        query_fps=5,renderer_fps=33,control=False,training=False,GT_for_online_selection=False,
        host=str(OUT/'forest30m_host001'),sensor_reuse='EXACT_INTEGER_NATIVE_RAY_EXTRACTION'))
    dump(run/'ENVIRONMENT.json',dict(executable=sys.executable,bridge_device='cpu',goal_W=[30,0,2],local_goal_m=2.4))
    dump(run/'COMMAND.json',dict(launcher_argv=['/bin/bash',str(ROOT/'experiments/risk_aware_set_physics_v1/forest30m_commands_v1.sh'),'start',run.name]))
def check(run):
    spec=json.loads((run/'RUN_SPEC.json').read_text());host=Path(spec['host'])
    require(spec['global_goal_W']==[30,0,2] and spec['local_goal_m']==2.4,'FOREST30_SPEC')
    for p,v in json.loads((run/'BINDING.json').read_text())['files'].items():require(sha(p)==v,'BOUND_FILE_DRIFT '+p)
    events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
    frames=[x for x in events if x['type']=='FRAME']
    camera=[json.loads(l) for l in (host/'CAMERA_TRACE.jsonl').read_text().splitlines()]
    require(not any(x.get('type')=='FATAL' for x in camera),'CAMERA_FATAL');bystamp={x['stamp_ns']:x for x in camera}
    for x in camera:
        if 'proof' in x:
            with np.load(host/x['proof'],allow_pickle=False) as z:
                q,y=extract(z['full'])
                require(np.array_equal(q,z['query'],equal_nan=True) and np.array_equal(y,z['yopo'],equal_nan=True),'NATIVE_EXTRACTION_PROOF')
    stamps=[];fingerprints=set()
    for x in frames:
        with np.load(run/('frame_%06d.npz'%x['index']),allow_pickle=False) as z:
            require(np.array_equal(z['global_goal_W'],[30,0,2]),'WORLD_GOAL')
            require(np.linalg.norm(z['velocity_W'])>=.05,'STATIC_FRAME')
            digest=hashlib.sha256(z['depth_input'].tobytes()).hexdigest()
            require(digest==bystamp[x['stamp_ns']]['query_encoded_sha'],'SOURCE_SENSOR_PROVENANCE')
            fingerprints.add(digest);stamps.append(x['stamp_ns'])
    summary=dict(frames=len(frames),dynamic_sensor_span_s=(stamps[-1]-stamps[0])/1e9 if len(stamps)>1 else 0,
        distinct_depths=len(fingerprints),max_gap_s=float(max(np.diff(stamps))/1e9) if len(stamps)>1 else None,
        rejects=dict(collections.Counter(x['reason'] for x in events if x['type']=='REJECT')),
        per_model_p50_p95_s={k:np.quantile([x['model_generation_s'][k] for x in frames],[.5,.95]).tolist() if frames else None for k in ['C1','E3']},
        paired_p50_p95_s=np.quantile([x['generation_s'] for x in frames],[.5,.95]).tolist() if frames else None,control=False,F3='BLOCKED')
    print(json.dumps(summary,indent=2),flush=True)
    require(len(frames)>=50,'FOREST30_INSUFFICIENT_DYNAMIC_FRAMES')
    require(summary['dynamic_sensor_span_s']>=9 and summary['max_gap_s']<=.3,'FOREST30_DYNAMIC_TIMELINE')
    require(json.loads((run/'GLOBAL_GOAL.json').read_text())['goal_W']==[30.,0.,2.],'RECORDED_GLOBAL_GOAL')
    legacy.check(run)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check']);p.add_argument('--run',required=True)
    a=p.parse_args();run=Path(a.run).resolve();require(run.parent==OUT,'RUN_LOCATION')
    (prepare if a.action=='prepare' else check)(run)
if __name__=='__main__':main()
