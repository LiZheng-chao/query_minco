"""Manual shadow preparation/checking only; no ROS, simulator, or GPU execution."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from query_minco.inference.shadow import FROZEN, sha, require, K, FrozenPair
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.ros_shadow import OUTROOT, dump, topology_check

PREP = OUTROOT/'preparation001'
SIMROOT = Path('/home/lizheng/YOPO/Simulator')
CONFIG = SIMROOT/'src/config/config.yaml'

def verify_binding():
    b = json.loads((PREP/'configs/EXECUTION_BINDING.json').read_text())
    for n, h in b['files'].items():
        require(sha(n) == h, 'SOURCE_OR_ASSET_DRIFT '+n)
    for root, head in b['heads'].items():
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == head, 'GIT_DRIFT '+root)
    return b

def prepare(run):
    b = verify_binding()
    require(run.parent == OUTROOT and not run.exists(), 'RUN_MUST_BE_NEW')
    cfg = yaml.safe_load(CONFIG.read_text())
    require(cfg['random_map'] and cfg['maze_type'] == 5, 'NATIVE_FOREST_CONFIG_REQUIRED')
    # Second native renderer, same forest/config as YOPO, Query-specific intrinsics.
    cfg.update(render_lidar=False, render_stereo=False, render_depth=True, depth_fps=5,
               depth_topic='/query_shadow/depth', camera_info_topic='/query_shadow/camera_info',
               stereo_topic='/query_shadow/unused_stereo', lidar_topic='/query_shadow/unused_lidar')
    cfg['camera'].update(fx=100., fy=100., cx=79.5, cy=63.5, image_width=160,
                         image_height=128, pitch=0., max_depth_dist=20., normalize_depth=False)
    cfg['tree_file'] = str(SIMROOT/'src/pointcloud/tree.ply')
    cfg['ply_file'] = str(SIMROOT/'src/pointcloud/forest.ply')
    run.mkdir()
    for n in ['logs', 'runtime', 'roslog']:
        (run/n).mkdir()
    (run/'sensor.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    dump(run/'BINDING.json', b)
    dump(run/'COMMAND.json', dict(launcher_argv=['/bin/bash',str(ROOT/'experiments/risk_aware_set_physics_v1/forest_shadow_commands_v1.sh'),'start',run.name], argv_spec='Exact process argv also retained in logs/launcher.log via bash xtrace'))
    dump(run/'ENVIRONMENT.json',dict(executable=sys.executable,python=sys.version,numpy=np.__version__,ROS_MASTER_URI='http://127.0.0.1:11311',model_device='cpu',bridge_cuda_visible_devices='',renderer_cuda_visible_devices='0'))
    dump(run/'RUN_SPEC.json', dict(schema='query_shadow_v1', models=['C1','E3'],
        run=str(run), camera=dict(K=K.tolist(), optical_origin='native_odom_zero_lever_arm'),
        seed=cfg['seed'], tree_dist=cfg['tree_dist'], scene_config_sha=sha(CONFIG),
        sensor_config_sha=sha(run/'sensor.yaml'), frozen_models_sha=sha(FROZEN),
        global_goal='native_sim_initial_position_plus_world_x_15m_readonly_ROS_parameters',
        local_goal_m=2.4, terminal='REJECT_LT_2P4_NO_BRAKE',
        max_wall_seconds=180, bridge_seconds=120, max_frames=300,
        renderer_fps=5, cpu_inference=True, control=False, GT_for_online_selection=False))
    print(json.dumps(dict(run=str(run), frozen_sha=sha(FROZEN))))

def check(run):
    require(run.parent == OUTROOT, 'RUN_LOCATION')
    # Verify the binding captured by this run, not an updated receipt.
    binding = json.loads((run/'BINDING.json').read_text())
    for n,h in binding['files'].items():
        require(sha(n)==h, 'BOUND_FILE_DRIFT '+n)
    spec = json.loads((run/'RUN_SPEC.json').read_text())
    require(sha(run/'sensor.yaml')==spec['sensor_config_sha'], 'SENSOR_CONFIG_DRIFT')
    require(json.loads((run/'LAUNCHER_EXIT.json').read_text())['returncode']==0, 'LAUNCHER_EXIT')
    result = json.loads((run/'RESULT.json').read_text())
    exit_record = json.loads((run/'PROCESS_EXIT.json').read_text())
    require(exit_record['bridge_exit']==0 and exit_record['sensor_exited_early'] is False and exit_record['resource_violation'] is False, 'PROCESS_EXIT')
    require(result['status']=='RUNTIME_COMPLETE' and not result['control'], 'RESULT')
    events = [json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    frames = [x for x in events if x['type']=='FRAME']
    topology = [x for x in events if x['type']=='TOPOLOGY']
    require(len(frames)>=50 and len(topology)>=2, 'INSUFFICIENT_LIVE_EVIDENCE')
    require(not any(x['type']=='FATAL' for x in events), 'FATAL')
    for x in topology:
        pubs = [(t, ['/query_minco_shadow']) for t in x['shadow_publishers']]
        pubs += list(x['control_publishers'].items()) + [('/sim/odom',x['odom_publishers'])]
        topology_check(pubs)
    pair=FrozenPair(spec['frozen_models_sha'])
    start=json.loads((run/'START.json').read_text())
    require(start['mode']=='SHADOW_ONLY' and start['models']==pair.bindings and not start['control'] and not start['training'], 'START_IDENTITY')
    stamps=[]
    fingerprints=set()
    for x in frames:
        require(x['selected_query'] is None and x['online_safe_status'].startswith('UNKNOWN'), 'SELECTOR_OR_SAFETY_CLAIM')
        require(0 <= x['input_age_s'] <= .25 and 0 <= x['output_age_s'] <= .25, 'AGE')
        with np.load(run/('frame_%06d.npz'%x['index']), allow_pickle=False) as z:
            InferenceAdapter.inputs(z['depth_input'][None],z['state'][None],None)
            require(abs(np.linalg.norm(z['state'][6:9])-2.4)<1e-5, 'LOCAL_GOAL')
            require(float(z['stamp'])==x['stamp'] and int(z['stamp_ns'])==x['stamp_ns'], 'STAMP_LOG')
            for label in ['C1','E3']:
                require(z[label+'_raw'].shape==(6,21) and z[label+'_coefficients'].shape==(6,3,3,6), 'OUTPUT_SHAPE')
                require(z[label+'_solve_valid'].all(), 'SOLVE')
                d=z[label+'_durations']
                require(d.shape==(6,3) and (d>=.12-1e-6).all() and (d.sum(-1)>=.8-1e-6).all() and (d.sum(-1)<=2.2+1e-6).all(), 'HORIZON')
                head=np.stack([np.zeros(3),z['state'][:3],z['state'][3:6]])
                require(np.max(np.abs(z[label+'_head_pva']-head))<1e-5,'HEAD_PVA')
            require(all(np.isfinite(z[k]).all() for k in z.files), 'FINITE')
            if x is frames[0] or x is frames[-1]:
                rebuilt=pair.forward(z['depth_input'],z['state'])
                for label,data in rebuilt.items():
                    for k,v in data.items():
                        require(np.array_equal(v,z[label+'_'+k]),'CPU_REBUILD '+label+' '+k)
            fingerprints.add(__import__('hashlib').sha256(z['depth_input'].tobytes()).hexdigest())
        stamps.append(x['stamp_ns'])
    require(np.all(np.diff(stamps)>0) and stamps[-1]-stamps[0]>=9_000_000_000, 'TIMELINE')
    require(len(fingerprints)>=3, 'NO_CHANGING_SENSOR_EVIDENCE')
    require(result['counters']['accepted']==len(frames)<=300, 'FRAME_ACCOUNTING')
    report=dict(status='PASS_CPU_CHECKER_FIXTURE_ONLY' if start.get('CPU_ONLY_SYNTHETIC_TRANSPORT') else 'PASS_SHADOW_RUNTIME_ONLY', frames=len(frames), distinct_depths=len(fingerprints),
        sensor_span_s=(stamps[-1]-stamps[0])/1e9,
        output_age_p50_p95_s=np.quantile([x['output_age_s'] for x in frames],[.5,.95]).tolist(),
        generation_p50_p95_s=np.quantile([x['generation_s'] for x in frames],[.5,.95]).tolist(),
        rejects=dict(collections.Counter(x['reason'] for x in events if x['type']=='REJECT')),
        counters=result['counters'], F3='BLOCKED', control=False)
    # Re-running the check is read-only; it never edits old receipts.
    seal=run/'SHADOW_RUN_SEAL.json'
    if not seal.exists():
        dump(seal, dict(files={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*'))
                              if p.is_file() and p.name!='SHADOW_RUN_SEAL.json'}))
    else:
        sealed=json.loads(seal.read_text())['files']
        require(set(sealed)=={str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name!='SHADOW_RUN_SEAL.json'}, 'RUN_SEAL_MEMBERSHIP')
        for n,h in sealed.items():
            require(sha(run/n)==h, 'RUN_SEAL_DRIFT '+n)
    print(json.dumps(report, indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','check'])
    p.add_argument('--run',required=True)
    a=p.parse_args()
    run=Path(a.run).resolve()
    (prepare if a.action=='prepare' else check)(run)

if __name__=='__main__':
    main()
