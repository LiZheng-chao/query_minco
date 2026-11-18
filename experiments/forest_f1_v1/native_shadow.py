"""Offline native-input K1 Shadow. No ROS Publisher or actuator interface."""
from pathlib import Path
import json,time,types,hashlib
import numpy as np
import torch
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation,sample_world
from query_minco.audit.k1_signal_diagnostics_v1 import model_at
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import norm_max
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared,sample_batch
from query_minco.integration.observed_space_v1.map import Config
from query_minco.integration.depth_minco_t2_v1.space import Space,authorized_snapshot
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/query_k1_native_shadow_20261118_v1'
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def evaluate_frame(plan,snapshot,now):
    result=dict(dynamics_maxima=plan.maxima,dynamics_pass=all(x<=y+1e-7 for x,y in zip(plan.maxima,[1,8,100])))
    try:result['certificate']=snapshot.certify(Prepared(plan),now);result['geometry_pass']=True
    except Exception as ex:result.update(geometry_pass=False,reason=str(ex),details=getattr(ex,'record',None))
    values=sample_batch(plan,np.linspace(0,2,129));points=values[:,0]
    result['world_pva_samples']=values.tolist()
    result['world_samples']=points.tolist();result['endpoint_world']=points[-1].tolist()
    # This is observed point-hit distance, not a GT or full-cell clearance certificate.
    if snapshot.frames:result['observed_clearance']='See full swept-tube certification; GT clearance NOT_AVAILABLE'
    return result

def run_shadow(pairs,depth_probes,selected,grant):
    OUT.mkdir();(OUT/'frames').mkdir();model,identity=model_at(ROOT,128)
    identity.update(label='HISTORICAL_PILOT_K1_STEP128_NOT_CURRENT_FORMAL',K6='NOT_AVAILABLE',device='cpu')
    write(OUT/'K1_CHECKPOINT_IDENTITY.json',identity)
    dec=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);minco=MincoAdapter(num_candidates=1)
    run=ROOT/'research_outputs/f0_controller_consumption_20261118_v1/runs/F0_20261118_104715_1163/none'
    goal=np.asarray(json.loads((run/'reference_data/STATUS.json').read_text())['target'])
    ledger=ObservedMap(Config(shape=(32,24,16)));rows=[];arrays={}
    for index,p in enumerate(pairs):
        start=time.perf_counter();row=dict(stamp=p['stamp'],source=p['path'],checkpoint_sha=identity['file_sha256'],goal=goal.tolist(),goal_source='recorded F0 Reference STATUS; fixed goal',acceleration_source='exact-stamp native IMU a_world',depth_odom_delta_s=0.,depth_imu_delta_s=0.,timing={})
        try:
            dep=np.load(p['path'])['depth'];state=p['pose'];now=p['stamp']
            x,s,R=observation(dep,state['q'],state['v'],p['imu']['a_world'],state['p'],goal,[80.,0,80.,0,80.,45.,0,0,1.])
            row.update(depth_shape=list(x.shape),state_shape=list(s.shape),state12=s.tolist(),frame='world / current-body FLU / camera optical-Z')
            t=time.perf_counter()
            with torch.no_grad():raw=model(torch.from_numpy(x),torch.from_numpy(s))
            row['timing']['forward_s']=time.perf_counter()-t;t=time.perf_counter();params=dec(raw,torch.from_numpy(s));row['timing']['decode_s']=time.perf_counter()-t;t=time.perf_counter()
            tr=minco(params.head_pva,params.tail_pva,params.inner_pts,params.durations);row['timing']['minco_s']=time.perf_counter()-t
            c=tr.coeffs[0,0].detach().numpy();d=tr.durations[0,0].detach().numpy();row['raw21']=raw.tolist();row['solve_valid']=bool(tr.solve_valid.all())
            if not row['solve_valid'] or not np.isfinite(c).all():raise ValueError('NETWORK_MINCO_INVALID')
            plan=types.SimpleNamespace(c=c,d=d,origin=np.asarray(state['p']),R=R,id=hashlib.sha256(c.tobytes()+d.tobytes()).hexdigest())
            plan.sample=lambda t:sample_world(c,d,plan.origin,R,t)
            plan.maxima=[max(norm_max(a,float(t),k) for a,t in zip(c,d)) for k in [1,2,3]]
            t=time.perf_counter();ledger.add(dep,state,now,now,now);sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,grant(now),now,sensor.epoch,lambda:sensor.epoch)
            row['evaluation']=evaluate_frame(plan,snap,now);
            from query_minco.integration.observed_space_v1.map import Frame
            hits=Frame(dep,state,now,now,now,'native_static_scene','native_camera')
            pts=np.asarray(row['evaluation']['world_samples'])
            row['evaluation']['nearest_observed_hit_distance_m']=float(hits.hit_tree.query(pts)[0].min()) if hits.hit_tree is not None else None
            row['evaluation']['clearance_scope']='DISTANCE_TO_MEASURED_HITS_ONLY_NOT_FREE_SPACE_CERTIFICATE'
            row['timing']['evaluation_s']=time.perf_counter()-t
            row['coefficients']=c.tolist();row['durations']=d.tolist();end=np.asarray(row['evaluation']['endpoint_world']);row['progress_m']=float(np.linalg.norm(goal-plan.origin)-np.linalg.norm(goal-end));row['feasible']=row['evaluation']['geometry_pass'] and row['evaluation']['dynamics_pass']
            # Same sensor, state, actual recorded goal and same safety contract.
            pair_tick=time.perf_counter();clock=lambda:now+(time.perf_counter()-pair_tick)
            try:
                dp,proof=DepthPlanner(clock,space=Space(snap,clock)).propose(dict(state,a=p['imu']['a_world']),goal,now)
                row['paired_depth_minco']=dict(status='CERTIFIED',plan=dp.record,maxima=dp.maxima,proof=proof,stage_times=dp._planning_profile)
            except Exception as ex:row['paired_depth_minco']=dict(status='REJECTED',reason=str(ex))
            row['status']='TRAJECTORY_PRODUCED';arrays['frame%03d_raw21'%index]=raw.detach().numpy();arrays['frame%03d_coeff'%index]=c
        except Exception as ex:row.update(status='REJECTED',reason=str(ex),feasible=False)
        row['timing']['end_to_end_s']=time.perf_counter()-start;rows.append(row);write(OUT/'frames'/('frame_%03d.json'%index),row)
    np.savez_compressed(OUT/'SHADOW_ARRAYS.npz',**arrays)
    result=dict(status='OFFLINE_NATIVE_INPUT_SHADOW_COMPLETED',checkpoint=identity,frames=len(rows),trajectory_frames=sum(x['status']=='TRAJECTORY_PRODUCED' for x in rows),feasible_candidate_fraction=sum(x['feasible'] for x in rows)/max(1,len(rows)),K6='NOT_AVAILABLE',control_publishers=0,optimizer_updates=0,records=rows,
        limitations=['Recorded native inputs only; not live ROS Shadow','Camera K from frozen contract; CameraInfo raw message unavailable','Historical Pilot K1, not silently promoted to formal model','Offline certificate evaluation is not online execution authorization'])
    write(OUT/'K1_SHADOW_RESULT.json',result);print('K1_SHADOW',result['trajectory_frames'],result['frames'],result['feasible_candidate_fraction'])
