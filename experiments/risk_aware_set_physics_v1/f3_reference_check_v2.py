"""Read-only native bench metrics; never upgrades CPU results to ROS success."""
import json,hashlib,collections
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution002';R=OUT/'native_ref001'
def lines(n):return list(map(json.loads,(R/n).read_text().splitlines()))
def quant(x):return np.quantile(x,[.5,.95,1]).tolist() if len(x) else None
def main():
    for p,h in json.loads((OUT/'NATIVE_EXECUTION_BINDING_v2.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    scene=json.loads((R/'SCENE_READY.json').read_text());expected=json.loads((OUT/'scene/SCENE_VALIDATION.json').read_text())
    assert scene['loaded_map_quantised_sha256']==expected['quantised_xyz_sha256'] and scene['point_count']==24802
    with np.load(R/'LOADED_MAP.npz') as z:
        assert hashlib.sha256(z['quantised'].astype('<i4').tobytes()).hexdigest()==expected['quantised_xyz_sha256']
    events=lines('ARBITRATION.jsonl');cmd=lines('COMMAND.jsonl');odom=lines('ODOM.jsonl');so3=lines('SO3.jsonl');imu=lines('IMU.jsonl')
    camera=lines('CAMERA_TRACE.jsonl')
    assert not any(x.get('type')=='FATAL' for x in camera)
    assert [x['type'] for x in camera if 'type' in x][-4:]==['QUIESCE_BEGIN','CALLBACKS_DRAINED','PUBLISHERS_CLOSED','EXIT']
    topo=[e for e in events if e['type']=='TOPOLOGY']
    assert all(dict(e['publishers']).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'] for e in topo)
    assert all(not any('/yopo_net' in ns or '/query_minco_shadow' in ns for _,ns in e['publishers']) for e in topo)
    assert len(cmd)==json.loads((R/'ARBITER_RESULT.json').read_text())['commands']
    stamps=np.array([e['stamp_ns']/1e9 for e in cmd]);assert (np.diff(stamps)>0).all()
    assert [e['seq'] for e in cmd]==list(range(len(cmd)))
    installs=[e for e in events if e['type']=='INSTALL']
    nominal=installs[0];activation=nominal['activation']
    dwell=next((e for e in events if e['type']=='NOMINAL_STOP_DWELL_PASS'),None)
    trigger=next((e for e in events if e['type']=='WATCHDOG_TRIGGER'),None)
    abort=next((e for e in events if e['type']=='ABORT'),None)
    plans={}
    h0=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float);h1=np.array([[0,0,2],[.8,0,0],[0,0,0]],float)
    plans['NOMINAL_REFERENCE_3M']=concatenate(quintic(h0,h1,4),terminal(h1,[3,0,2]))
    plans['WATCHDOG_REFERENCE']=terminal(np.array([[3,0,2],[0,0,0],[0,0,0]],float),[0,0,2])
    if trigger:plans['WATCHDOG_BRAKE']=brake(np.array(trigger['brake_head']))[0]
    if abort:
        a=next(e for e in installs if e['tag']=='ABORT_BRAKE')
        wb=next(e for e in installs if e['tag']=='WATCHDOG_BRAKE')
        plans['ABORT_BRAKE']=brake(plans['WATCHDOG_BRAKE'].pva(a['activation']-wb['activation']))[0]
    for i in installs:
        p=plans[i['tag']];p.validate(MotionLimits())
        assert p.digest==i['certificate']['trajectory_sha256']
        assert (np.array(i['pva_gap'])<=[.0002,.002,.01]).all()
        v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],i['certificate']['valid_until'],'verified-room'),),Scope.EXTERNAL_SUPERVISOR)
        assert v.certify(p.coefficients,p.durations,i['certificate']['checked_at'],required_until=i['activation']+p.duration)['status']=='CERTIFIED_FREE'
    activations=[]
    for i in installs:
        first=next((c for c in cmd if c['phase']==i['tag'] and c['stamp_ns']/1e9>=i['activation']),None)
        if first:activations.append(dict(tag=i['tag'],requested=i['activation'],first_published=first['stamp_ns']/1e9,lateness_ms=(first['stamp_ns']/1e9-i['activation'])*1000))
    stage={}
    for tag in plans:
        rr=[e for e in odom if e['phase']==tag]
        err=np.array([e['errors'] for e in rr])
        stage[tag]=dict(samples=len(rr),errors_norm_p50_p95_max=np.quantile(err,[.5,.95,1],axis=0).tolist() if len(rr) else None,
            samples_exceeding_tracking_budget=int(np.any(err>[.02,.05,.1],axis=1).sum()) if len(rr) else 0,
            actual_speed_p50_p95_max=quant([np.linalg.norm(e['v']) for e in rr]),
            actual_acceleration_p50_p95_max=quant([np.linalg.norm(e['a_derived']) for e in rr]))
    nominal_rows=[e for e in odom if e['phase']=='NOMINAL_REFERENCE_3M']
    # Exact before/after commanded and measured stop facts.
    wake=[e for e in odom if e['stamp_ns']/1e9>=activation+4]
    scored_start=min(wake,key=lambda e:abs(e['stamp_ns']/1e9-(activation+4)))
    latencies=[]
    for e in so3:
        t=e['stamp_ns']/1e9
        if t<activation:continue
        idx=np.searchsorted(stamps,t,side='right')-1
        if idx>=0 and 0<=t-stamps[idx]<=.04:latencies.append((t-stamps[idx])*1000)
    # Estimate physical acceleration phase lag on the watchdog/abort interval.
    # This is retrospective correlation, not an acknowledged actuator timestamp.
    aa=[e for e in odom if trigger and trigger['activation']<=e['stamp_ns']/1e9<odom[-1]['stamp_ns']/1e9]
    cmd_a=np.array([np.array(e['pva'])[2,0] for e in cmd])
    lagfits=[]
    for lag in np.arange(0,.1001,.005):
        ids=np.searchsorted(stamps,np.array([e['stamp_ns']/1e9 for e in aa])-lag,side='right')-1;ids=np.maximum(ids,0)
        err=np.array([e['a_derived'][0] for e in aa])-cmd_a[ids]
        lagfits.append(dict(lag_s=float(lag),rms=float(np.sqrt(np.mean(err**2)))))
    best=min(lagfits,key=lambda e:e['rms']) if lagfits else None
    res=lines('RESOURCE_TRACE.jsonl');he=lines('HOST_EVENTS.jsonl')
    report=dict(status='FAIL_CONTROLLER_CONTRACT',Query_control=False,nominal_runtime_stop_dwell_event=dwell,
        nominal_strict_raw_tracking_pass=stage['NOMINAL_REFERENCE_3M']['samples_exceeding_tracking_budget']==0,
        nominal_scored_start_actual=scored_start,nominal_goal_stop=dwell,
        phases=stage,command_count=len(cmd),command_gap_p50_p95_max_ms=quant(np.diff(stamps)*1000),
        effective_hz=float((len(cmd)-1)/(stamps[-1]-stamps[0])),
        activations=activations,source_to_nearest_prior_command_SO3_publish_ms=quant(latencies),
        latency_limitation='SO3 has no source trajectory_id echo; this is temporal association, not a per-command acknowledgement or motor response proof',
        watchdog=trigger,abort=abort,last_actual_state=odom[-1],
        watchdog_detection_delay_s=None if trigger is None else trigger['detection']-trigger['last_heartbeat'],
        watchdog_acceleration_phase_lag_estimate=best,lag_estimate_is_not_causal_proof=True,
        imu_zero_stamp_count=sum(e['stamp_ns']==0 for e in imu),imu_count=len(imu),
        topology_checks=len(topo),unique_publisher=True,scene_verified=True,
        resources=dict(WSL_min_available_GiB=min(x['MemAvailable_kb'] for x in res)/1048576,
            Windows_min_available_GiB=min(x['available_kb'] for x in he if x['type']=='WINDOWS_AVAILABLE')/1048576,
            GPU_peak_MiB=max(x['gpu']['used_mib'] for x in res),Swap_peak_kB=max(x['SwapUsed_kb'] for x in res)),
        cleanup_remaining=json.loads((R/'CLEANUP.json').read_text())['remaining'],
        no_retry_with_relaxed_contract=True)
    with (OUT/'NATIVE_MEASURED_RESULTS.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['nominal_runtime_stop_dwell_event','nominal_goal_stop','nominal_scored_start_actual','last_actual_state','watchdog','abort']},indent=2))
if __name__=='__main__':main()
