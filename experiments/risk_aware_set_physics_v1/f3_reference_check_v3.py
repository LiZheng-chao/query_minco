"""Independent read-only checker for actual V3 bench; no ROS publication."""
import json,hashlib
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution003'
R=OUT/'native_ref002'
def rows(n):return [json.loads(x) for x in (R/n).read_text().splitlines()]
def q(x):return np.quantile(x,[.5,.95,1]).tolist() if len(x) else None
def main():
    binding=json.loads((OUT/'NATIVE_EXECUTION_BINDING_attempt002.json').read_text())
    for p,h in binding.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    ev=rows('ARBITRATION.jsonl');cmd=rows('COMMAND.jsonl');odom=rows('ODOM.jsonl');imu=rows('IMU.jsonl')
    topo=[e for e in ev if e['type']=='TOPOLOGY'];install=[e for e in ev if e['type']=='INSTALL']
    act=[e for e in ev if e['type']=='ACTIVATED'];inject=[e for e in ev if e['type']=='INJECTED_ACTIVATION_REJECT']
    expected={'SCHEDULING_DELAY':'LATE_ACTIVATION','CLOCK_JUMP':'CLOCK_JUMP','DROPPED_ODOM':'STALE_ODOM','PUBLISHER_CONFLICT':'PUBLISHER_CONFLICT'}
    injected={e['probe']:dict(expected=expected[e['probe']],actual=e['reason'],passed=e['reason']==expected[e['probe']],actual_ros=e['actual_ros'],planned=e['planned']) for e in inject}
    assert len(injected)==4 and all(e['passed'] for e in injected.values())
    plans=[]
    for e in install:
        p=Program(e['coefficients'],e['durations']).validate(MotionLimits())
        assert p.digest==e['certificate']['trajectory_sha256']
        vol=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],e['certificate']['valid_until'],'recheck'),),Scope.EXTERNAL_SUPERVISOR)
        proof=vol.certify(p.coefficients,p.durations,e['certificate']['checked_at'],required_until=e['activation']+p.duration)
        assert proof['status']=='CERTIFIED_FREE'
        plans.append(dict(tag=e['tag'],duration=p.duration,extrema=p.extrema(),certified_scope='EXTERNAL_SUPERVISOR',actually_activated=any(a['phase']==e['tag'] for a in act),reference_head=p.pva(0).tolist()))
    scene=json.loads((R/'SCENE_READY.json').read_text());validation=json.loads((OUT/'scene/SCENE_VALIDATION.json').read_text())
    with np.load(R/'LOADED_MAP.npz') as z:
        assert hashlib.sha256(z['quantised'].astype('<i4').tobytes()).hexdigest()==validation['quantised_xyz_sha256']
    assert scene['loaded_map_quantised_sha256']==validation['quantised_xyz_sha256']
    assert len(cmd)==json.loads((R/'ARBITER_RESULT.json').read_text())['commands']
    phases={}
    for tag in sorted(set(e['phase'] for e in odom)):
        rr=[e for e in odom if e['phase']==tag]
        err=np.array([e['errors'] for e in rr])
        phases[tag]=dict(samples=len(rr),tracking_error_norm_p50_p95_max=np.quantile(err,[.5,.95,1],axis=0).tolist(),
            raw_exceed_tracking_count=int(np.any(err>[.02,.05,.1],axis=1).sum()),actual_speed_p50_p95_max=q([np.linalg.norm(e['v']) for e in rr]),
            actual_acceleration_p50_p95_max=q([np.linalg.norm(e['a_derived']) for e in rr]))
    stamps=np.array([e['stamp_ns']/1e9 for e in cmd]);begins=np.array([e['publish_begin'] for e in cmd])
    trigger=next(e for e in ev if e['type']=='WATCHDOG_TRIGGER')
    ins=next(e for e in install if e['tag']=='WATCHDOG_BRAKE')
    previous=next(e for e in install if e['tag']=='WATCHDOG_REFERENCE')
    # Lower bound on actual command starvation before failing publish attempt:
    # trigger event was logged after all search/certification, before failed publish.
    trigger_log_ros_approx=trigger['received_wall_ns']/1e9
    lastcmd=cmd[-1]
    stall=dict(detection_ros=trigger['detection'],search_proof_install_complete_ros=ins['request_stamp'],
        search_proof_install_total_ms=(ins['request_stamp']-trigger['detection'])*1000,
        last_actual_publish_begin=lastcmd['publish_begin'],trigger_log_wall_seconds=trigger_log_ros_approx,
        starvation_lower_bound_ms=(trigger_log_ros_approx-lastcmd['publish_begin'])*1000,
        wall_vs_ros_note='ROS uses wall clock without /clock; bound uses event receive_wall; no exact failing-publish timestamp was persisted')
    nominal=next(e for e in install if e['tag']=='NOMINAL_REFERENCE_3M')
    # Stable moving cruise before crossing x=0; evaluate actual state only.
    stable=[];current=[]
    for e in odom:
        t=e['stamp_ns']/1e9;pos=np.array(e['p']);v=np.array(e['v']);acc=np.array(e['a_derived'])
        qxyzw=np.array(e['q']);tilt=float(np.arccos(np.clip(1-2*(qxyzw[0]**2+qxyzw[1]**2),-1,1)))
        global_delta=np.array([3,0,2])-pos
        x,y,z,w=qxyzw
        rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        goal=rot.T@global_delta
        direction=goal[0]>0 and np.max(np.abs(goal[1:]/goal[0]))<=.175
        prelude=nominal['activation']+2.5<=t<=nominal['activation']+3.25
        ok=prelude and np.linalg.norm(v-[.8,0,0])<=.03 and np.linalg.norm(acc)<=.05 and tilt<=np.deg2rad(.5) and direction and np.linalg.norm(global_delta)>=2.4
        item=dict(stamp=t,p=e['p'],v=e['v'],a=e['a_derived'],tilt_deg=float(np.rad2deg(tilt)),goal_B0=goal.tolist())
        if ok and (not current or t-current[-1]['stamp']<=.04):current.append(item)
        else:
            if current:stable.append(current)
            current=[item] if ok else []
    if current:stable.append(current)
    longest=max(stable,key=lambda x:x[-1]['stamp']-x[0]['stamp']) if stable else []
    crossing=min((e for e in odom if e['phase']=='NOMINAL_REFERENCE_3M'),key=lambda e:abs(e['p'][0]))
    cam=rows('CAMERA_TRACE.jsonl');cleanup=json.loads((R/'CLEANUP.json').read_text())
    res=rows('RESOURCE_TRACE.jsonl');he=rows('HOST_EVENTS.jsonl')
    ar=json.loads((R/'ARBITER_RESULT.json').read_text())
    report=dict(status='FAIL_BLOCKED',Query_control=False,raw_arbiter_status=ar['status'],exception=ar.get('traceback'),
        activation_normal_events=act,normal_first_publish_lateness_ms=q([a['lateness_s']*1000 for a in act]),
        normal_activations_subcheck_pass=len(act)==2 and all(0<=a['lateness_s']<=.01 for a in act),
        fault_input_rejections=injected,injection_scope='input fault gates only; no actual second ROS publisher or modified system/ROS clock',
        activation_full_qualification_pass=False,reason='watchdog search stalls publisher >40ms before brake activation',
        command_count=len(cmd),recorded_command_gap_p50_p95_max_ms=q(np.diff(begins)*1000),
        recorded_effective_hz=float((len(cmd)-1)/(begins[-1]-begins[0])),starvation=stall,
        planned_programs=plans,watchdog_trigger=trigger,watchdog_brake_actually_executed=False,
        watchdog_braking_tracking_measured=False,safety_abort_trial='NOT_RUN_AFTER_FIRST_PHYSICAL_CONTRACT_FAILURE',
        phases=phases,nominal_stop_event=next(e for e in ev if e['type']=='NOMINAL_STOP_DWELL_PASS'),
        measured_cruise_window=dict(duration_s=0 if not longest else longest[-1]['stamp']-longest[0]['stamp'],samples=len(longest),first=None if not longest else longest[0],last=None if not longest else longest[-1]),
        nearest_x0_actual=crossing,literal_fixed_position_handover_pass=False,
        handover_semantics='0.8+/-0.03m/s for0.5s implies >=0.385m displacement >0.04m fixed-position tolerance width; separate moving-state diagnostic is not v1 literal PASS',
        topology_checks=len(topo),unique_publisher=all(dict(e['publishers']).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'] for e in topo),
        no_Query_or_YOPO_control=all(not any('/yopo_net' in ns or '/query_minco_shadow' in ns for _,ns in e['publishers']) for e in topo),
        camera_fatal=any(e.get('type')=='FATAL' for e in cam),camera_final_sequence=[e['type'] for e in cam if 'type'in e][-4:],
        imu_zero_stamp_count=sum(e['stamp_ns']==0 for e in imu),imu_count=len(imu),
        resources=dict(WSL_min_available_GiB=min(x['MemAvailable_kb'] for x in res)/1048576,
            Windows_min_available_GiB=min(x['available_kb'] for x in he if x['type']=='WINDOWS_AVAILABLE')/1048576,
            GPU_peak_MiB=max(x['gpu']['used_mib'] for x in res),Swap_peak_kB=max(x['SwapUsed_kb'] for x in res)),
        cleanup_remaining=cleanup['remaining'],C1_pipeline='NOT_RUN_AB_GATE_FAILED',C1_live_qualified_prefixes=0,C1_candidates_generated=0,C1_latency_ms=None)
    with (OUT/'NATIVE_MEASURED_RESULTS.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['phases','nearest_x0_actual','watchdog_trigger','planned_programs']},indent=2))
if __name__=='__main__':main()
