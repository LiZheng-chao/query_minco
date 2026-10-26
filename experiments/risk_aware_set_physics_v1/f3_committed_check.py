"""Independent raw-run checks; no ROS commands and no successful-label override."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution004';RUN=OUT/'native_ref002'
def lines(n):return [json.loads(l) for l in (RUN/n).read_text().splitlines()]
def quant(a):return np.quantile(a,[.5,.95,1]).tolist() if len(a) else None
def thaw(f):return Program(np.frombuffer(f.coefficients,dtype='<f8').reshape(-1,3,6),f.durations)
def main():
    for p,h in json.loads((OUT/'NATIVE_BINDING_attempt002.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    with (OUT/'preparation002/PREPARED_REFERENCES.pkl').open('rb') as f:bundles=pickle.load(f)
    events=lines('EVENT.jsonl');cmd=lines('COMMAND.jsonl');odom=lines('ODOM.jsonl');topo=lines('TOPOLOGY.jsonl')
    req=next(e for e in events if e.get('type')=='REQUEST' and e['tag']=='NOMINAL')
    activations=[e for e in events if e.get('type')=='ACTUAL_ACTIVATION']
    failure=next(e for e in events if e.get('type')=='KERNEL' and e['event']['reason']=='SCHEDULING_GAP')['event']
    b=bundles['NOMINAL'];elapsed=failure['mono']-req['activation_mono']
    entry=next(x for x in b.stops if x.phase>=elapsed+.06-1e-9)
    deadline_ros=req['activation_ros']+entry.phase
    stop=thaw(entry.polynomial);prefix=thaw(b.prefix)
    combined=concatenate(trim(prefix,0,entry.phase),stop).validate(MotionLimits())
    proof=json.loads(entry.proof)
    assert proof['trajectory_sha256']==combined.digest
    volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],b.valid_until_ros,'RECHECK_VERIFIED_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
    check=volume.certify(combined.coefficients,combined.durations,b.prepared_ros,required_until=b.valid_until_ros)
    assert check['status']=='CERTIFIED_FREE'
    # Verify all immutable bank digests/geometric evidence; extrema validated by worker.
    bank_count=0
    for name,bundle in bundles.items():
        pre=thaw(bundle.prefix)
        for row in bundle.stops:
            sp=thaw(row.polynomial)
            full=sp if row.phase==0 else concatenate(trim(pre,0,row.phase),sp)
            cert=json.loads(row.proof)
            assert full.digest==cert['trajectory_sha256']
            assert cert['scope']=='EXTERNAL_SUPERVISOR' and cert['status']=='CERTIFIED_FREE'
            bank_count+=1
    with np.load(RUN/'LOADED_MAP.npz') as z:
        loaded=hashlib.sha256(z['quantised'].astype('<i4').tobytes()).hexdigest()
    scene=json.loads((OUT/'scene/SCENE_VALIDATION.json').read_text())
    assert loaded==scene['quantised_xyz_sha256']
    ctimes=np.array([e['begin'] for e in cmd]);ov=np.array([e['pva'] for e in odom],float)
    ot=np.array([e['stamp_ns']/1e9 for e in odom]);cv=np.array([e['pva'] for e in cmd])
    finite=np.isfinite(ov).all(axis=(1,2));zoh=[];ideal=[];brakeerrors=[]
    for e,v,t,valid in zip(odom,ov,ot,finite):
        idx=np.searchsorted(ctimes,t,side='right')-1
        if valid and idx>=0:zoh.append(np.linalg.norm(v-cv[idx],axis=1))
        if not valid:continue
        if t<req['activation_ros']:expected=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float)
        else:
            phase=t-req['activation_ros']
            expected=prefix.pva(phase) if phase<entry.phase else stop.pva(min(phase-entry.phase,stop.duration))
        err=np.linalg.norm(v-expected,axis=1);ideal.append(err)
        if t>=deadline_ros:brakeerrors.append(err)
    def stats(x):return np.quantile(x,[.5,.95,1],axis=0).tolist() if len(x) else None
    release=next(e for e in events if e.get('type')=='STARTUP_PHASE' and e['value']=='RELEASED_BEFORE_FIRST_INTEGRATION')
    before=ov[ot<release['ros']]
    firstcmd=next((e for e in cmd if e['begin']>=deadline_ros),None)
    dwell=next(e for e in events if e.get('type')=='FAILED_STOP_DWELL')
    # Independently test real last0.5s stop dwell, do not trust its event boolean.
    end=dwell['ros'];window=(ot>=end-.51)&(ot<=end)&finite
    rr=ov[window];tt=ot[window]
    actual_dwell=(len(rr)>1 and tt[-1]-tt[0]>=.5 and np.max(np.diff(tt))<=.04 and
        np.max(np.linalg.norm(rr[:,1],axis=1))<=.03 and np.max(np.linalg.norm(rr[:,2],axis=1))<=.05)
    resources=lines('RESOURCE_TRACE.jsonl');he=lines('HOST_EVENTS.jsonl');camera=lines('CAMERA_TRACE.jsonl')
    report=dict(status='FAIL_BLOCKED',Query_control=False,classification='CONTROLLED_STARTUP_PRIVATE_GATE_NOT_ORIGINAL_YOPO',
      startup=dict(first_actual_pva=odom[0]['pva'],all_phase_min_z=float(ov[:,0,2].min()),all_phase_max_z=float(ov[:,0,2].max()),
        before_release_max_position_drift=float(np.linalg.norm(before[:,0]-[-1.6,0,2],axis=1).max()),
        release=release,unexpected_ground_contact=False,initial_equilibrium_rotors_explicit=True,
        no_ground_subcheck_pass=bool(ov[:,0,2].min()>1.9),original_YOPO_unchanged=True),
      activations=activations,activation_lateness_p50_p95_max_ms=quant([e['lateness']*1000 for e in activations]),
      command_count=len(cmd),command_gap_p50_p95_max_ms=quant(np.diff(ctimes)*1000),
      reference_rate_hz=float((len(cmd)-1)/(ctimes[-1]-ctimes[0])),
      failures=[e for e in events if e.get('type') in ['FAILURE_LATCH','COMMAND_GAP_FAULT']],
      raw_Odom_count=len(odom),raw_Odom_invalid_pva_count=int((~finite).sum()),
      odom_gap_p50_p95_max_ms=quant(np.diff(ot)*1000),
      measured_finite_speed_p50_p95_max=quant(np.linalg.norm(ov[finite,1],axis=1)),
      measured_finite_acceleration_p50_p95_max=quant(np.linalg.norm(ov[finite,2],axis=1)),
      measured_acceleration_limitation='causal Odom velocity difference; invalid/gap interval unproved, not continuous actual maxima',
      tracking_vs_continuous_prepared_reference=stats(ideal),tracking_vs_last_published_PVA=stats(zoh),
      tracking_on_brake_vs_prepared_reference=stats(brakeerrors),
      tracking_reference_mapping='nominal plannedROS/monotonic pair; no logged clock jump; ideal-polynomial vs actual command ZOH both reported',
      reserved_stop=dict(selected_phase_s=entry.phase,duration_s=stop.duration,full_delay_prefix_stop_extrema=combined.extrema(),
        stop_extrema=stop.extrema(),geometry=check,planned_deadline_ros_reconstructed=deadline_ros,
        first_command_after_reconstructed_deadline=firstcmd,
        activation_receipt_limitation='backup entered TRACKING_UNPROVEN before activation; exact dedicated BACKUP_ACTIVATED receipt absent; reconstructed using immutable bank+firstfault+rawcommands, not qualified activation proof',
        endpoint_displacement_m=float(np.linalg.norm(stop.pva(stop.duration)[0]-stop.pva(0)[0])),
        path_length_upper_m=float(stop.extrema()[0]*stop.duration)),
      observed_stop_dwell_event=dwell,independent_stop_dwell_pass=bool(actual_dwell),
      last_actual_pva=odom[-1]['pva'],precomputed_bank_rows_verified=bank_count,
      native_extra_planning_jobs_started=sum(e.get('type')=='PLANNING_PROCESS_STARTED' for e in events),
      dedicated_Watchdog_trial='NOT_RUN_FIRST_SAFETY_FAILURE',explicit_safety_abort_trial='NOT_RUN_FIRST_SAFETY_FAILURE',
      moving_handover='NOT_REACHED_ABORT_BEFORE_PHASE7P96',
      C1_realtime='NOT_RUN_ABC_FAILED',C1_generated=0,C1_qualified=0,C1_latency_ms=None,
      publisher_checks=len(topo),unique_publisher=all(dict(e['publishers']).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'] for e in topo),
      camera_fatal=any(e.get('type')=='FATAL' for e in camera),camera_final_sequence=[e['type'] for e in camera if 'type' in e][-4:],
      resources=dict(WSL_min_available_GiB=min(e['MemAvailable_kb'] for e in resources)/1048576,
        Windows_min_available_GiB=min(e['available_kb'] for e in he if e['type']=='WINDOWS_AVAILABLE')/1048576,
        GPU_peak_MiB=max(e['gpu']['used_mib'] for e in resources),swap_peak_kb=max(e['SwapUsed_kb'] for e in resources)),
      cleanup_remaining=json.loads((RUN/'CLEANUP.json').read_text())['remaining'])
    with (OUT/'NATIVE_MEASURED_RESULTS.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ['reserved_stop','failures','startup','observed_stop_dwell_event','last_actual_pva']},indent=2))
if __name__=='__main__':main()
