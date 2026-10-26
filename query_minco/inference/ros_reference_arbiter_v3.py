"""V3 NON-QUERY deadline-aware native-controller bench. Single PositionCommand publisher.
No network/checkpoint/candidate input exists. External static room supervision only.
"""
import argparse,collections,json,threading,time,traceback
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,Measurement,quintic,terminal,brake,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.execution_clock_v3 import soft_brake as brake,activation_guard,cruise_segment
def main():
    import rospy,rosgraph
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from quadrotor_msgs.msg import PositionCommand,SO3Command
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference-only',action='store_true',required=True);a=p.parse_args()
    run=Path(a.run);rospy.init_node('f3_reference_arbiter',disable_signals=True)
    logs={k:(run/(k+'.jsonl')).open('x',buffering=1) for k in ['ODOM','IMU','COMMAND','SO3','ARBITRATION']}
    def log(k,x):logs[k].write(json.dumps(dict(received_wall_ns=time.time_ns(),**x),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
    def dump(n,x):
        with (run/n).open('x') as f:json.dump(x,f,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    master=rosgraph.Master('/f3_reference_arbiter')
    assert not dict(master.getSystemState()[0]).get('/so3_control/pos_cmd'), 'PREEXISTING_POSITION_SOURCE'
    pub=rospy.Publisher('/so3_control/pos_cmd',PositionCommand,queue_size=1,tcp_nodelay=True)
    lock=threading.RLock();closing=[False];latest=[None];previous=[None];timeline=[];graph_ok=threading.Event();done=threading.Event()
    error_stop=[];sequence=0;last_tick=None;clock_start=time.monotonic();violations=[];events=[];last_publish_ros=None
    origin=np.array([-1.6,0,2.]);static=np.zeros((1,3,6));static[0,:,0]=origin
    timeline.append((0.,Program(static,[1000000000000.]),'SETTLE'))
    def reference(stamp):
        activation,prog,tag=next(x for x in reversed(timeline) if x[0]<=stamp)
        return prog.pva(float(np.clip(stamp-activation,0,prog.duration))),tag
    def odom_cb(o):
        ns=o.header.stamp.to_nsec();stamp=ns/1e9
        po=o.pose.pose;v=o.twist.twist.linear
        pos=np.array([po.position.x,po.position.y,po.position.z]);vel=np.array([v.x,v.y,v.z])
        with lock:
            prev=previous[0];previous[0]=(stamp,vel.copy())
            acc=np.zeros(3) if prev is None else (vel-prev[1])/(stamp-prev[0]) if stamp>prev[0] else np.full(3,np.nan)
            expected,tag=reference(stamp)
            pva=np.stack([pos,vel,acc]);latest[0]=Measurement(stamp,pva) if np.isfinite(pva).all() else None
        log('ODOM',dict(stamp_ns=ns,p=pos,v=vel,a_derived=acc,q=[po.orientation.x,po.orientation.y,po.orientation.z,po.orientation.w],
            frame=o.header.frame_id,child_frame=o.child_frame_id,expected_pva=expected,phase=tag,
            acceleration_source='past_native_world_velocity_difference',errors=np.linalg.norm(pva-expected,axis=1)))
    def imu_cb(m):
        log('IMU',dict(stamp_ns=m.header.stamp.to_nsec(),a=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],
            note='native header stamp may be0; not assigned a fabricated timestamp; primary acceleration uses stamped Odom difference'))
    def so3_cb(m):
        log('SO3',dict(stamp_ns=m.header.stamp.to_nsec(),force=[m.force.x,m.force.y,m.force.z],
            q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w]))
    def guarded(fn):
        def inner(msg):
            with lock:
                if not closing[0]:fn(msg)
        return inner
    subs=[rospy.Subscriber('/sim/odom',Odometry,guarded(odom_cb),queue_size=10,tcp_nodelay=True),
        rospy.Subscriber('/sim/imu',Imu,guarded(imu_cb),queue_size=10),rospy.Subscriber('/so3_cmd',SO3Command,guarded(so3_cb),queue_size=10,tcp_nodelay=True)]
    def topology():
        while not done.is_set():
            try:
                ps=master.getSystemState()[0];sources=dict(ps).get('/so3_control/pos_cmd',[])
                okay=sources==['/f3_reference_arbiter'] and dict(ps).get('/sim/odom')==['/quadrotor_simulator_so3']
                topology_sources[:]=sources
                if okay:graph_ok.set()
                elif sources and sources!=['/f3_reference_arbiter']:error_stop.append('PUBLISHER_CONFLICT')
                if any('/yopo_net' in ns or '/query_minco_shadow' in ns for _,ns in ps):error_stop.append('FORBIDDEN_PLANNER_PRESENT')
                log('ARBITRATION',dict(type='TOPOLOGY',stamp=rospy.Time.now().to_sec(),publishers=ps,okay=okay,scope='NON_QUERY_REFERENCE_ONLY'))
            except Exception as e:log('ARBITRATION',dict(type='TOPOLOGY_ERROR',error=str(e)))
            done.wait(.2)
    topology_sources=[]
    worker=threading.Thread(target=topology,daemon=True);worker.start()
    volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],rospy.Time.now().to_sec()+120,'RUNTIME_VERIFIED_STATIC_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
    limits=MotionLimits();stage='SETTLE';stable_since=None;plan_start=None;nominal_end=None;fault_trial_start=None
    heartbeat_stamp=None;fault_at=None;fault_detect=None;stopped_since=None;abort_after=None;run_result={'status':'NOT_RUN','Query_control':False}
    proofs=[];pending=[];source_cache=[];clock_delta=0.;activation_events=[]
    def install(prog,activation,tag):
        prog.validate(limits)
        cert=volume.certify(prog.coefficients,prog.durations,rospy.Time.now().to_sec(),required_until=activation+prog.duration)
        if cert['status']!='CERTIFIED_FREE':raise ValueError('EXTERNAL_ROOM_SPACE_UNPROVEN')
        with lock:
            before,prevtag=reference(activation)
            gap=np.max(np.abs(prog.pva(0)-before),axis=1)
            if (gap>[.0002,.002,.01]).any():raise ValueError('REFERENCE_PVA_DISCONTINUITY '+str(gap))
            if pending:raise ValueError('PENDING_ALREADY_PRESENT')
            pending.append((activation,prog,tag))
        item=dict(type='INSTALL',activation=activation,tag=tag,previous=prevtag,pva_gap=gap,certificate=cert,extrema=prog.extrema(),duration=prog.duration,coefficients=prog.coefficients,durations=prog.durations,request_stamp=rospy.Time.now().to_sec(),source_measurement=None if latest[0] is None else dict(stamp=latest[0].stamp,pva=latest[0].pva))
        proofs.append(item);log('ARBITRATION',item)
    def emergency(now,reason):
        nonlocal stage,abort_after
        if stage.startswith('ABORT'):return
        pending.clear()
        violations.append(reason);log('ARBITRATION',dict(type='ABORT',stamp=now,reason=reason))
        activation=now+.06
        with lock:head,_=reference(activation)
        try:
            stop,_=brake(head,limits);install(stop,activation,'ABORT_BRAKE');abort_after=activation+stop.duration+.5
        except Exception as exc:
            violations.append('ABORT_BRAKE_UNPROVEN '+str(exc));abort_after=now+.1
        stage='ABORTING'
    h0=np.stack([origin,np.zeros(3),np.zeros(3)])
    h1=np.array([[0,0,2],[.8,0,0],[0,0,0]],float)
    # Independent non-Query bench prelude: same actual spawn; cruise crosses x=0.
    # This is explicitly NOT a reinterpretation of fixed-position0.5s handover.
    ramp_tail=np.array([[-.6,0,2],[.8,0,0],[0,0,0]],float)
    warm=quintic(h0,ramp_tail,2.5)
    cruise=cruise_segment(ramp_tail,.75)
    nominal_terminal=quintic(h1,np.array([[3,0,2],[0,0,0],[0,0,0]],float),7.5)
    preset_nominal=concatenate(concatenate(warm,cruise),nominal_terminal)
    preset_watchdog=quintic(np.array([[3,0,2],[0,0,0],[0,0,0]],float),np.array([[0,0,2],[0,0,0],[0,0,0]],float),8.3)
    preset_nominal.validate(limits);preset_watchdog.validate(limits)
    probes=['SCHEDULING_DELAY','CLOCK_JUMP','DROPPED_ODOM','PUBLISHER_CONFLICT']
    probe_index=0;next_probe=None;probe_active=None;probe_rejections=[]
    abort_reference_start=None;abort_test_at=None;abort_stop_end=None
    last_ros=None;last_mono=None;next_tick=time.monotonic();nominal_stopped=False;watchdog_stopped=False
    def stopped(m):
        return m is not None and np.linalg.norm(m.pva[1])<=.03 and np.linalg.norm(m.pva[2])<=.05
    def schedule(prog,now,tag):
        activation=now+.06
        install(prog,activation,tag)
        return activation
    def publish_current(now,activation=None):
        nonlocal sequence,last_publish_ros
        with lock:ref,tag=reference(now)
        msg=PositionCommand();msg.header.stamp=rospy.Time.from_sec(now);msg.header.seq=sequence;msg.header.frame_id='world'
        msg.trajectory_id=sequence;msg.trajectory_flag=PositionCommand.TRAJECTORY_STATUS_EMPTY
        msg.position.x,msg.position.y,msg.position.z=map(float,ref[0])
        msg.velocity.x,msg.velocity.y,msg.velocity.z=map(float,ref[1])
        msg.acceleration.x,msg.acceleration.y,msg.acceleration.z=map(float,ref[2]);msg.yaw=0.
        begin=rospy.Time.now().to_sec()
        if last_publish_ros is not None and begin-last_publish_ros>.04:raise ValueError('COMMAND_GAP_GT40MS')
        if activation is not None:
            # Validate again immediately at the physical publisher boundary.
            activation_guard(activation,begin,latest[0].stamp,topology_sources,clock_delta)
        pub.publish(msg);end=rospy.Time.now().to_sec();last_publish_ros=begin
        log('COMMAND',dict(stamp_ns=msg.header.stamp.to_nsec(),publish_begin=begin,publish_end=end,seq=sequence,pva=ref,phase=tag,publisher='/f3_reference_arbiter',Query=False))
        sequence+=1
        if activation is not None:
            dt=end-activation
            ev=dict(type='ACTIVATED',planned=activation,first_publish_begin=begin,first_publish_end=end,lateness_s=dt,phase=tag)
            activation_events.append(ev);log('ARBITRATION',ev)
            if dt>.01:raise ValueError('PUBLISH_COMPLETION_TOO_LATE')
    dump('ARBITER_READY.json',dict(pid=__import__('os').getpid(),publisher='/f3_reference_arbiter',reference_only=True,Query_control=False))
    try:
        while not rospy.is_shutdown() and time.monotonic()-clock_start<65:
            mono=time.monotonic();now=rospy.Time.now().to_sec()
            clock_delta=0. if last_ros is None else (now-last_ros)-(mono-last_mono)
            last_ros=now;last_mono=mono
            if abs(clock_delta)>.005:emergency(now,'CLOCK_JUMP')
            with lock:m=latest[0]
            if error_stop:
                run_result.update(status='FAIL_AUTHORITY',reasons=error_stop,physical_stop_guaranteed=False);break
            if m is not None and stage not in ('SETTLE','PROBES'):
                err=np.linalg.norm(m.pva-reference(m.stamp)[0],axis=1)
                if not 0<=now-m.stamp<=.05:emergency(now,'STALE_ODOM')
                if (err>[.02,.05,.1]).any():emergency(now,'TRACKING_TUBE_EXCEEDED '+str(err))
                if np.linalg.norm(m.pva[1])>1 or np.linalg.norm(m.pva[2])>1.5:emergency(now,'ACTUAL_LIMIT_EXCEEDED')
            if (run/'HOST_STOP').exists():emergency(now,'HOST_RESOURCE_STOP')
            if stage=='SETTLE' and m is not None:
                err=np.linalg.norm(m.pva-np.stack([origin,np.zeros(3),np.zeros(3)]),axis=1)
                okay=(err<=[.02,.05,.1]).all() and graph_ok.is_set() and (run/'SCENE_READY.json').exists()
                if okay:
                    if stable_since is None:stable_since=now
                    if now-stable_since>=.5:stage='PROBES';next_probe=now+.06
                else:stable_since=None
                if time.monotonic()-clock_start>15:emergency(now,'SETTLE_OR_SCENE_GATE_TIMEOUT')
            elif stage=='PROBES':
                if probe_index<len(probes) and now>=next_probe:
                    probe=probes[probe_index];planned=next_probe;meas=m
                    if probe=='SCHEDULING_DELAY':time.sleep(.015);now=rospy.Time.now().to_sec()
                    sources=topology_sources if probe!='PUBLISHER_CONFLICT' else ['/f3_reference_arbiter','/INJECTED_CONFLICT_NO_ROS_PUBLISHER']
                    ms=meas.stamp
                    if probe=='DROPPED_ODOM':
                        # An actually received earlier Odom stamp; do not fabricate a sensor stamp.
                        ms=source_cache[0] if source_cache else ms
                    try:
                        activation_guard(planned,now,ms,sources,.2 if probe=='CLOCK_JUMP' else 0.)
                        raise RuntimeError('FAULT_PROBE_NOT_REJECTED '+probe)
                    except ValueError as exc:
                        item=dict(type='INJECTED_ACTIVATION_REJECT',probe=probe,requested=planned-.06,planned=planned,actual_ros=now,reason=str(exc),measurement_stamp=ms,
                            injection_scope='gate_input_only; real ROS clock and physical publisher unchanged')
                        probe_rejections.append(item);log('ARBITRATION',item)
                    probe_index+=1;next_probe=now+.10
                if probe_index==len(probes) and now>=next_probe:
                    plan_start=schedule(preset_nominal,now,'NOMINAL_REFERENCE_3M')
                    nominal_end=plan_start+preset_nominal.duration;stage='NOMINAL'
                    log('ARBITRATION',dict(type='NOMINAL_BEGIN',activation=plan_start,scored_begin=plan_start+3.25,goal=[3,0,2],
                        prelude='2.5s ramp then0.75s actual cruise; diagnostic stable-window semantics explicit'))
            elif stage=='NOMINAL' and now>=nominal_end:
                if stopped(m) and np.linalg.norm(m.pva[0]-[3,0,2])<=.1:
                    if stopped_since is None:stopped_since=now
                    if now-stopped_since>=.5:
                        nominal_stopped=True;log('ARBITRATION',dict(type='NOMINAL_STOP_DWELL_PASS',stamp=now,actual=m.pva))
                        fault_trial_start=schedule(preset_watchdog,now,'WATCHDOG_REFERENCE')
                        fault_at=fault_trial_start+2.5;heartbeat_stamp=now;stage='WATCHDOG';stopped_since=None
                else:stopped_since=None
                if now-nominal_end>3:emergency(now,'NOMINAL_STOP_DWELL_FAILED')
            elif stage=='WATCHDOG':
                if now<fault_at:heartbeat_stamp=now
                elif fault_detect is None and now-heartbeat_stamp>.1:
                    fault_detect=now;activation=now+.06;head,_=reference(activation)
                    stop,bounds=brake(head,limits);install(stop,activation,'WATCHDOG_BRAKE')
                    log('ARBITRATION',dict(type='WATCHDOG_TRIGGER',fault_injection=fault_at,last_heartbeat=heartbeat_stamp,detection=now,activation=activation,
                        measured_state=m.pva,measurement_stamp=m.stamp,predicted_actual_activation=m.predict(activation),brake_head=head,bounds=bounds,
                        head_semantics='continuous previous reference; real measured prediction must independently match tracking tube'))
                    abort_after=activation+stop.duration;stage='WATCHDOG_STOP'
            elif stage=='WATCHDOG_STOP' and now>=abort_after:
                if stopped(m):
                    if stopped_since is None:stopped_since=now
                    if now-stopped_since>=.5:
                        watchdog_stopped=True;log('ARBITRATION',dict(type='WATCHDOG_STOP_DWELL_PASS',stamp=now,actual=m.pva))
                        head=reference(now+.06)[0]
                        tail=head.copy();tail[0]+=np.array([2.0,0,0])
                        abort_ref=quintic(head,tail,8.)
                        abort_reference_start=schedule(abort_ref,now,'SAFETY_ABORT_REFERENCE')
                        abort_test_at=abort_reference_start+2.4;stage='ABORT_TEST';stopped_since=None
                else:stopped_since=None
                if now-abort_after>3:emergency(now,'WATCHDOG_NOT_STOPPED')
            elif stage=='ABORT_TEST' and now>=abort_test_at:
                activation=now+.06;head=reference(activation)[0];stop,bounds=brake(head,limits)
                install(stop,activation,'SAFETY_ABORT_BRAKE')
                log('ARBITRATION',dict(type='SAFETY_ABORT_TRIGGER',detection=now,activation=activation,measurement_stamp=m.stamp,measured_state=m.pva,
                    predicted_actual_activation=m.predict(activation),brake_head=head,bounds=bounds))
                abort_stop_end=activation+stop.duration;stage='ABORT_TEST_STOP';stopped_since=None
            elif stage=='ABORT_TEST_STOP' and now>=abort_stop_end:
                if stopped(m):
                    if stopped_since is None:stopped_since=now
                    if now-stopped_since>=.5:
                        log('ARBITRATION',dict(type='SAFETY_ABORT_STOP_DWELL_PASS',stamp=now,actual=m.pva))
                        run_result.update(status='REFERENCE_TRIALS_COMPLETE',nominal=nominal_stopped,watchdog=watchdog_stopped,safety_abort=True);break
                else:stopped_since=None
                if now-abort_stop_end>3:emergency(now,'SAFETY_ABORT_NOT_STOPPED')
            if stage=='ABORTING' and abort_after is not None and now>=abort_after:
                run_result.update(status='FAIL_CONTROLLER_CONTRACT',violations=violations);break
            # Absolute deadline wakeup. Activation is an independent event; no Rate.sleep20ms quantisation.
            did_activate=False
            if pending and now>=pending[0][0]:
                activation,prog,tag=pending.pop(0)
                try:
                    actual=rospy.Time.now().to_sec()
                    activation_guard(activation,actual,m.stamp,topology_sources,clock_delta)
                    predicted=m.predict(actual);old_head=reference(actual)[0]
                    if (np.linalg.norm(predicted-old_head,axis=1)>[.02,.05,.1]).any():raise ValueError('ACTIVATION_REAL_STATE_TUBE')
                    with lock:timeline.append((activation,prog,tag))
                    log('ARBITRATION',dict(type='ACTIVATION_STATE',planned=activation,actual_ros=actual,measurement_stamp=m.stamp,actual_measured=m.pva,
                        predicted_actual=predicted,previous_reference=old_head,prediction_residual=np.linalg.norm(predicted-old_head,axis=1)))
                    publish_current(rospy.Time.now().to_sec(),activation);did_activate=True;next_tick=time.monotonic()+.02
                except ValueError as exc:
                    log('ARBITRATION',dict(type='ACTIVATION_REJECT',planned=activation,actual_ros=rospy.Time.now().to_sec(),reason=str(exc),tag=tag))
                    emergency(rospy.Time.now().to_sec(),str(exc))
            if not did_activate and time.monotonic()>=next_tick:
                now=rospy.Time.now().to_sec()
                if m is not None:
                    source_cache.append(m.stamp);source_cache[:]=source_cache[-100:]
                publish_current(now);next_tick+=.02
                if next_tick<time.monotonic()-.02:emergency(now,'MISSED_50HZ_TICKS')
            # Bound real command gaps and don't busy spin for substantial waits.
            next_wake=next_tick
            if pending:next_wake=min(next_wake,time.monotonic()+max(0,pending[0][0]-rospy.Time.now().to_sec()))
            time.sleep(max(0,min(.002,next_wake-time.monotonic())))
        else:run_result.update(status='FAIL_RUNTIME_LIMIT',violations=violations)
    except Exception:
        run_result.update(status='FAIL_EXCEPTION',traceback=traceback.format_exc(),violations=violations)
    finally:
        done.set();worker.join(timeout=2)
        with lock:closing[0]=True
        for s in subs:s.unregister()
        log('ARBITRATION',dict(type='CALLBACKS_DRAINED',stamp=rospy.Time.now().to_sec()))
        pub.unregister()
        dump('ARBITER_RESULT.json',dict(**run_result,commands=sequence,proofs=proofs,elapsed_s=time.monotonic()-clock_start,violations_all=violations,source='NON_QUERY_REFERENCE_ONLY',activation_events=activation_events,probe_rejections=probe_rejections))
        for f in logs.values():f.close()
        rospy.signal_shutdown('bounded reference-only bench finished')
if __name__=='__main__':main()
