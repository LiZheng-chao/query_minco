"""Authorised NON-QUERY native-controller bench. Single PositionCommand publisher.
No network/checkpoint/candidate input exists. External static room supervision only.
"""
import argparse,collections,json,threading,time,traceback
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,Measurement,quintic,terminal,brake,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
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
    error_stop=[];sequence=0;last_tick=None;clock_start=time.monotonic();violations=[];events=[]
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
                if okay:graph_ok.set()
                elif sources and sources!=['/f3_reference_arbiter']:error_stop.append('PUBLISHER_CONFLICT')
                if any('/yopo_net' in ns or '/query_minco_shadow' in ns for _,ns in ps):error_stop.append('FORBIDDEN_PLANNER_PRESENT')
                log('ARBITRATION',dict(type='TOPOLOGY',stamp=rospy.Time.now().to_sec(),publishers=ps,okay=okay,scope='NON_QUERY_REFERENCE_ONLY'))
            except Exception as e:log('ARBITRATION',dict(type='TOPOLOGY_ERROR',error=str(e)))
            done.wait(.2)
    worker=threading.Thread(target=topology,daemon=True);worker.start()
    volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],rospy.Time.now().to_sec()+120,'RUNTIME_VERIFIED_STATIC_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
    limits=MotionLimits();stage='SETTLE';stable_since=None;plan_start=None;nominal_end=None;fault_trial_start=None
    heartbeat_stamp=None;fault_at=None;fault_detect=None;stopped_since=None;abort_after=None;run_result={'status':'NOT_RUN','Query_control':False}
    proofs=[]
    def install(prog,activation,tag):
        prog.validate(limits)
        cert=volume.certify(prog.coefficients,prog.durations,rospy.Time.now().to_sec(),required_until=activation+prog.duration)
        if cert['status']!='CERTIFIED_FREE':raise ValueError('EXTERNAL_ROOM_SPACE_UNPROVEN')
        with lock:
            before,prevtag=reference(activation)
            gap=np.max(np.abs(prog.pva(0)-before),axis=1)
            if (gap>[.0002,.002,.01]).any():raise ValueError('REFERENCE_PVA_DISCONTINUITY '+str(gap))
            timeline.append((activation,prog,tag))
        item=dict(type='INSTALL',activation=activation,tag=tag,previous=prevtag,pva_gap=gap,certificate=cert,extrema=prog.extrema(),duration=prog.duration)
        proofs.append(item);log('ARBITRATION',item)
    def emergency(now,reason):
        nonlocal stage,abort_after
        if stage.startswith('ABORT'):return
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
    preset_nominal=concatenate(quintic(h0,h1,4.),terminal(h1,[3,0,2]))
    preset_watchdog=terminal(np.array([[3,0,2],[0,0,0],[0,0,0]],float),[0,0,2])
    preset_nominal.validate(limits);preset_watchdog.validate(limits)
    dump('ARBITER_READY.json',dict(pid=__import__('os').getpid(),publisher='/f3_reference_arbiter',reference_only=True,Query_control=False))
    try:
        rate=rospy.Rate(50)
        while not rospy.is_shutdown() and time.monotonic()-clock_start<65:
            now=rospy.Time.now().to_sec()
            if last_tick is not None:
                if now<=last_tick:emergency(now,'CLOCK_JUMP')
                if now-last_tick>.04:emergency(now,'MISSED_50HZ_TICKS')
            last_tick=now
            with lock:m=latest[0];ref,tag=reference(now)
            if error_stop:
                run_result.update(status='FAIL_AUTHORITY',reasons=error_stop,physical_stop_guaranteed=False);break
            if m is not None and stage!='SETTLE':
                err=np.linalg.norm(m.pva-reference(m.stamp)[0],axis=1)
                if not 0<=now-m.stamp<=.05:emergency(now,'STALE_ODOM')
                if (err>[.02,.05,.1]).any():emergency(now,'TRACKING_TUBE_EXCEEDED '+str(err))
                if not graph_ok.is_set() or error_stop:emergency(now,'AUTHORITY_FAULT '+str(error_stop))
            if (run/'HOST_STOP').exists():emergency(now,'HOST_RESOURCE_STOP')
            if stage=='SETTLE' and m is not None:
                err=np.linalg.norm(m.pva-np.stack([origin,np.zeros(3),np.zeros(3)]),axis=1)
                okay=(err<=[.02,.05,.1]).all() and graph_ok.is_set() and (run/'SCENE_READY.json').exists()
                if okay:
                    if stable_since is None:stable_since=now
                    if now-stable_since>=.5:
                        plan_start=now+.06
                        plan=preset_nominal
                        install(plan,plan_start,'NOMINAL_REFERENCE_3M')
                        nominal_end=plan_start+plan.duration
                        stage='NOMINAL';log('ARBITRATION',dict(type='NOMINAL_BEGIN',activation=plan_start,scored_begin=plan_start+4,goal=[3,0,2]))
                else:stable_since=None
                if time.monotonic()-clock_start>15:emergency(now,'SETTLE_OR_SCENE_GATE_TIMEOUT')
            elif stage=='NOMINAL' and now>=nominal_end:
                if m is not None and np.linalg.norm(m.pva[0]-[3,0,2])<=.1 and np.linalg.norm(m.pva[1])<=.03 and np.linalg.norm(m.pva[2])<=.05:
                    if stopped_since is None:stopped_since=now
                    if now-stopped_since>=.5:
                        log('ARBITRATION',dict(type='NOMINAL_STOP_DWELL_PASS',stamp=now,actual=m.pva))
                        # New non-Query internal producer lease for deliberate heartbeat fault trial.
                        activation=now+.06
                        new=preset_watchdog
                        install(new,activation,'WATCHDOG_REFERENCE')
                        fault_trial_start=activation;fault_at=activation+1.4;heartbeat_stamp=now;stage='WATCHDOG';stopped_since=None
                else:stopped_since=None
                if now-nominal_end>3:emergency(now,'NOMINAL_STOP_DWELL_FAILED')
            elif stage=='WATCHDOG':
                if now<fault_at:heartbeat_stamp=now
                elif fault_detect is None:
                    if now-heartbeat_stamp>.1:
                        fault_detect=now;activation=now+.06
                        head,_=reference(activation)
                        stop,bounds=brake(head,limits)
                        install(stop,activation,'WATCHDOG_BRAKE')
                        log('ARBITRATION',dict(type='WATCHDOG_TRIGGER',fault_injection=fault_at,last_heartbeat=heartbeat_stamp,detection=now,activation=activation,measured_state=None if m is None else m.pva,brake_head=head,bounds=bounds))
                        abort_after=activation+stop.duration;stage='WATCHDOG_STOP'
            elif stage=='WATCHDOG_STOP' and now>=abort_after:
                if m is not None and np.linalg.norm(m.pva[1])<=.03 and np.linalg.norm(m.pva[2])<=.05:
                    if stopped_since is None:stopped_since=now
                    if now-stopped_since>=.5:
                        run_result.update(status='REFERENCE_TRIALS_COMPLETE',nominal=True,watchdog=True);break
                else:stopped_since=None
                if now-abort_after>3:emergency(now,'WATCHDOG_NOT_STOPPED')
            if stage=='ABORTING' and abort_after is not None and now>=abort_after:
                run_result.update(status='FAIL_CONTROLLER_CONTRACT',violations=violations);break
            with lock:ref,tag=reference(now)
            msg=PositionCommand();msg.header.stamp=rospy.Time.from_sec(now);msg.header.seq=sequence;msg.header.frame_id='world'
            msg.trajectory_id=sequence;msg.trajectory_flag=PositionCommand.TRAJECTORY_STATUS_EMPTY
            msg.position.x,msg.position.y,msg.position.z=map(float,ref[0])
            msg.velocity.x,msg.velocity.y,msg.velocity.z=map(float,ref[1])
            msg.acceleration.x,msg.acceleration.y,msg.acceleration.z=map(float,ref[2])
            msg.yaw=0.
            log('COMMAND',dict(stamp_ns=msg.header.stamp.to_nsec(),seq=sequence,pva=ref,phase=tag,publisher='/f3_reference_arbiter',Query=False))
            pub.publish(msg);sequence+=1;rate.sleep()
        else:run_result.update(status='FAIL_RUNTIME_LIMIT',violations=violations)
    except Exception:
        run_result.update(status='FAIL_EXCEPTION',traceback=traceback.format_exc(),violations=violations)
    finally:
        done.set();worker.join(timeout=2)
        with lock:closing[0]=True
        for s in subs:s.unregister()
        log('ARBITRATION',dict(type='CALLBACKS_DRAINED',stamp=rospy.Time.now().to_sec()))
        pub.unregister()
        dump('ARBITER_RESULT.json',dict(**run_result,commands=sequence,proofs=proofs,elapsed_s=time.monotonic()-clock_start,violations_all=violations,source='NON_QUERY_REFERENCE_ONLY'))
        for f in logs.values():f.close()
        rospy.signal_shutdown('bounded reference-only bench finished')
if __name__=='__main__':main()
