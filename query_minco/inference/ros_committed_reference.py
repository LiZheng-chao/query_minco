"""Reference-only ROS adapter of committed_reference. Query input is absent."""
import argparse,collections,json,os,pickle,subprocess,threading,time,traceback
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import ReferenceKernel,StateSnapshot
from query_minco.inference.constrained_prefix import validate_activation

def main():
    import rospy,rosgraph
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String
    from quadrotor_msgs.msg import PositionCommand,SO3Command
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--reference-only',required=True,action='store_true')
    a=parser.parse_args();run=Path(a.run);out=run.parent
    # Large immutable prepared payload loaded BEFORE any physical publication.
    with (out/'PREPARED_REFERENCES.pkl').open('rb') as f:bundles=pickle.load(f)
    rospy.init_node('f3_reference_arbiter',disable_signals=True)
    master=rosgraph.Master('/f3_reference_arbiter')
    assert not dict(master.getSystemState()[0]).get('/so3_control/pos_cmd'),'EXISTING_PUBLISHER'
    pub=rospy.Publisher('/so3_control/pos_cmd',PositionCommand,queue_size=1,tcp_nodelay=True)
    kernel=ReferenceKernel();queue=collections.deque();latest=[None];previous=[None]
    authority=[((),0.)];phase=[''];so3=[None];stop=threading.Event();closing=[False]
    job=[None];abort_ready=[None];children=[];io_errors=[]
    def log(kind,**x):queue.append(dict(kind=kind,received_wall_ns=time.time_ns(),received_mono=time.monotonic(),**x))
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    files={n:(run/(n+'.jsonl')).open('x',buffering=1) for n in ['ODOM','COMMAND','EVENT','SO3','IMU','TOPOLOGY']}
    def io_loop():
        try:
            while not stop.is_set() or queue:
                if queue:
                    row=queue.popleft();kind=row.pop('kind');files[kind].write(json.dumps(row,default=default)+'\n')
                    if kind=='EVENT' and row.get('type')=='HOLD_REFERENCE_READY':
                        with (run/'HOLD_READY.json').open('x') as f:json.dump(row,f,default=default)
                else:stop.wait(.002)
        except Exception:io_errors.append(traceback.format_exc())
    io=threading.Thread(target=io_loop);io.start()
    def planner_loop():
        try:
            while not stop.is_set():
                req=job[0]
                if req is None:stop.wait(.01);continue
                job[0]=None
                with (run/(req['kind']+'_JOB.json')).open('x') as f:json.dump(req,f)
                cmd=['/home/lizheng/anaconda3/envs/yopo/bin/python','-B','-s',
                     str(Path(__file__).resolve().parents[2]/'experiments/risk_aware_set_physics_v1/prepare_committed_bench.py'),
                     '--out',str(run),'--job',str(run/(req['kind']+'_JOB.json'))]
                with (run/(req['kind']+'_WORKER.log')).open('xb') as f:
                    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);children.append(child)
                    log('EVENT',type='PLANNING_PROCESS_STARTED',pid=child.pid,job=req['kind'])
                    while child.poll() is None and not stop.wait(.02):pass
                    if child.poll() is None:child.terminate();child.wait(timeout=3)
                    log('EVENT',type='PLANNING_PROCESS_FINISHED',pid=child.pid,job=req['kind'],returncode=child.returncode)
                    if child.returncode:io_errors.append('PLANNER_FAILED '+str(child.returncode))
                if req['kind']=='abort':
                    with (run/'ABORT_REFERENCE.pkl').open('rb') as f:abort_ready[0]=pickle.load(f)
        except Exception:io_errors.append(traceback.format_exc())
    planner=threading.Thread(target=planner_loop);planner.start()
    def graph_loop():
        while not stop.is_set():
            try:
                pubs=master.getSystemState()[0]
                sources=tuple(dict(pubs).get('/so3_control/pos_cmd',()))
                if any('/yopo_net' in ns for _,ns in pubs):sources+=('/FORBIDDEN_YOPO',)
                authority[0]=(sources,time.monotonic())
                log('TOPOLOGY',publishers=pubs,stamp=rospy.Time.now().to_sec())
            except Exception:authority[0]=((),time.monotonic())
            stop.wait(.2)
    graph=threading.Thread(target=graph_loop);graph.start()
    def odom_cb(m):
        if closing[0]:return
        ns=m.header.stamp.to_nsec();t=ns/1e9;rec=time.monotonic()
        p=m.pose.pose.position;v=m.twist.twist.linear;q=m.pose.pose.orientation
        pos=np.array([p.x,p.y,p.z]);vel=np.array([v.x,v.y,v.z]);last=previous[0]
        if last is None:previous[0]=(t,vel);return
        dt=t-last[0]
        if dt<.005:return
        previous[0]=(t,vel.copy())
        acc=(vel-last[1])/dt if dt<=.05 else np.full(3,np.nan)
        pva=np.stack([pos,vel,acc]);src,at=authority[0]
        latest[0]=StateSnapshot(t,rec,tuple(map(tuple,pva)),src,at)
        log('ODOM',stamp_ns=ns,pva=pva,q=[q.x,q.y,q.z,q.w],phase=phase[0],acceleration_dt=dt)
    def so3_cb(m):
        if closing[0]:return
        so3[0]=m
        log('SO3',stamp_ns=m.header.stamp.to_nsec(),force=[m.force.x,m.force.y,m.force.z],
            q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w])
    def phase_cb(m):
        if m.data!=phase[0]:phase[0]=m.data;log('EVENT',type='STARTUP_PHASE',value=m.data,ros=rospy.Time.now().to_sec())
    def imu_cb(m):
        if not closing[0]:log('IMU',stamp_ns=m.header.stamp.to_nsec(),a=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
    subs=[rospy.Subscriber('/sim/odom',Odometry,odom_cb,queue_size=5,tcp_nodelay=True),
          rospy.Subscriber('/so3_cmd',SO3Command,so3_cb,queue_size=5,tcp_nodelay=True),
          rospy.Subscriber('/sim/imu',Imu,imu_cb,queue_size=5),
          rospy.Subscriber('/quadrotor_simulator_so3/startup_phase',String,phase_cb,queue_size=2)]
    with (run/'ARBITER_READY.json').open('x') as f:json.dump(dict(pid=os.getpid(),reference_only=True,Query_control=False),f)
    stage='HOLD';hold_ready=False;stable=None;pending_stage=None;nominal_start=None;wd=False
    evidence_stress=False;reject_injected=False;result={'status':'NOT_RUN','Query_control':False}
    begin=time.monotonic();due=begin;seq=0;last_publish=None;emergency=False
    def submit(bundle,tag,ros,mono):
        nonlocal pending_stage
        kernel.submit(bundle,ros+.06,mono+.06);pending_stage=tag
        log('EVENT',type='REQUEST',tag=tag,ros=ros,mono=mono,activation_ros=ros+.06,activation_mono=mono+.06)
    try:
        while not rospy.is_shutdown() and time.monotonic()-begin<72:
            wake=due
            if kernel.pending is not None:
                am=kernel.pending[2]
                wake=am if due<=am<=due+.005 else min(due,am)
            if kernel.active is not None:
                deadline=(kernel.start_mono+kernel.active.prefix.duration) if kernel.stop_entry is None else (kernel.stop_deadline if kernel.mode=='BACKUP_ARMED' else None)
                if deadline is not None and deadline>=time.monotonic():
                    wake=deadline if due<=deadline<=due+.005 else min(wake,deadline)
            while time.monotonic()<wake:time.sleep(max(0,min(.001,wake-time.monotonic())))
            mono=time.monotonic();ros=rospy.Time.now().to_sec();m=latest[0]
            # No file checks, graph RPC, computation, or evidence writes below.
            if kernel.active is None and kernel.pending is None and m is not None and authority[0][0]==('/f3_reference_arbiter',):
                submit(bundles['STATIC'],'STATIC',ros,mono)
            previous_commitment=(kernel.active,kernel.start_ros,kernel.start_mono,kernel.stop_entry,kernel.stop_deadline,kernel.mode)
            ref=kernel.tick(ros,mono,m)
            if stage=='HOLD' and phase[0]=='RELEASED_BEFORE_FIRST_INTEGRATION' and kernel.active is bundles['STATIC'] and m:
                actual=np.array(m.pva);within=np.linalg.norm(actual[0]-[-1.6,0,2])<=.02 and np.linalg.norm(actual[1])<=.03 and np.linalg.norm(actual[2])<=.05
                if within:
                    if stable is None:stable=m.stamp_ros
                    if m.stamp_ros-stable>=.5:submit(bundles['NOMINAL'],'NOMINAL',ros,mono);stage='NOMINAL';stable=None
                else:stable=None
            if stage=='NOMINAL' and kernel.active is bundles['NOMINAL']:
                if nominal_start is None:nominal_start=kernel.start_mono
                if mono-kernel.start_mono>2 and not reject_injected:
                    # A late new plan is rejected without replacing existing commitment.
                    kernel.submit(bundles['STATIC'],ros-.02,mono-.02);reject_injected=True
                    log('EVENT',type='LATE_PROPOSAL_INJECTED',ros=ros,mono=mono)
                if mono-kernel.start_mono>=kernel.active.prefix.duration and m:
                    state=np.array(m.pva)
                    within=np.linalg.norm(state[0]-[3,0,2])<=.1 and np.linalg.norm(state[1])<=.03 and np.linalg.norm(state[2])<=.05
                    if within:
                        if stable is None:stable=m.stamp_ros
                        if m.stamp_ros-stable>=.5:
                            log('EVENT',type='NOMINAL_STOP_DWELL',actual=m.pva,ros=ros,mono=mono)
                            submit(bundles['WATCHDOG'],'WATCHDOG',ros,mono);stage='WATCHDOG';stable=None
                    else:stable=None
            if stage=='WATCHDOG' and kernel.active is bundles['WATCHDOG']:
                elapsed=mono-kernel.start_mono
                if elapsed>.8 and not evidence_stress:
                    job[0]={'kind':'stress'};evidence_stress=True
                if elapsed>=2.5 and not wd:
                    kernel.request_stop('WATCHDOG_HEARTBEAT_TIMEOUT',ros,mono);wd=True
                    log('EVENT',type='WATCHDOG_TRIGGER',ros=ros,mono=mono,actual=None if m is None else m.pva,
                        actual_stamp=None if m is None else m.stamp_ros,backup_phase=kernel.stop_entry.phase,
                        backup_duration=kernel.stop_entry.polynomial.duration)
                    stage='WATCHDOG_STOP'
            if stage in ('WATCHDOG_STOP','ABORT_STOP','FAILED_STOP') and kernel.stop_entry and mono>=kernel.stop_deadline+kernel.stop_entry.polynomial.duration and m:
                state=np.array(m.pva);within=np.linalg.norm(state[1])<=.03 and np.linalg.norm(state[2])<=.05
                if within:
                    if stable is None:stable=m.stamp_ros
                    if m.stamp_ros-stable>=.5:
                        log('EVENT',type=stage+'_DWELL',actual=m.pva,ros=ros,mono=mono)
                        if stage=='FAILED_STOP':result.update(status='FAIL_SAFETY_CONTRACT_STOP_COMPLETED');break
                        if stage=='ABORT_STOP':result.update(status='NON_QUERY_NATIVE_TRIALS_COMPLETE',watchdog=True,safety_abort=True);break
                        job[0]={'kind':'abort','head':ref.tolist()};stage='PREPARE_ABORT';stable=None
                else:stable=None
            if stage=='PREPARE_ABORT' and abort_ready[0] is not None:
                submit(abort_ready[0],'SAFETY_ABORT',ros,mono);stage='ABORT'
            if stage=='ABORT' and kernel.active is abort_ready[0] and mono-kernel.start_mono>=2.4:
                kernel.request_stop('EXPLICIT_SAFETY_ABORT',ros,mono);stage='ABORT_STOP'
                log('EVENT',type='SAFETY_ABORT_TRIGGER',ros=ros,mono=mono,actual=None if m is None else m.pva,
                    backup_phase=kernel.stop_entry.phase,backup_duration=kernel.stop_entry.polynomial.duration)
            while kernel.events:
                log('EVENT',type='KERNEL',stage=stage,event=kernel.events.popleft())
            if io_errors or kernel.qualification_failed:
                if not emergency:
                    emergency=True
                    log('EVENT',type='FAILURE_LATCH',ros=ros,mono=mono,io_errors=io_errors,kernel_mode=kernel.mode)
                    kernel.request_stop('FAILURE_LATCH',ros,mono);stage='FAILED_STOP';stable=None
            if ref is not None:
                msg=PositionCommand();msg.header.stamp=rospy.Time.now();msg.header.frame_id='world';msg.header.seq=seq;msg.trajectory_id=seq
                msg.trajectory_flag=PositionCommand.TRAJECTORY_STATUS_EMPTY
                msg.position.x,msg.position.y,msg.position.z=map(float,ref[0]);msg.velocity.x,msg.velocity.y,msg.velocity.z=map(float,ref[1])
                msg.acceleration.x,msg.acceleration.y,msg.acceleration.z=map(float,ref[2]);msg.yaw=0.
                before=rospy.Time.now().to_sec()
                changed=kernel.active is not previous_commitment[0]
                if changed:
                    try:validate_activation(kernel.start_ros,before)
                    except ValueError as exc:
                        kernel.active,kernel.start_ros,kernel.start_mono,kernel.stop_entry,kernel.stop_deadline,kernel.mode=previous_commitment
                        log('EVENT',type='PHYSICAL_ACTIVATION_REJECT',ros=before,reason=str(exc))
                        if kernel.active is None:
                            due+=.02
                            continue
                        ref=kernel._reference(time.monotonic())
                        msg.position.x,msg.position.y,msg.position.z=map(float,ref[0]);msg.velocity.x,msg.velocity.y,msg.velocity.z=map(float,ref[1])
                        msg.acceleration.x,msg.acceleration.y,msg.acceleration.z=map(float,ref[2]);changed=False
                pub.publish(msg);after=rospy.Time.now().to_sec()
                if changed:
                    log('EVENT',type='ACTUAL_ACTIVATION',planned=kernel.start_ros,first_publish_begin=before,first_publish_end=after,lateness=after-kernel.start_ros)
                    if after-kernel.start_ros>.01:kernel.qualification_failed=True
                if kernel.active is bundles['STATIC'] and not hold_ready:
                    hold_ready=True;log('EVENT',type='HOLD_REFERENCE_READY',ros=after,mono=time.monotonic())
                log('COMMAND',stamp_ns=msg.header.stamp.to_nsec(),begin=before,end=after,pva=ref,stage=stage,mode=kernel.mode,seq=seq,Query=False)
                if last_publish is not None and before-last_publish>.04:
                    kernel.qualification_failed=True;log('EVENT',type='COMMAND_GAP_FAULT',gap=before-last_publish)
                last_publish=before;seq+=1
            if wake>=due-.001:due+=.02
        else:result.update(status='FAIL_RUNTIME_LIMIT')
    except Exception:result.update(status='FAIL_EXCEPTION',traceback=traceback.format_exc())
    finally:
        closing[0]=True
        for sub in subs:sub.unregister()
        log('EVENT',type='CALLBACKS_DRAINED',ros=rospy.Time.now().to_sec())
        stop.set();graph.join(2);planner.join(5);io.join(5)
        for child in children:
            if child.poll() is None:child.terminate();child.wait(timeout=3)
        pub.unregister()
        for f in files.values():f.close()
        with (run/'ARBITER_RESULT.json').open('x') as f:json.dump(dict(**result,commands=seq,kernel_failed=kernel.qualification_failed,
            kernel_mode=kernel.mode,elapsed=time.monotonic()-begin,io_errors=io_errors,physical_stop_requires_raw_verification=True),f,indent=2)
        rospy.signal_shutdown('bounded reference-only committed bench finished')
if __name__=='__main__':main()
