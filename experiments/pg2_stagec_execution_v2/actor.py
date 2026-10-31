"""Opt-in empty dynamics-rig actors. Never consumes Query candidates or a GT safety certificate."""
import argparse,os,sys,time,json,threading,copy,subprocess
from pathlib import Path
sys.path.insert(0,'/home/lizheng/query_minco_mainline')
import numpy as np
import rospy,rosgraph
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand,SO3Command
from mavros_msgs.msg import AttitudeTarget
from query_minco.integration.stagec_execution_v2.reference import Lease,pva,brake,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['gateway','source']);ap.add_argument('folder');ap.add_argument('--enable-isolated-dynamics-test',action='store_true');args=ap.parse_args()
    if not args.enable_isolated_dynamics_test:raise ValueError('DEFAULT_DISABLED')
    out=Path(args.folder).resolve();cfg=json.loads((out/'CASE.json').read_text());protocol=json.loads((out/'PROTOCOL.json').read_text())
    if cfg['scope']!='EMPTY_DYNAMICS_RIG_ONLY_NOT_ONLINE_FREE_SPACE_CERTIFICATE' or os.environ.get('ROS_MASTER_URI')!='http://127.0.0.1:'+str(cfg['port']):raise ValueError('ISOLATED_MASTER_SCOPE')
    rospy.init_node('stagec_'+args.role,disable_signals=True);lock=threading.RLock();odom=None;past=None;acc=np.zeros(3);logs={}
    def emit(kind,**x):
        if kind not in logs:logs[kind]=(out/(args.role+'_'+kind+'.jsonl')).open('x',buffering=1)
        x.update(wall_ns=time.time_ns(),steady=time.monotonic(),ros=rospy.Time.now().to_sec());logs[kind].write(json.dumps(x,allow_nan=False)+'\n')
    def odom_cb(m):
        nonlocal odom,past,acc
        with lock:
            pos=np.array([m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z]);vel=np.array([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]);stamp=m.header.stamp.to_sec()
            if odom is not None and stamp>odom['stamp'] and .005<=stamp-odom['stamp']<=.03:acc=(vel-odom['v'])/(stamp-odom['stamp'])
            odom=dict(p=pos,v=vel,stamp=stamp,q=[m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w])
            if args.role=='gateway':emit('ODOM',p=pos.tolist(),v=vel.tolist(),a_causal=acc.tolist(),stamp=stamp,q=odom['q'])
    rospy.Subscriber('/sim/odom',Odometry,odom_cb,queue_size=5,tcp_nodelay=True)
    if args.role=='source':
        pub=rospy.Publisher('/stagec/plan',String,queue_size=10);start=time.monotonic()
        while time.monotonic()-start<12 and not (out/'GATEWAY_READY.json').exists():time.sleep(.02)
        if not (out/'GATEWAY_READY.json').exists():raise ValueError('GATEWAY_NOT_READY')
        time.sleep(.2)
        with lock:
            if odom is None:raise ValueError('NO_ODOM')
            origin=odom['p'].copy()
        stamp=rospy.Time.now().to_sec();v=cfg['speed'];T=3.;c=np.zeros((2,3,6));c[0,:,0]=origin;c[0,0,3]=v/T**2;c[0,0,4]=-v/(2*T**3);c[1,:,0]=origin;c[1,0,0]+=.5*v*T;c[1,0,1]=v
        if cfg.get('semantics_probe'):
            c[:]=0;c[:,:,0]=origin;c[:,0,0]+=.15
        plan=dict(source_stamp=stamp,coefficients=c.tolist(),durations=[T,12.]);msg=dict(owner='stagec_source',sequence=1,heartbeat=0,plan=plan);faulted=False;start=time.monotonic();hb=0
        from geometry_msgs.msg import PoseStamped
        gp=rospy.Publisher('/move_base_simple/goal',PoseStamped,queue_size=1,latch=True);goal=PoseStamped();goal.header.frame_id='world';goal.header.stamp=rospy.Time.now();goal.pose.position.x=float(origin[0]+10);goal.pose.position.y=float(origin[1]);goal.pose.position.z=float(origin[2]);goal.pose.orientation.w=1
        while time.monotonic()-start<15 and not rospy.is_shutdown():
            if gp.get_num_connections():gp.publish(goal)
            hb+=1;msg['heartbeat']=hb
            if not faulted and time.monotonic()-start>=5.:
                faulted=True;kind=cfg['fault'];event=dict(kind=kind,wall_ns=time.time_ns(),ros=rospy.Time.now().to_sec());(out/'INJECTION.json').write_text(json.dumps(event,indent=2));emit('FAULT',fault_kind=kind)
                if kind in ('publisher_exit','gateway_exit'):break
                if kind=='computation_timeout':time.sleep(.6)
                elif kind=='duplicate':msg['heartbeat']=hb-1
                elif kind=='owner_conflict':msg['owner']='rogue_owner'
                elif kind=='clock_reset_adapter':msg['clock_adapter_fault_s']=-1.
                else:
                    msg=copy.deepcopy(msg);msg['sequence']=2
                    if kind=='future':msg['plan']['source_stamp']=rospy.Time.now().to_sec()+1.
                    elif kind=='stale':msg['plan']['source_stamp']=rospy.Time.now().to_sec()-2.
                    elif kind=='expired':msg['plan']['source_stamp']=rospy.Time.now().to_sec()-20.
            pub.publish(String(data=json.dumps(msg)));emit('REF',message=msg);time.sleep(.02)
        (out/'SOURCE_RESULT.json').write_text(json.dumps(dict(status='EXECUTED',fault=cfg['fault'],source_stamp=stamp,plan_sha256=digest(plan)),indent=2));rospy.signal_shutdown('source scenario completed');return
    # Downstream is an independent controller reference channel; ordinary Query pos_cmd is never published.
    pub=rospy.Publisher('/stagec/controller_ref',PositionCommand,queue_size=1);master=rosgraph.Master(rospy.get_name());lease=Lease('stagec_source');fault=None;fallback=None;fallback_start=None;initial=None;trigger=None;history=[];graph_ok=False;ready_since=None;stop_since=None;stopped=False;armed=False
    def latch(reason):
        nonlocal fault,trigger
        if fault is None:fault=reason;trigger=dict(reason=reason,wall_ns=time.time_ns(),ros=rospy.Time.now().to_sec(),steady=time.monotonic());emit('TRIGGER',**trigger)
    def ref_cb(m):
        with lock:
            if fault:return
            now=rospy.Time.now().to_sec()
            try:
                msg=json.loads(m.data)
                if 'clock_adapter_fault_s' in msg:
                    if cfg['fault']!='clock_reset_adapter':raise ValueError('UNAUTHORIZED_TEST_CLOCK_ADAPTER')
                    lease.clock(now+float(msg['clock_adapter_fault_s']))
                lease.receive(msg,now,time.monotonic());emit('ACCEPT',sequence=msg['sequence'],heartbeat=msg['heartbeat'],source_stamp=msg['plan']['source_stamp'])
            except Exception as ex:latch(str(ex))
    rospy.Subscriber('/stagec/plan',String,ref_cb,queue_size=100)
    def so3_cb(m):emit('SO3',force=[m.force.x,m.force.y,m.force.z],q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],stamp=m.header.stamp.to_sec())
    def att_cb(m):emit('ATTITUDE',thrust=m.thrust,stamp=m.header.stamp.to_sec())
    rospy.Subscriber('/so3_cmd',SO3Command,so3_cb,queue_size=300);rospy.Subscriber('/mavros/setpoint_raw/attitude',AttitudeTarget,att_cb,queue_size=300)
    start=time.monotonic();last_graph=0;result=None
    try:
        while time.monotonic()-start<28 and not rospy.is_shutdown():
            now=rospy.Time.now().to_sec();steady=time.monotonic()
            if steady-last_graph>=.2:
                graph=master.getSystemState();nodes=master.getPid();pubs=dict(graph[0]);names=set(n for group in graph for _,nn in group for n in nn)
                allowed={'/rosout','/quadrotor_simulator_so3','/network_controller_node','/stagec_source','/stagec_gateway'}
                if cfg.get('case_id','').startswith('domain_shadow'):allowed.update({'/pg2_moving_readonly_shadow','/forest30_camera_lattice','/sensor_simulator_node','/stagec_full_volume_shadow'})
                if cfg.get('native_watchdog_test'):allowed.add('/stagec_native_observer')
                graph_ok=not(names-allowed) and pubs.get('/stagec/controller_ref',[])==['/stagec_gateway'] and rospy.get_param('/network_controller_node/is_simulation',False) is True
                emit('GRAPH',graph=graph,nodes=sorted(names),simulation_scope_ok=graph_ok)
                if not graph_ok:latch('ROS_OWNER_OR_HARDWARE_SCOPE_CONFLICT')
                last_graph=steady
            with lock:
                now=rospy.Time.now().to_sec();steady=time.monotonic() # after graph RPCs and concurrent state callbacks
                if odom is None:time.sleep(.02);continue
                if initial is None:initial=np.asarray(cfg.get('initial_position',[0.,0.,2.]),float)
                if not np.isfinite(odom['p']).all() or not np.isfinite(odom['v']).all():raise ValueError('NONFINITE_PLANT_STATE')
                if armed and now-odom['stamp']>.1:latch('STALE_STATE')
                if armed and float(np.linalg.norm(odom['v']))>protocol['max_plant_speed_mps']:latch('PLANT_SPEED_BOUND')
                if not armed:
                    settled=graph_ok and steady-start>3 and 0<=now-odom['stamp']<=.05 and np.linalg.norm(odom['v'])<.02 and np.linalg.norm(odom['p']-initial)<.02
                    ready_since=steady if settled and ready_since is None else ready_since if settled else None
                    emit('BOOTSTRAP',settled=bool(settled),state_age=now-odom['stamp'],speed=float(np.linalg.norm(odom['v'])),position_error=float(np.linalg.norm(odom['p']-initial)),held=0 if ready_since is None else steady-ready_since)
                    if ready_since is not None and steady-ready_since>=1.:
                        armed=True;(out/'GATEWAY_READY.json').write_text(json.dumps(dict(wall_ns=time.time_ns(),ros=now,p=odom['p'].tolist(),simulation_only=True,settled_hover_s=steady-ready_since)))
                    else:
                        # Native takeoff has its own controller. Never fight it by sending a ground hold.
                        pass # outer-loop sleep occurs AFTER releasing the odometry lock
                if fault is None and lease.plan is not None:
                    try:desired=lease.check(now,steady)
                    except ValueError as ex:latch(str(ex))
                else:desired=np.stack([initial,np.zeros(3),np.zeros(3)])
                if fault:
                    if fallback is None:
                        try:fallback=brake(odom['p'],odom['v'],acc,protocol['max_reference_a_mps2'],protocol['max_reference_j_mps3'])
                        except ValueError as ex:
                            # Explicitly unqualified hold request, not a zero-acceleration success claim.
                            fallback=dict(coefficients=[np.column_stack([odom['p'],np.zeros((3,5))]).tolist()],durations=[2.],endpoint=odom['p'].tolist(),unqualified_hold_reason=str(ex))
                        fallback['source_stamp']=now;fallback_start=dict(p=odom['p'].tolist(),v=odom['v'].tolist(),a=acc.tolist(),wall_ns=time.time_ns(),steady=steady);emit('BRAKE_PLAN',plan=fallback,initial=fallback_start)
                    elapsed=steady-fallback_start['steady']
                    desired=pva(fallback,now) if now<=fallback['source_stamp']+sum(fallback['durations']) else np.stack([fallback['endpoint'],np.zeros(3),np.zeros(3)])
                    if np.linalg.norm(odom['v'])<=protocol['stop_speed_mps']:
                        stop_since=steady if stop_since is None else stop_since
                        if steady-stop_since>=protocol['stop_hold_s']:stopped=True
                    else:stop_since=None
                    stopped=stop_since is not None and steady-stop_since>=protocol['stop_hold_s']
                    history.append(dict(elapsed=elapsed,p=odom['p'].tolist(),v=odom['v'].tolist()))
                    if elapsed>=protocol['stop_deadline_s']:
                        distance=max(np.linalg.norm(np.array(x['p'])-fallback_start['p']) for x in history);dz=max(abs(x['p'][2]-initial[2]) for x in history)
                        result=dict(status='DYNAMICS_STOP_PASS' if stopped and distance<=protocol['stop_displacement_limit_m'] and dz<=protocol['altitude_error_limit_m'] and 'unqualified_hold_reason' not in fallback else 'DYNAMICS_STOP_FAIL',trigger=trigger,fallback=fallback,initial=fallback_start,stopped_hold=stopped,maximum_stop_displacement_m=float(distance),maximum_altitude_error_m=float(dz),last_state=history[-1],FALLBACK_VERIFIED=False,online_volume='BLOCKED',motor_bounds='NOT_VERIFIED_NO_RPM_TELEMETRY',clock_scope='gateway clock-adapter injection only' if cfg['fault']=='clock_reset_adapter' else 'real ROS clock',Query_navigation=False);break
                cmd=PositionCommand();cmd.header.stamp=rospy.Time.from_sec(now);cmd.header.frame_id='world';cmd.trajectory_flag=0 if fault else cfg.get('controller_flag',0)
                for obj,value in zip([cmd.position,cmd.velocity,cmd.acceleration],desired):obj.x,obj.y,obj.z=map(float,value)
                if armed:pub.publish(cmd);emit('COMMAND',sample_stamp=now,header_stamp=cmd.header.stamp.to_sec(),pva=desired.tolist(),flag=cmd.trajectory_flag,fault=fault,source_stamp=None if lease.plan is None else lease.plan['source_stamp'],last_valid_receipt=lease.rx)
            time.sleep(.02)
        if result is None:result=dict(status='BLOCKED',reason='SCENARIO_TIMEOUT',fault=fault,FALLBACK_VERIFIED=False)
    finally:
        if result is not None:(out/'GATEWAY_RESULT.json').write_text(json.dumps(result,indent=2))
        for f in logs.values():f.close()
        rospy.signal_shutdown('bounded dynamics rig ended')
if __name__=='__main__':main()
