"""Independent recorder survives deliberate gateway Publisher loss. No command Publisher."""
from pathlib import Path
import sys,time,json,threading
import rospy,numpy as np
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import SO3Command
from mavros_msgs.msg import AttitudeTarget
from std_msgs.msg import String
out=Path(sys.argv[1]);cfg=json.loads((out/'CASE.json').read_text());protocol=json.loads((out/'PROTOCOL.json').read_text());rospy.init_node('stagec_native_observer',disable_signals=True);lock=threading.RLock();rows=[];watch=[];logs={}
def emit(kind,data):
 data.update(wall_ns=time.time_ns(),ros=rospy.Time.now().to_sec())
 if kind not in logs:logs[kind]=(out/('native_'+kind+'.jsonl')).open('x',buffering=1)
 logs[kind].write(json.dumps(data)+'\n');return data

def odom(m):
 with lock:rows.append(emit('ODOM',dict(p=[m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z],v=[m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z],stamp=m.header.stamp.to_sec())))
def state(m):
 with lock:watch.append(emit('WATCHDOG',json.loads(m.data)))
def so3(m):emit('SO3',dict(force=[m.force.x,m.force.y,m.force.z],stamp=m.header.stamp.to_sec()))
def att(m):emit('ATTITUDE',dict(thrust=m.thrust,stamp=m.header.stamp.to_sec()))
rospy.Subscriber('/sim/odom',Odometry,odom,queue_size=300,tcp_nodelay=True);rospy.Subscriber('/network_controller_node/stagec_watchdog_state',String,state,queue_size=300);rospy.Subscriber('/so3_cmd',SO3Command,so3,queue_size=300);rospy.Subscriber('/mavros/setpoint_raw/attitude',AttitudeTarget,att,queue_size=300)
assert rospy.get_param('/network_controller_node/stagec_watchdog_enable',None)==cfg['native_watchdog_enabled'],'NATIVE_PARAMETER_BINDING'
start=time.monotonic();result=None
while time.monotonic()-start<35 and not rospy.is_shutdown():
 if (out/'GATEWAY_KILLED.json').exists():
  event=json.loads((out/'GATEWAY_KILLED.json').read_text())
  if time.time_ns()-event['wall_ns']>=7_000_000_000:
   with lock:
    motion=[x for x in rows if event['wall_ns']<=x['wall_ns']<=event['wall_ns']+6_500_000_000];assert motion
    p0=np.array(motion[0]['p']);distance=max(float(np.linalg.norm(np.array(x['p'])-p0)) for x in motion);tail=[x for x in motion if x['wall_ns']>=motion[-1]['wall_ns']-1_000_000_000];stopped=bool(tail and motion[-1]['wall_ns']-tail[0]['wall_ns']>=980_000_000 and all(np.linalg.norm(x['v'])<=.05 for x in tail));bounds=bool(watch and all(x['bounded_reference'] for x in watch));dz=max(abs(x['p'][2]-2) for x in motion)
    result=dict(status='DYNAMICS_STOP_PASS' if stopped and distance<=1.25 and dz<=.15 and bounds else 'DYNAMICS_STOP_FAIL',maximum_stop_displacement_m=distance,altitude_error_m=dz,stopped_hold=stopped,initial_speed_mps=float(np.linalg.norm(motion[0]['v'])),final_speed_mps=float(np.linalg.norm(motion[-1]['v'])),watchdog_samples=len(watch),first_watchdog_after_kill_s=(watch[0]['wall_ns']-event['wall_ns'])*1e-9 if watch else None,reference_bounds_proved_by_Bernstein=bounds,FALLBACK_VERIFIED=False,space_certificate='BLOCKED',motor_RPM='NOT_VERIFIED',native_watchdog_enabled=cfg['native_watchdog_enabled']);break
 time.sleep(.02)
if result is None:result=dict(status='BLOCKED',reason='NO_COMPLETE_GATEWAY_LOSS_OBSERVATION',FALLBACK_VERIFIED=False)
(out/'NATIVE_WATCHDOG_RESULT.json').write_text(json.dumps(result,indent=2))
for f in logs.values():f.close()
rospy.signal_shutdown('native observer complete')
