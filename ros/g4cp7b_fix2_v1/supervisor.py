"""Native sensor/command supervisor;all controller publication follows locked admission."""
from pathlib import Path
import sys,argparse,json,time,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--forest',action='store_true');a=p.parse_args()
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Imu,Image,CameraInfo
 from quadrotor_msgs.msg import SO3Command,PositionCommand
 from std_msgs.msg import String
 from query_minco.integration.yopo_execution_v4.readiness import Readiness
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 from query_minco.integration.yopo_execution_v4.trajectory import Admission
 from query_minco.integration.forest_fix2_v1.contract import certify
 from query_minco.integration.forest_fix2_v1.pairing import Pairing
 from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace
 from query_minco.integration.yopo_shadow_v1.command_contract import position_command
 run=Path(a.run);out=run/'supervisor_data';out.mkdir();rospy.init_node('p2_execution_supervisor')
 journal=Journal(out/'EVENTS.jsonl');r=Readiness(True,True,'equilibrium_hover');gate=Admission();last={};poses=collections.OrderedDict();camera=None;space=None;pairer=Pairing();faults=[];mode='RUNNING';stop_confirmed=None
 def log(kind,stamp,data,accepted=True):
  journal.append(dict(kind=kind,stamp=stamp,ros_now=rospy.Time.now().to_sec(),monotonic=time.monotonic(),mode=r.mode,lifecycle=mode,payload=data,accepted=accepted))
 def abort(reason):
  if r.mode!='ABORT':
   r.abort(reason);faults.append(dict(reason=str(reason),monotonic=time.monotonic(),ros_now=rospy.Time.now().to_sec()));log('fault',None,faults[-1],False)
 def accept(k,stamp,d):
  ok=r.observe(k,stamp,rospy.Time.now().to_sec(),d);log(k,stamp,d,ok)
  if ok:last[k]=d
  return ok
 def odom(m):
  with journal.callback() as active:
   if not active or mode=='TERMINATING':return
   q=m.pose.pose;v=m.twist.twist.linear;s=m.header.stamp.to_sec()
   d=dict(frame=m.header.frame_id,p=[q.position.x,q.position.y,q.position.z],v=[v.x,v.y,v.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w])
   if accept('odom',s,d):
    poses[s]=d
    pairer.pose(s,d)
    while len(poses)>512:poses.popitem(last=False)
    drain('ODOM_ARRIVAL')
 def imu(m):
  with journal.callback() as active:
   if active and mode!='TERMINATING':accept('imu',m.header.stamp.to_sec(),dict(frame=m.header.frame_id,a_world=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z]))
 def cam(m):
  nonlocal camera
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='camera_link' or list(m.K)!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:abort('CAMERA_CONTRACT')
   else:camera=m;drain('CAMERA_ARRIVAL')
 def drain(trigger):
  nonlocal space
  if camera is None:return
  now=rospy.Time.now().to_sec()
  for s,item,pose in pairer.take(now):
   m,arrival,arrival_ros=item
   try:
    d=np.ndarray((90,160),dtype='<f4',buffer=m.data,strides=(m.step,4)).copy();valid=np.isfinite(d)&(d>=.04)&(d<20)
    okay=accept('depth',s,dict(frame=m.header.frame_id,stamp=s,source_odom_stamp=s,valid_count=int((np.isfinite(d)&(d>0)&(d<=20)).sum()),observed_hit_pixels=int(valid.sum())))
    if okay:space=ObservedSpace(d,pose,s)
    journal.append(dict(kind='depth_chain',source_stamp=s,receipt_monotonic=arrival,receipt_ros=arrival_ros,processed_monotonic=time.monotonic(),processed_ros=rospy.Time.now().to_sec(),trigger=trigger,accepted=okay,reason='MATCHED' if okay else r.reason,pending_depth=len(pairer.depths),pose_cache=len(pairer.poses)))
   except Exception as e:abort('DEPTH '+repr(e))
  while pairer.drops:
   stamp,reason=pairer.drops.pop(0);journal.append(dict(kind='depth_chain',source_stamp=stamp,processed_monotonic=time.monotonic(),accepted=False,reason=reason))
 def depth(m):
  arrival=time.monotonic();arrival_ros=rospy.Time.now().to_sec()
  with journal.callback() as active:
   if not active or mode=='TERMINATING':return
   s=m.header.stamp.to_sec()
   journal.append(dict(kind='depth_arrival',source_stamp=s,receipt_monotonic=arrival,receipt_ros=arrival_ros,lock_acquired_monotonic=time.monotonic(),pending_depth=len(pairer.depths),configured_depth_queue=2,receive_buffer_bytes=1048576))
   if m.header.frame_id!='camera_link' or m.encoding!='32FC1' or m.is_bigendian or (m.height,m.width)!=(90,160) or m.step<640:abort('DEPTH_FORMAT');return
   pairer.depth(s,(m,arrival,arrival_ros));drain('DEPTH_ARRIVAL')
 def request(m):
  with journal.callback() as active:
   if not active or mode=='TERMINATING':return
   try:
    envelope=json.loads(m.data);s=envelope['stamp'];d=dict(frame=envelope['frame'],pva=envelope['pva'])
    if 'odom' not in last or 'imu' not in last:return
    actual=dict(last['odom'],a=last['imu']['a_world'])
    if envelope.get('fatal'):raise ValueError(envelope['fatal'])
    cert=None
    if a.forest:
     if space is None or rospy.Time.now().to_sec()-space.stamp>.2:raise ValueError('UNKNOWN_STALE_GEOMETRY')
     cert=space.certify
    pva=gate.check(envelope,actual,rospy.Time.now().to_sec(),certify=cert)
    if (run/'BIRTH_CERTIFICATE.json').exists():
     certify(run,plan=gate.plan if envelope['kind']=='trajectory' else None,pva=pva,now=rospy.Time.now().to_sec())
    elif r.mode=='ARMED':raise ValueError('BIRTH_CERTIFICATE_MISSING')
    if accept('request',s,d):
     msg=position_command(PositionCommand,pva,rospy.Time.from_sec(s),0.,envelope.get('sequence',1))
     # Preserve baseline effective gains via zero overrides.
     msg.kx=[0.,0.,0.];msg.kv=[0.,0.,0.]
     command.publish(msg);log('controller_command',s,dict(pva=pva.tolist(),plan_id=gate.plan.id if gate.plan else None))
   except Exception as e:abort('REQUEST '+repr(e))
 def raw(m):
  with journal.callback() as active:
   if not active or mode=='TERMINATING':return
   s=m.header.stamp.to_sec();d=dict(frame=m.header.frame_id,force_world=[m.force.x,m.force.y,m.force.z],q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],kR=list(m.kR),kOm=list(m.kOm),motors_enabled=bool(m.aux.enable_motors))
   if accept('raw',s,d) and r.forward_allowed(rospy.Time.now().to_sec()):
    final.publish(m);r.forwarded(s);log('final_so3',s,d)
 def plant(m):
  with journal.callback() as active:
   if not active or mode=='TERMINATING':return
   try:
    d=json.loads(m.data);accept('plant',d['stamp'],d)
   except Exception as e:abort('PLANT '+repr(e))
 command=rospy.Publisher('/p2/position_cmd',PositionCommand,queue_size=1);final=rospy.Publisher('/p2/so3_cmd',SO3Command,queue_size=1)
 subs=[rospy.Subscriber('/p2/odom',Odometry,odom,queue_size=10,tcp_nodelay=True),rospy.Subscriber('/p2/imu',Imu,imu,queue_size=10,tcp_nodelay=True),rospy.Subscriber('/p2/depth',Image,depth,queue_size=2,buff_size=1048576,tcp_nodelay=True),rospy.Subscriber('/p2/camera_info',CameraInfo,cam,queue_size=1),rospy.Subscriber('/p2/request',String,request,queue_size=10,tcp_nodelay=True),rospy.Subscriber('/p2/so3_raw',SO3Command,raw,queue_size=10,tcp_nodelay=True),rospy.Subscriber('/p2/startup_state',String,plant,queue_size=10,tcp_nodelay=True)]
 rate=rospy.Rate(100);errors=[]
 try:
  while not rospy.is_shutdown():
   with journal.lock:
    control=run/'STOP_CONFIRMED.json'
    if control.exists():
     stop_confirmed=json.loads(control.read_text());mode='TERMINATING';log('stop_confirmed',None,stop_confirmed,False);break
    before=r.mode;r.tick(rospy.Time.now().to_sec())
    if r.mode=='ABORT' and not faults:
     faults.append(dict(reason=r.reason,monotonic=time.monotonic()));log('fault',None,faults[-1],False)
    if before!=r.mode:log('transition',None,dict(before=before,after=r.mode),False)
    if 'imu' in last and np.linalg.norm(last['imu']['a_world'])>8+1e-6:abort('MEASURED_ACCELERATION')
    atomic_json(out/'HEARTBEAT.json',dict(monotonic=time.monotonic(),ros_now=rospy.Time.now().to_sec(),mode=r.mode,lifecycle=mode,reason=r.reason,counts=r.counts,ack_stamps=sorted(r.acks),plant=last.get('plant'),state=last.get('odom'),imu=last.get('imu'),odom_stamp=r.samples.get('odom',(0,))[0],plan_id=gate.plan.id if gate.plan else None))
   rate.sleep()
 except rospy.ROSInterruptException:pass
 except BaseException as e:errors.append(repr(e))
 finally:
  close_errors=journal.close([s.unregister for s in subs]);mode='TERMINATED'
  atomic_json(out/'RESULT.json',dict(lifecycle=mode,stop_confirmed=stop_confirmed,faults=faults,errors=errors,cleanup_errors=close_errors,status='PASS' if stop_confirmed and not errors and not close_errors else 'FAIL',late_callbacks_ignored=journal.late_callbacks))
 if errors or close_errors or not stop_confirmed:raise SystemExit(2)
if __name__=='__main__':main()
