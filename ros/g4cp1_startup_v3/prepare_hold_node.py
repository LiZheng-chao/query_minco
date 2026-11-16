"""Version3 stationary HOLD only;native observations and explicit state handshake."""
from pathlib import Path
import sys,argparse,json,threading,time,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare-authorized',action='store_true');p.add_argument('--arm-authorized',action='store_true');p.add_argument('--motor-initial-state',choices=['zero','equilibrium_hover'],default='zero');a=p.parse_args()
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Imu,Image,CameraInfo
 from quadrotor_msgs.msg import PositionCommand,SO3Command
 from std_msgs.msg import String
 from query_minco.integration.yopo_startup_v3.readiness import Readiness
 out=Path(a.output);out.mkdir();rospy.init_node('g4cp1_hold_content_gate')
 r=Readiness(a.prepare_authorized,a.arm_authorized,a.motor_initial_state);lock=threading.RLock();last={};poses=collections.OrderedDict();camera=None;anchor=None;previous_mode='HELD'
 stream=(out/'EVENTS.jsonl').open('x')
 def log(kind,stamp,data):
  record=dict(kind=kind,stamp=stamp,ros_now=rospy.Time.now().to_sec(),monotonic=time.monotonic(),mode=r.mode,data=data)
  try:text=json.dumps(record,allow_nan=False)
  except ValueError:text=json.dumps(dict(kind=kind,stamp_repr=repr(stamp),payload_repr=repr(data),mode=r.mode))
  stream.write(text+'\n');stream.flush()
 def receive(kind,mstamp,data):
  ok=r.observe(kind,mstamp,rospy.Time.now().to_sec(),data);log(kind,mstamp,dict(accepted=ok,payload=data,reason=r.reason))
  if ok:last[kind]=data
  return ok
 def odom(m):
  with lock:
   q=m.pose.pose;v=m.twist.twist.linear;stamp=m.header.stamp.to_sec()
   d=dict(frame=m.header.frame_id,p=[q.position.x,q.position.y,q.position.z],v=[v.x,v.y,v.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w])
   if receive('odom',stamp,d):
    poses[stamp]=d
    while len(poses)>512:poses.popitem(last=False)
 def imu(m):
  with lock:receive('imu',m.header.stamp.to_sec(),dict(frame=m.header.frame_id,a_world=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z]))
 def cam(m):
  nonlocal camera
  with lock:
   if m.header.frame_id!='camera_link' or m.width!=160 or m.height!=90 or list(m.K)!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:r.abort('CAMERA_CALIBRATION')
   else:camera=m
 def depth(m):
  with lock:
   try:
    stamp=m.header.stamp.to_sec()
    if m.encoding!='32FC1' or m.is_bigendian or m.width!=160 or m.height!=90 or m.step<640:raise ValueError('DEPTH_ENCODING')
    if camera is None:return
    # An exact header match proves which actual native odometry sample generated this image.
    # Startup races wait for a matched frame;they never synthesize a pose or refresh a stamp.
    if stamp not in poses:return
    d=np.ndarray((m.height,m.width),dtype='<f4',buffer=m.data,strides=(m.step,4))
    mask=np.isfinite(d)&(d>0)&(d<=20)
    receive('depth',stamp,dict(frame=m.header.frame_id,stamp=stamp,source_odom_stamp=stamp,source_pose=poses[stamp],valid_count=int(mask.sum()),min_m=float(d[mask].min()) if mask.any() else None,max_m=float(d[mask].max()) if mask.any() else None))
   except Exception as e:r.abort('DEPTH '+repr(e))
 def raw(m):
  with lock:
   stamp=m.header.stamp.to_sec()
   d=dict(frame=m.header.frame_id,force_world=[m.force.x,m.force.y,m.force.z],q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],kR=list(m.kR),kOm=list(m.kOm),motors_enabled=bool(m.aux.enable_motors))
   if not receive('raw',stamp,d):return
   if r.forward_allowed(rospy.Time.now().to_sec()):
    final.publish(m);r.forwarded(stamp);log('final_so3_sent',stamp,d)
 def plant(m):
  with lock:
   try:
    d=json.loads(m.data);receive('plant',d['stamp'],d)
   except Exception as e:r.abort('PLANT '+repr(e))
 final=rospy.Publisher('/g4cp1/so3_cmd',SO3Command,queue_size=1)
 controller=rospy.Publisher('/g4cp1/position_cmd',PositionCommand,queue_size=1)
 request=rospy.Publisher('/g4cp1/request',PositionCommand,queue_size=1)
 subs=[rospy.Subscriber('/g4cp1/odom',Odometry,odom,queue_size=10),rospy.Subscriber('/g4cp1/imu',Imu,imu,queue_size=10),
  rospy.Subscriber('/g4cp1/depth',Image,depth,queue_size=2),rospy.Subscriber('/g4cp1/camera_info',CameraInfo,cam,queue_size=1),
  rospy.Subscriber('/g4cp1/so3_raw',SO3Command,raw,queue_size=10),rospy.Subscriber('/g4cp1/startup_state',String,plant,queue_size=10)]
 rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown():
   with lock:
    now=rospy.Time.now();seconds=now.to_sec()
    if a.prepare_authorized and 'odom' in last and r.mode!='ABORT':
     if anchor is None:anchor=list(last['odom']['p'])
     if np.linalg.norm(np.array(last['odom']['p'])-anchor)>.25:r.abort('HOLD_POSITION_ERROR')
     if r.mode!='ABORT':
      msg=PositionCommand();msg.header.stamp=now;msg.header.frame_id='world';msg.trajectory_flag=1
      msg.position.x,msg.position.y,msg.position.z=anchor;msg.kx=[0.,0.,0.];msg.kv=[0.,0.,0.]
      data=dict(frame='world',pva=[anchor,[0.,0.,0.],[0.,0.,0.]])
      if receive('request',seconds,data):request.publish(msg);controller.publish(msg)
    mode=r.tick(seconds)
    if mode!=previous_mode:log('transition',seconds,dict(before=previous_mode,after=mode,reason=r.reason));previous_mode=mode
    payload=dict(odom_stamp=r.samples.get('odom',(0,))[0],imu_stamp=r.samples.get('imu',(0,))[0],depth_stamp=r.samples.get('depth',(0,))[0],mode=mode,reason=r.reason,monotonic=time.monotonic(),ros_now=seconds,counts=r.counts,duplicate_counts=r.duplicates,
     ack_stamps=sorted(r.acks),last_sent=r.last_sent,plant=last.get('plant'),state=last.get('odom'),request=last.get('request'),imu=last.get('imu'),depth=last.get('depth'),transitions=r.transitions)
    temporary=out/'HEARTBEAT.tmp';temporary.write_text(json.dumps(payload,allow_nan=False));temporary.replace(out/'HEARTBEAT.json')
    if mode=='ABORT':raise RuntimeError('LATCHED '+str(r.reason))
   rate.sleep()
 finally:stream.close()
if __name__=='__main__':main()
