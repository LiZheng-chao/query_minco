"""Manual-only HOLD bootstrap. No trajectory execution,network,or online Hard."""
from pathlib import Path
import sys,argparse,json,threading,time,math
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--authorize-bootstrap-hold',action='store_true');a=p.parse_args()
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Imu,Image,CameraInfo
 from quadrotor_msgs.msg import PositionCommand,SO3Command
 from std_msgs.msg import String
 from query_minco.integration.yopo_startup_v2.readiness import Readiness
 out=Path(a.output);out.mkdir();rospy.init_node('g4c_explicit_hold_prepare')
 ready=Readiness(a.authorize_bootstrap_hold);lock=threading.RLock();last={};camera=None;anchor=None
 events=(out/'EVENTS.jsonl').open('x')
 def event(kind,stamp,payload):
  events.write(json.dumps(dict(monotonic=time.monotonic(),ros_now=rospy.Time.now().to_sec(),kind=kind,stamp=stamp,payload=payload),allow_nan=False)+'\n');events.flush()
 def receive(kind,stamp,data):
  with lock:
   ready.observe(kind,stamp,rospy.Time.now().to_sec(),data);last[kind]=data
   event(kind,stamp,data)
 def odom(m):
  q=m.pose.pose;p0=[q.position.x,q.position.y,q.position.z];v=m.twist.twist.linear
  receive('odom',m.header.stamp.to_sec(),dict(p=p0,v=[v.x,v.y,v.z],quaternion=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w]))
 def imu(m):
  aa=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z]
  if not np.isfinite(aa).all():ready.abort('IMU_NAN');return
  receive('imu',m.header.stamp.to_sec(),dict(acceleration_world=aa))
 def depth(m):
  try:
   if m.encoding!='32FC1' or m.is_bigendian or m.width!=160 or m.height!=90 or m.step<640:raise ValueError('DEPTH_CONTRACT')
   d=np.ndarray((m.height,m.width),dtype='<f4',buffer=m.data,strides=(m.step,4))
   valid=np.isfinite(d)&(d>0)&(d<=20)
   if not valid.any():raise ValueError('NO_VALID_DEPTH')
   if camera is None:return
   if list(camera.K)!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:raise ValueError('CAMERA_CONTRACT')
   receive('depth',m.header.stamp.to_sec(),dict(valid_count=int(valid.sum()),range_m=[float(d[valid].min()),float(d[valid].max())],encoding=m.encoding))
  except Exception as e:ready.abort(str(e))
 def cam(m):
  nonlocal camera
  camera=m
 def raw(m):
  receive('raw',m.header.stamp.to_sec(),dict(force=[m.force.x,m.force.y,m.force.z],orientation=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w]))
  with lock:
   q=np.array(last['raw']['orientation']);n=float(np.linalg.norm(q))
   if not np.isfinite(q).all() or abs(n-1)>1e-3:ready.abort('RAW_QUATERNION')
   if ready.forward_allowed(rospy.Time.now().to_sec()):
    final.publish(m);event('final_forward',m.header.stamp.to_sec(),last['raw'])
 def plant(m):
  try:
   d=json.loads(m.data);receive('plant',d['stamp'],d)
  except Exception as e:ready.abort('PLANT_TELEMETRY '+repr(e))
 final=rospy.Publisher('/g4c/so3_cmd',SO3Command,queue_size=1)
 command=rospy.Publisher('/g4c/position_cmd',PositionCommand,queue_size=1)
 req=rospy.Publisher('/g4c/bootstrap_request',PositionCommand,queue_size=1)
 subscribers=[rospy.Subscriber('/g4c/odom',Odometry,odom,queue_size=1),rospy.Subscriber('/g4c/imu',Imu,imu,queue_size=1),
  rospy.Subscriber('/g4c/depth',Image,depth,queue_size=1),rospy.Subscriber('/g4c/camera_info',CameraInfo,cam,queue_size=1),
  rospy.Subscriber('/g4c/so3_raw',SO3Command,raw,queue_size=1),rospy.Subscriber('/g4c/startup_state',String,plant,queue_size=1)]
 rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown():
   with lock:
    now=rospy.Time.now()
    if a.authorize_bootstrap_hold and 'odom' in last and ready.mode!='ABORT':
     pos=last['odom']['p']
     if anchor is None:anchor=list(pos)
     if np.linalg.norm(np.array(pos)-anchor)>.25:ready.abort('HOLD_POSITION_ERROR');continue
     pos=anchor;m=PositionCommand();m.header.stamp=now;m.header.frame_id='world';m.trajectory_flag=1
     m.position.x,m.position.y,m.position.z=pos;m.kx=[0.,0.,0.];m.kv=[0.,0.,0.]
     # HOLD request is explicitly authorized;WAIT still blocks all final SO3 execution.
     pva=[pos,[0.,0.,0.],[0.,0.,0.]];receive('request',now.to_sec(),pva);req.publish(m);command.publish(m)
    mode=ready.tick(now.to_sec())
    payload=dict(mode=mode,reason=ready.reason,monotonic=time.monotonic(),ros_now=now.to_sec(),counts=ready.counts,plant=last.get('plant'),state=last.get('odom'),authorized_hold=a.authorize_bootstrap_hold)
    temp=out/'HEARTBEAT.tmp';temp.write_text(json.dumps(payload));temp.replace(out/'HEARTBEAT.json')
    if mode=='ABORT':raise RuntimeError('STARTUP_ABORT '+str(ready.reason))
   rate.sleep()
 finally:events.close()
if __name__=='__main__':main()
