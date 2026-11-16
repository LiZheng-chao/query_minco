"""Native-input typed MINCO candidate bridge. Never publishes executable commands."""
from pathlib import Path
import sys,argparse,time,collections,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=10);a=p.parse_args()
 if not 1<=a.duration<=60:raise ValueError('DURATION')
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo,Imu
 from quadrotor_msgs.msg import PositionCommand,SO3Command
 from query_minco.integration.yopo_native_bridge_v1.contract import acceleration,inputs,reference,imu_fresh
 from query_minco.integration.yopo_forest_bridge_v1.pairing import Pairing
 from query_minco.integration.yopo_shadow_v1.command_contract import position_command
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);journal=Journal(out/'EVENTS.jsonl')
 rospy.init_node('query_minco_native_shadow_bridge')
 poses=collections.OrderedDict();cache={};errors=[];plans=0;last_plan=None;pairer=Pairing();fixed_goal=None
 def odom(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='world':errors.append('ODOM_FRAME');return
   q=m.pose.pose;v=m.twist.twist.linear;s=m.header.stamp.to_sec()
   poses[s]=dict(stamp=s,p=[q.position.x,q.position.y,q.position.z],v=[v.x,v.y,v.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w])
   pairer.add('pose',s,poses[s],rospy.Time.now().to_sec())
   while len(poses)>512:poses.popitem(last=False)
 def store(k,m):
  with journal.callback() as active:
   if active:
    cache[k]=m
    if k=='depth':pairer.add('depth',m.header.stamp.to_sec(),m,rospy.Time.now().to_sec())
 pub=rospy.Publisher('/query_minco/p5/position_candidate',PositionCommand,queue_size=1)
 subs=[rospy.Subscriber('/sim/odom',Odometry,odom,queue_size=20),rospy.Subscriber('/sim/imu',Imu,lambda m:store('imu',m),queue_size=10),
  rospy.Subscriber('/depth_image',Image,lambda m:store('depth',m),queue_size=2),rospy.Subscriber('/camera_info',CameraInfo,lambda m:store('camera',m),queue_size=1),
  rospy.Subscriber('/so3_cmd',SO3Command,lambda m:store('so3',m),queue_size=10)]
 begin=time.monotonic();rate=rospy.Rate(100);rejections=collections.Counter()
 try:
  while not rospy.is_shutdown() and time.monotonic()-begin<a.duration:
   with journal.lock:
    c=dict(cache);ps=dict(poses);matched=pairer.latest(rospy.Time.now().to_sec())
   now=rospy.Time.now().to_sec();acc=None
   try:
    if 'depth' not in c or 'camera' not in c:raise ValueError('MISSING_SENSOR')
    if matched is None:raise ValueError('WAITING_FOR_FRESH_MATCH')
    s,m,paired_state=matched;camera=c['camera']
    if m.header.frame_id!='camera_link' or camera.header.frame_id!='camera_link' or (m.height,m.width)!=(90,160) or m.encoding!='32FC1' or m.is_bigendian or m.step<640:raise ValueError('DEPTH_CONTRACT')
    if s not in ps:raise ValueError('NO_EXACT_POSE')
    if abs(camera.header.stamp.to_sec()-s)>.05:raise ValueError('CAMERA_STALE')
    prior=[x for x in ps if x<s]
    if not prior:raise ValueError('NO_ACCELERATION_HISTORY')
    state=paired_state;acc=acceleration(ps[max(prior)],state)
    if fixed_goal is None:fixed_goal=np.asarray(state['p'])+[.4,0,0]
    goal=fixed_goal
    d=np.ndarray((90,160),dtype='<f4',buffer=m.data,strides=(m.step,4)).copy()
    x,st,R=inputs(d,state,acc,goal,list(camera.K),s,now)
    if last_plan is None or now-last_plan.epoch>=.2:
     last_plan=reference(state,acc,goal);plans+=1
     journal.append(dict(kind='plan',stamp=s,record=last_plan.record,maxima=last_plan.maxima,state12=st.tolist(),acceleration_source='actual odom velocity finite difference;shadow only'))
    msg=position_command(PositionCommand,last_plan.at(now),rospy.Time.from_sec(now),0.,plans)
    msg.kx=[0.,0.,0.];msg.kv=[0.,0.,0.];pub.publish(msg)
    imu_status='MISSING'
    if 'imu' in c:
     try:imu_fresh(c['imu'].header.stamp.to_sec(),now);imu_status='VALID'
     except ValueError:imu_status='INVALID_SOURCE_STAMP'
    journal.append(dict(kind='candidate',stamp=now,plan_id=last_plan.id,imu=imu_status,so3_seen='so3' in c,control_authority=False))
   except Exception as e:
    rejections[str(e)]+=1
    journal.append(dict(kind='rejection',reason=str(e),receipt_ros=now,source_stamp=matched[0] if matched else None,state=matched[2] if matched else None,a_world=acc.tolist() if acc is not None else None,goal=fixed_goal.tolist() if fixed_goal is not None else None))
   rate.sleep()
 except rospy.ROSInterruptException:pass
 finally:
  cleanup=journal.close([s.unregister for s in subs]);atomic_json(out/'RESULT.json',dict(status='SHADOW_ONLY',plans=plans,rejections=dict(rejections),errors=errors,cleanup=cleanup,physical_execution='NOT_RUN',control_authority=False))
if __name__=='__main__':main()
