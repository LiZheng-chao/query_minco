"""Real controller consumer observation and fault injection;no quadrotor process."""
from pathlib import Path
import argparse,time,json,os,signal
import numpy as np,rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from quadrotor_msgs.msg import PositionCommand,SO3Command
def main():
 p=argparse.ArgumentParser()
 for name in ['prefix','fault','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--fault-at',type=float,required=True);p.add_argument('--controller-pgid',type=int,required=True);p.add_argument('--planner-pgid',type=int,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir();rospy.init_node('consumption_probe',anonymous=True)
 statuses=[];raw=[];forward=[];commands=[]
 rospy.Subscriber(a.prefix+'/status',String,lambda m:statuses.append(json.loads(m.data)),queue_size=500)
 rospy.Subscriber(a.prefix+'/so3_raw',SO3Command,lambda m:raw.append(dict(t=time.monotonic(),force=[m.force.x,m.force.y,m.force.z],stamp=m.header.stamp.to_sec())),queue_size=1000)
 rospy.Subscriber(a.prefix+'/so3_cmd',SO3Command,lambda m:forward.append(dict(t=time.monotonic(),force=[m.force.x,m.force.y,m.force.z])),queue_size=1000)
 rospy.Subscriber(a.prefix+'/position_cmd',PositionCommand,lambda m:commands.append(dict(t=time.monotonic(),p=[m.position.x,m.position.y,m.position.z],v=[m.velocity.x,m.velocity.y,m.velocity.z],a=[m.acceleration.x,m.acceleration.y,m.acceleration.z],kx=list(m.kx),kv=list(m.kv))),queue_size=1000)
 odom=rospy.Publisher(a.prefix+'/odom',Odometry,queue_size=1);depth=rospy.Publisher(a.prefix+'/depth',Image,queue_size=1)
 rate=rospy.Rate(100);injected=False;start=time.monotonic();counter=0
 while not rospy.is_shutdown() and time.monotonic()<a.fault_at+1.3:
  t=time.monotonic();fault=t>=a.fault_at
  if fault and not injected:
   if a.fault=='controller_fault':os.killpg(a.controller_pgid,signal.SIGINT)
   if a.fault=='planner_stop':os.killpg(a.planner_pgid,signal.SIGKILL)
   injected=True
  m=Odometry();m.header.stamp=rospy.Time.now();m.header.frame_id='world';m.child_frame_id='body';m.pose.pose.position.z=2.;m.pose.pose.orientation.w=1.
  if a.fault=='odom_delay' and fault:m.header.stamp=rospy.Time.from_sec(m.header.stamp.to_sec()-.4)
  odom.publish(m);counter+=1
  if counter%5==0 and not(a.fault=='depth_loss' and fault):
   d=Image();d.header.stamp=rospy.Time.now();d.header.frame_id='camera_link';d.height=1;d.width=1;d.encoding='32FC1';d.step=4;d.data=np.array([2.],np.float32).tobytes();depth.publish(d)
  rate.sleep()
 aborted=[x for x in statuses if x['mode']=='ABORT'];active=[x for x in statuses if x['mode']=='ACTIVE'];assert aborted and active,'ACTIVE_OR_ABORT_NOT_OBSERVED'
 abort=aborted[0]['monotonic'];healthy=[x for x in raw if a.fault_at-1<x['t']<a.fault_at-.1];post=[x for x in raw if x['t']>abort+.2]
 hold=[x for x in commands if x['t']>abort+.1];final_after=[x for x in forward if x['t']>abort+.05]
 consumption=bool(healthy and max(abs(x['force'][0]) for x in healthy)>.01)
 hold_verified=bool(hold and all(np.linalg.norm(np.asarray(x['p'])-np.array([0,0,2]))<1e-9 and np.linalg.norm(x['v'])==0 and np.linalg.norm(x['a'])==0 for x in hold))
 controller_replaced=bool(post and max(abs(x['force'][0]) for x in post)<1e-8) if a.fault!='controller_fault' else True
 passed=consumption and hold_verified and controller_replaced and not final_after and abort<=a.fault_at+.4
 result=dict(status='PASS' if passed else 'FAIL',fault=a.fault,actual_native_controller_consumed=consumption,hold_message_verified=hold_verified,
  actual_so3_old_reference_replaced=controller_replaced,raw_max_healthy_fx=max(abs(x['force'][0]) for x in healthy) if healthy else None,
  raw_max_post_abort_fx=max(abs(x['force'][0]) for x in post) if post else None,final_so3_after_abort=len(final_after),
  detection_delay_s=abort-a.fault_at,reason=aborted[0]['reason'],gain_overrides_zero=all(not any(x['kx']+x['kv']) for x in commands),
  no_physics_process=True,physical_braking='NOT_RUN',status_records=statuses,raw_so3=raw,forwarded_so3=forward,position_commands=commands)
 with (out/'RESULT.json').open('x') as f:json.dump(result,f,indent=2)
 print('ACTUAL_CONTROLLER_CASE',a.fault,result['status'],result['reason'],result['detection_delay_s'],flush=True)
 assert passed,result
if __name__=='__main__':main()
