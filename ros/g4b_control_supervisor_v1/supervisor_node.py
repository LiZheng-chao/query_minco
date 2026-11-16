"""Actual PositionCommand consumer supervisor and internal SO3 gate;no physics launch."""
from pathlib import Path
import sys,json,argparse,time,threading,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from quadrotor_msgs.msg import PositionCommand,SO3Command
from query_minco.integration.yopo_control_supervisor_v1.lease import Lease
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation
from query_minco.integration.yopo_shadow_v1.command_contract import position_command
def main():
 p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--output',required=True);p.add_argument('--message-test-profile',action='store_true');args=p.parse_args()
 out=Path(args.output);out.mkdir();rospy.init_node('lease_supervisor',anonymous=True)
 lease=Lease(.2);lock=threading.RLock();last_raw=[None];records=[];pub=rospy.Publisher(args.prefix+'/position_cmd',PositionCommand,queue_size=1)
 actuator=rospy.Publisher(args.prefix+'/so3_cmd',SO3Command,queue_size=1);status=rospy.Publisher(args.prefix+'/status',String,queue_size=1)
 def odom(m):
  with lock:
   try:rotation([m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w])
   except ValueError:return lease.trip('odom_QUATERNION',time.monotonic())
   lease.odom(time.monotonic(),m.header.stamp.to_sec(),rospy.Time.now().to_sec(),[m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z],[m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z])
 def depth(m):
  with lock:
   valid=m.encoding=='32FC1' and not m.is_bigendian and m.step==m.width*4 and len(m.data)==m.height*m.step and len(m.data)>0
   valid=valid and (args.message_test_profile or (m.height,m.width)==(90,160))
   if valid:
    d=np.frombuffer(m.data,dtype='<f4');valid=bool((np.isfinite(d)&(d>=.04)&(d<20)).mean()>=.01)
   lease.depth(time.monotonic(),m.header.stamp.to_sec(),rospy.Time.now().to_sec(),valid)
 def request(m):
  with lock:
   if m.header.frame_id!='world' or m.trajectory_flag!=m.TRAJECTORY_STATUS_READY:return lease.trip('request_FRAME_OR_STATUS',time.monotonic())
   lease.command(time.monotonic(),m.header.stamp.to_sec(),rospy.Time.now().to_sec(),[[m.position.x,m.position.y,m.position.z],[m.velocity.x,m.velocity.y,m.velocity.z],[m.acceleration.x,m.acceleration.y,m.acceleration.z]])
 def raw(m):
  with lock:
   lease.raw(time.monotonic(),m.header.stamp.to_sec(),rospy.Time.now().to_sec(),[m.force.x,m.force.y,m.force.z,m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w])
   last_raw[0]=m
   if lease.forward_allowed(time.monotonic()):actuator.publish(m)
 subscribers=[rospy.Subscriber(args.prefix+'/odom',Odometry,odom,queue_size=1),rospy.Subscriber(args.prefix+'/depth',Image,depth,queue_size=1),
  rospy.Subscriber(args.prefix+'/request',PositionCommand,request,queue_size=1),rospy.Subscriber(args.prefix+'/so3_raw',SO3Command,raw,queue_size=1)]
 rate=rospy.Rate(100);identifier=0
 try:
  while not rospy.is_shutdown():
   with lock:
    now=time.monotonic();pva=lease.tick(now);identifier+=1
    if pva is not None:
     msg=position_command(PositionCommand,pva,rospy.Time.now(),0.,identifier)
     # Zero gain overrides preserve the exact frozen launch/controller gains.
     msg.kx=[0.,0.,0.];msg.kv=[0.,0.,0.];pub.publish(msg)
    event=dict(monotonic=now,ros_stamp=rospy.Time.now().to_sec(),mode=lease.mode,reason=lease.reason,
      final_so3_gate_open=lease.forward_allowed(now),last_receipts={k:v[0] for k,v in lease.last.items()})
    status.publish(json.dumps(event));records.append(event)
    temporary=out/'heartbeat.tmp'
    with temporary.open('w') as f:json.dump(event,f)
    os.replace(temporary,out/'HEARTBEAT.json')
   rate.sleep()
 except rospy.ROSInterruptException:pass
 finally:
  with (out/'RESULT.json').open('x') as f:json.dump(dict(transitions=lease.transitions,records=records,mode=lease.mode,reason=lease.reason),f,indent=2)
if __name__=='__main__':main()
