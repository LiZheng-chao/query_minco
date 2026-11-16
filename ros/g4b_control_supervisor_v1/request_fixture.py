"""Explicit synthetic command source for no-plant consumption/fault tests."""
import argparse,time,os,rospy
from quadrotor_msgs.msg import PositionCommand
def main():
 p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--fault',required=True);p.add_argument('--fault-at',type=float,required=True);a=p.parse_args()
 rospy.init_node('request_fixture',anonymous=True);pub=rospy.Publisher(a.prefix+'/request',PositionCommand,queue_size=1);rate=rospy.Rate(50)
 try:
  while not rospy.is_shutdown():
   now=time.monotonic()
   if not(a.fault=='timeout' and now>=a.fault_at):
    m=PositionCommand();m.header.stamp=rospy.Time.now();m.header.frame_id='world';m.position.x=.02;m.position.z=2;m.trajectory_id=1;m.trajectory_flag=m.TRAJECTORY_STATUS_READY
    if a.fault=='nan' and now>=a.fault_at:m.position.x=float('nan')
    pub.publish(m)
   rate.sleep()
 except rospy.ROSInterruptException:pass
if __name__=='__main__':main()
