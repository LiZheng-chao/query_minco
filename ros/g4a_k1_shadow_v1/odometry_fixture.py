"""Non-control scripted odometry input to the real native forest sensor."""
from pathlib import Path
import sys,json,argparse,math,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
def main():
 p=argparse.ArgumentParser();p.add_argument('--origin',nargs=3,type=float,required=True);args=p.parse_args()
 rospy.init_node('g4a_odom_fixture');pub=rospy.Publisher('/g4a/shadow/odom_input',Odometry,queue_size=10)
 goals=rospy.Publisher('/g4a/shadow/goal',PoseStamped,queue_size=1,latch=True)
 g=PoseStamped();g.header.frame_id='world';g.pose.position.x=args.origin[0]+6;g.pose.position.y=args.origin[1];g.pose.position.z=args.origin[2];g.pose.orientation.w=1;g.header.stamp=rospy.Time.now();goals.publish(g)
 start=time.monotonic();rate=rospy.Rate(30)
 while not rospy.is_shutdown():
  t=time.monotonic()-start;m=Odometry();m.header.stamp=rospy.Time.now();m.header.frame_id='world';m.child_frame_id='g4a_body'
  m.pose.pose.position.x=args.origin[0]+.05*math.sin(.3*t);m.pose.pose.position.y=args.origin[1]+.04*math.sin(.2*t);m.pose.pose.position.z=args.origin[2]
  q=Rotation.from_euler('xyz',[.08*math.sin(.4*t),.06*math.sin(.3*t),.1*math.sin(.2*t)]).as_quat()
  m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w=map(float,q)
  # Explicit native simulator convention: linear twist is WORLD velocity despite child_frame_id.
  m.twist.twist.linear.x=.015*math.cos(.3*t);m.twist.twist.linear.y=.008*math.cos(.2*t)
  pub.publish(m);rate.sleep()
if __name__=='__main__':main()
