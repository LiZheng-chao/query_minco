"""Versioned real-native message relay with explicit post-ARM fault schedule."""
from pathlib import Path
import sys,argparse,json,time,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 import rospy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Imu,Image,CameraInfo
 rospy.init_node('p2_native_relay');run=Path(a.run);subs=[]
 def fault():
  p=run/'FAULT.json'
  return json.loads(p.read_text()).get('case') if p.exists() else None
 for key,cls in [('odom',Odometry),('imu',Imu),('depth',Image),('camera_info',CameraInfo)]:
  pub=rospy.Publisher('/p2/'+key,cls,queue_size=5,latch=(key=='camera_info'))
  def cb(m,key=key,pub=pub):
   f=fault()
   if f=='depth_loss' and key=='depth':return
   if f=='odom_rollback' and key=='odom':m=copy.deepcopy(m);m.header.stamp-=rospy.Duration(1.)
   pub.publish(m)
  subs.append(rospy.Subscriber('/p2/native/'+key,cls,cb,queue_size=5))
 try:rospy.spin()
 except rospy.ROSInterruptException:pass
 finally:
  for s in subs:s.unregister()
if __name__=='__main__':main()
