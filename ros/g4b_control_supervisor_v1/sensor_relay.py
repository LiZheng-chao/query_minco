"""Native physics/sensor relay with explicit delayed/drop fault injection."""
import argparse,time,copy,rospy,json
from pathlib import Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,CameraInfo,Imu
def main():
 p=argparse.ArgumentParser();p.add_argument('--fault',default='none');p.add_argument('--fault-at',type=float,default=float('inf'));p.add_argument('--fault-at-file');a=p.parse_args();rospy.init_node('native_sensor_relay')
 odom=rospy.Publisher('/g4b/odom',Odometry,queue_size=1);depth=rospy.Publisher('/g4b/depth',Image,queue_size=1);cam=rospy.Publisher('/g4b/camera_info',CameraInfo,queue_size=1,latch=True);imu=rospy.Publisher('/g4b/imu',Imu,queue_size=1)
 def cb(m):
  if a.fault_at_file and Path(a.fault_at_file).exists():a.fault_at=json.loads(Path(a.fault_at_file).read_text())['monotonic']
  if a.fault=='odom_delay' and time.monotonic()>=a.fault_at:m=copy.deepcopy(m);m.header.stamp=rospy.Time.from_sec(m.header.stamp.to_sec()-.4)
  odom.publish(m)
 def image(m):
  if a.fault_at_file and Path(a.fault_at_file).exists():a.fault_at=json.loads(Path(a.fault_at_file).read_text())['monotonic']
  if not(a.fault=='depth_loss' and time.monotonic()>=a.fault_at):depth.publish(m)
 subscribers=[rospy.Subscriber('/g4b/native/odom',Odometry,cb,queue_size=1),rospy.Subscriber('/g4b/native/depth',Image,image,queue_size=1),
  rospy.Subscriber('/g4b/native/camera_info',CameraInfo,cam.publish,queue_size=1),rospy.Subscriber('/g4b/native/imu',Imu,imu.publish,queue_size=1)]
 rospy.spin()
if __name__=='__main__':main()
