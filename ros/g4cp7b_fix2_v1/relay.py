"""Same fault semantics;per-layer input/output traces and bounded ROS queues."""
from pathlib import Path
import sys,argparse,time,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 import rospy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,Imu,CameraInfo
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal
 run=Path(a.run);out=run/'relay_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('fix2_native_relay')
 subs=[]
 for key,cls in [('odom',Odometry),('imu',Imu),('depth',Image),('camera_info',CameraInfo)]:
  pub=rospy.Publisher('/p2/'+key,cls,queue_size=5,latch=key=='camera_info',tcp_nodelay=True)
  def cb(m,key=key,pub=pub):
   arrived=time.monotonic();receipt_ros=rospy.Time.now().to_sec();stamp=m.header.stamp.to_sec()
   with journal.callback() as active:
    if not active:return
    p=run/'FAULT.json';f=json.loads(p.read_text()).get('case') if p.exists() else None
    filtered=f=='depth_loss' and key=='depth'
    if f=='odom_rollback' and key=='odom':m=copy.deepcopy(m);m.header.stamp-=rospy.Duration(1)
    before=time.monotonic()
    if not filtered:pub.publish(m)
    journal.append(dict(kind=key,source_stamp=stamp,output_stamp=m.header.stamp.to_sec(),receipt_monotonic=arrived,receipt_ros=receipt_ros,lock_wait_s=before-arrived,publish_end_monotonic=time.monotonic(),processing_s=time.monotonic()-arrived,published=not filtered,reason='INJECTED_DEPTH_LOSS' if filtered else 'FORWARDED',publisher_connections=pub.get_num_connections(),configured_queue=5,actual_queue_occupancy='NOT_EXPOSED_BY_ROSPY'))
  subs.append(rospy.Subscriber('/p2/native/'+key,cls,cb,queue_size=5,buff_size=1048576,tcp_nodelay=True))
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():rospy.sleep(.01)
 except rospy.ROSInterruptException:pass
 finally:journal.close([s.unregister for s in subs])
if __name__=='__main__':main()
