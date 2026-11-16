"""ROS native input integration,diagnostics only:no command or ARM publishers."""
from pathlib import Path
import sys,time,argparse,collections,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo
 from std_msgs.msg import String
 from query_minco.integration.observed_scheduler_v1.pipeline import Pipeline
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 rospy.init_node('p4_certification_observer');pipeline=Pipeline(clock=lambda:rospy.Time.now().to_sec());poses=collections.OrderedDict();calibrated=False
 journal=Journal(out/'EVENTS.jsonl')
 def odom(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='world':pipeline.fail('ODOM_FRAME');return
   q=m.pose.pose;s=m.header.stamp.to_sec()
   poses[s]=dict(frame='world',p=[q.position.x,q.position.y,q.position.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w])
   while len(poses)>512:poses.popitem(last=False)
 def camera(m):
  nonlocal calibrated
  with journal.callback() as active:
   if not active:return
   calibrated=m.header.frame_id=='camera_link' and list(m.K)==[80.,0.,80.,0.,80.,45.,0.,0.,1.]
   if not calibrated:pipeline.fail('CAMERA_CONTRACT')
 def depth(m):
  with journal.callback() as active:
   if not active:return
   s=m.header.stamp.to_sec()
   if not calibrated or s not in poses:return
   if m.header.frame_id!='camera_link' or m.encoding!='32FC1' or m.is_bigendian or (m.height,m.width)!=(90,160) or m.step<640:pipeline.fail('DEPTH_FORMAT');return
   try:
    d=np.ndarray((90,160),dtype='<f4',buffer=m.data,strides=(m.step,4))
    pipeline.ingest(d,poses[s],s,s)
   except Exception as e:pipeline.fail(repr(e))
 def request(m):
  with journal.callback() as active:
   if not active:return
   if len(m.data)>32768:pipeline.fail('REQUEST_BOUND');return
   try:
    d=json.loads(m.data)
    if d.get('kind')=='trajectory':pipeline.request(d['plan'])
   except Exception as e:pipeline.fail(repr(e))
 subs=[rospy.Subscriber('/p2/odom',Odometry,odom,queue_size=10),rospy.Subscriber('/p2/depth',Image,depth,queue_size=2),rospy.Subscriber('/p2/camera_info',CameraInfo,camera,queue_size=1),rospy.Subscriber('/p2/request',String,request,queue_size=1)]
 errors=[];rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown():
   start=time.perf_counter();status=pipeline.status()
   journal.append(dict(monotonic=time.monotonic(),status=status,consumer_s=time.perf_counter()-start))
   rate.sleep()
 except rospy.ROSInterruptException:pass
 except Exception as e:errors.append(repr(e))
 finally:
  errors+=journal.close([s.unregister for s in subs])
  try:pipeline.close()
  except Exception as e:errors.append(repr(e))
  atomic_json(out/'RESULT.json',dict(errors=errors,status='STATIC_STREAM_ONLY' if not errors else 'FAIL',motion_authority=False,physical_qualification='NOT_RUN'))
if __name__=='__main__':main()
