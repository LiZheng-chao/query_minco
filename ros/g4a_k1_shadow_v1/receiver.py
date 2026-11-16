"""Independent subscriber proving real ROS sensor and non-control output delivery."""
from pathlib import Path
import argparse,json,time,threading
import rospy
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Path as RosPath,Odometry
from geometry_msgs.msg import PoseStamped
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir()
 rospy.init_node('g4a_shadow_receiver');rows={};subs=[];lock=threading.Lock()
 topics={'/g4a/shadow/depth':Image,'/g4a/shadow/camera_info':CameraInfo,'/g4a/shadow/odom_input':Odometry,'/g4a/shadow/goal':PoseStamped}
 for method in ['fixed_forward','state_goal_quintic','K1_step128']:topics['/g4a/shadow/path/'+method]=RosPath
 def callback(topic,msg):
  with lock:
   row=dict(stamp=msg.header.stamp.to_sec(),received_wall=time.time(),frame=msg.header.frame_id)
   if hasattr(msg,'poses'):row['pose_count']=len(msg.poses)
   rows.setdefault(topic,[]).append(row)
 for topic,cls in topics.items():subs.append(rospy.Subscriber(topic,cls,lambda m,t=topic:callback(t,m),queue_size=100))
 start=time.monotonic()
 while time.monotonic()-start<10 and not rospy.is_shutdown():time.sleep(.05)
 graph=rospy.get_master().getSystemState();types=rospy.get_master().getTopicTypes()
 result=dict(status='PASS' if all(len(rows.get(t,[]))>=10 for t in topics if 'path/' in t) else 'FAIL',records=rows,
  message_identity={t:dict(type=c._type,md5=c._md5sum) for t,c in topics.items()},system_state=graph,topic_types=types,
  control_topic_subscribed_or_published=False)
 with (out/'RESULT.json').open('x') as f:json.dump(result,f,indent=2)
 print('RECEIVER',result['status'],{k:len(v) for k,v in rows.items()})
 if result['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
