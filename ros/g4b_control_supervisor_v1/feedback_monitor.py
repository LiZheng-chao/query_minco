"""Independent actual physics/control observer for researcher-manual runs."""
from pathlib import Path
import argparse,json,time,threading
import rospy,numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import SO3Command,PositionCommand
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir();rospy.init_node('independent_feedback_monitor')
 records={};lock=threading.Lock()
 def cb(topic,m):
  row=dict(monotonic=time.monotonic(),stamp=m.header.stamp.to_sec())
  if topic=='odom':row['p']=[m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z];row['v']=[m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]
  if topic=='imu':row['a']=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z]
  if topic=='request':row['pva']=[[m.position.x,m.position.y,m.position.z],[m.velocity.x,m.velocity.y,m.velocity.z],[m.acceleration.x,m.acceleration.y,m.acceleration.z]]
  with lock:records.setdefault(topic,[]).append(row)
 subs=[rospy.Subscriber('/g4b/native/odom',Odometry,lambda m:cb('odom',m),queue_size=1000),
  rospy.Subscriber('/g4b/native/imu',Imu,lambda m:cb('imu',m),queue_size=1000),
  rospy.Subscriber('/g4b/so3_raw',SO3Command,lambda m:cb('raw_controller',m),queue_size=1000),
  rospy.Subscriber('/g4b/so3_cmd',SO3Command,lambda m:cb('actuator',m),queue_size=1000),
  rospy.Subscriber('/g4b/request',PositionCommand,lambda m:cb('request',m),queue_size=1000)]
 try:rospy.spin()
 finally:
  rates={k:dict(n=len(v),hz=(len(v)-1)/(v[-1]['monotonic']-v[0]['monotonic']) if len(v)>1 else None) for k,v in records.items()}
  with (out/'RESULT.json').open('x') as f:json.dump(dict(records=records,rates=rates,independent_observer=True,no_online_path_selection=True),f,indent=2)
if __name__=='__main__':main()
