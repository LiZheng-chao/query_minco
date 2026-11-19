"""Read-only log-to-RViz adapter. Publishes exclusively /forest10m/display/*.
Replay markers are historical evidence, never live control authority.
"""
import argparse,json,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
 import rospy
 from visualization_msgs.msg import Marker,MarkerArray
 from nav_msgs.msg import Path as RosPath
 from geometry_msgs.msg import PoseStamped
 from query_minco.integration.yopo_execution_v4.trajectory import Plan
 rospy.init_node('forest10m_readonly_display',anonymous=True)
 prefix='/forest10m/display/'
 pubs={k:rospy.Publisher(prefix+k,MarkerArray if k in ['candidate','approved'] else Marker,queue_size=1,latch=True) for k in ['candidate','approved','drone']}
 actual=rospy.Publisher(prefix+'actual',RosPath,queue_size=1,latch=True);goalpub=rospy.Publisher(prefix+'goal',PoseStamped,queue_size=1,latch=True)
 cfg=json.loads(Path(__file__).with_name('config.json').read_text());goal=PoseStamped();goal.header.frame_id='world';goal.pose.orientation.w=1;goal.pose.position.x,goal.pose.position.y,goal.pose.position.z=cfg['goal'];goalpub.publish(goal)
 file=a.run/'supervisor_data/EVENTS.jsonl'
 if not file.exists():file=a.run/'none/supervisor_data/EVENTS.jsonl'
 if not file.exists():raise ValueError('RUN_EVENTS_NOT_FOUND')
 path=RosPath();path.header.frame_id='world';plans={};pending={};last_time=None;deadline=None
 def marker(plan,color,ns):
  from geometry_msgs.msg import Point
  m=Marker();m.header.frame_id='world';m.ns=ns;m.id=0;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.pose.orientation.w=1;m.scale.x=.03;m.color.r,m.color.g,m.color.b=color;m.color.a=1;m.text=plan.id
  m.points=[Point(*plan.sample(i*.025)[0].tolist()) for i in range(81)];return m
 with file.open() as f:
  while not rospy.is_shutdown():
   line=f.readline()
   if not line:
    if a.replay:break
    if deadline is not None and rospy.Time.now().to_sec()>deadline:
     m=Marker();m.action=Marker.DELETEALL;pubs['approved'].publish(MarkerArray(markers=[m]));deadline=None
    time.sleep(.02);continue
   e=json.loads(line);stamp=e.get('ros_now',e.get('stamp'));q=e.get('payload',{})
   if a.replay and isinstance(stamp,(int,float)):
    if last_time is not None:time.sleep(min(max(stamp-last_time,0),.2))
    last_time=stamp
   if e.get('kind')=='odom' and e.get('accepted'):
    pose=PoseStamped();pose.header.frame_id='world';pose.header.stamp=rospy.Time.from_sec(e['stamp']);pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=q['p'];pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w=q['q'];path.poses.append(pose);path.header.stamp=pose.header.stamp;actual.publish(path)
    m=Marker();m.header=pose.header;m.pose=pose.pose;m.type=Marker.SPHERE;m.scale.x=.3;m.scale.y=.3;m.scale.z=.15;m.color.r=1;m.color.g=.55;m.color.a=1;pubs['drone'].publish(m)
   if q.get('event')=='CANDIDATE_GENERATED' and 'plan' in q:
    plan=Plan(q['plan'])
    if plan.id!=q['plan_id']:raise ValueError('PLAN_ID_MISMATCH')
    plans[plan.id]=plan;pubs['candidate'].publish(MarkerArray(markers=[marker(plan,(0,0,1),'candidate')]))
   if q.get('event')=='ORACLE_PREFIX_COMMITTED':
    proof=q['oracle'];plan=plans.get(q['plan_id']);now=last_time if a.replay else rospy.Time.now().to_sec()
    if plan is not None and now is not None and proof['issued']<=now<=proof['expires']:
     m=marker(plan,(0,1,0),'approved');m.lifetime=rospy.Duration(max(.001,proof['expires']-now));pubs['approved'].publish(MarkerArray(markers=[m]));deadline=proof['expires']
 if a.replay:
  # Keep only historical candidate/odom latched; approved expires on original lease.
  rospy.spin()
if __name__=='__main__':main()
