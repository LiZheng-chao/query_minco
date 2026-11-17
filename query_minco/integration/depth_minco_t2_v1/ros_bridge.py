"""Used only inside the existing authoritative Supervisor; no SO3 publisher."""
from pathlib import Path
import json,time
import numpy as np
from .space import BirthGrant
from .engine import Engine

class SupervisorBridge:
 def __init__(self,run,clock):
  self.run=Path(run);self.clock=clock
  selected_path=Path(__file__).resolve().parents[3]/'research_outputs/t2_depth_minco_20261116_v1/BIRTH_SELECTION.json'
  self.selected=json.loads(selected_path.read_text())
  self.engine=Engine(lambda now:BirthGrant.load(self.run,self.selected,clock=clock),clock)
  self.engine.start()
 def state(self,last,stamp,imu_stamp):
  if 'odom' not in last or 'imu' not in last:raise ValueError('MISSING_PVA')
  if abs(imu_stamp-stamp)>.02:raise ValueError('IMU_ODOM_ALIGNMENT')
  state=dict(last['odom'],a=last['imu']['a_world'])
  self.engine.set_state(state,stamp)
 def observe(self,depth,pose,stamp):
  self.engine.observe(depth,pose,stamp,stamp)
 def request(self,envelope,last,now,armed,odom_stamp,imu_stamp):
  if envelope.get('kind')!='forest_goal' or envelope.get('frame')!='world':raise ValueError('FOREST_GOAL_CONTRACT')
  self.state(last,odom_stamp,imu_stamp);self.engine.set_goal(envelope['goal'],envelope['stamp'])
  if armed:return None
  # PREPARE/ACK hover is startup-only; PlantGate forbids integration before ARM.
  pva=np.stack([last['odom']['p'],np.zeros(3),np.zeros(3)])
  return pva
 def close(self):
  self.engine.close()
  (self.run/'T2_PLANNER_EVENTS.json').write_text(json.dumps(list(self.engine.events),indent=2))
  (self.run/'T2_PLANNER_COUNTS.json').write_text(json.dumps(dict(self.engine.counts),indent=2))

def reference_main(args):
 import rospy
 from nav_msgs.msg import Odometry
 from std_msgs.msg import String
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 run=Path(args.run);out=run/'reference_data';out.mkdir();rospy.init_node('t2_forest_goal_reference')
 journal=Journal(out/'EVENTS.jsonl');latest={};target=None;errors=[]
 def odom(m):
  with journal.callback() as active:
   if active:latest['odom']=m
 sub=rospy.Subscriber('/p2/odom',Odometry,odom,queue_size=10)
 pub=rospy.Publisher('/p2/request',String,queue_size=1);rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():
   with journal.lock:m=latest.get('odom')
   now=rospy.Time.now().to_sec()
   if m is None or not 0<=now-m.header.stamp.to_sec()<=.2:rate.sleep();continue
   if target is None:
    p=m.pose.pose.position;target=[p.x+3.,p.y,p.z]
   pub.publish(String(data=json.dumps(dict(kind='forest_goal',frame='world',stamp=now,goal=target))))
   hb=run/'supervisor_data/HEARTBEAT.json'
   h=json.loads(hb.read_text()) if hb.exists() else {}
   plans=(h.get('planner') or {}).get('published_plans',0)
   p=m.pose.pose.position;anchor=[target[0]-3.,target[1],target[2]]
   done=plans>=20 and p.x-anchor[0]>=2.
   atomic_json(out/'STATUS.json',dict(done=done,anchor=anchor,target=target,replans=plans,monotonic=time.monotonic()))

   rate.sleep()
 except rospy.ROSInterruptException:pass
 except Exception as e:errors.append(repr(e))
 finally:
  ce=journal.close([sub.unregister])
  atomic_json(out/'RESULT.json',dict(status='PASS' if not errors and not ce and (run/'STOP_CONFIRMED.json').exists() else 'FAIL',errors=errors,cleanup_errors=ce))
