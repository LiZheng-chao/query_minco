"""Independent native-state collision observer;no output reaches the planner."""
from pathlib import Path
import sys,argparse,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--map-folder');a=p.parse_args()
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 from query_minco.training.physics_only_v1_2_gate1.pose_preflight import certify_position,grid_boxes
 run=Path(a.run);out=run/'truth_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('p2_independent_truth_monitor')
 meta=None;counts=None;boxes=None;last=None
 if a.map_folder:
  folder=Path(a.map_folder);meta=json.loads((folder/'COMPACT_GRID.json').read_text());counts=np.fromfile(folder/'COMPACT_COUNTS.i32','<i4').reshape(meta['grid_shape']);boxes,_=grid_boxes(meta,counts)
 def cb(m):
  nonlocal last
  with journal.callback() as okay:
   if not okay or (run/'STOP_CONFIRMED.json').exists():return
   p=np.array([m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z]);v=np.array([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]);s=m.header.stamp.to_sec()
   if meta:
    cert=certify_position(p,meta,counts,boxes);clear=cert['clearance_lower_bound_m'] if cert.get('passed') else -1.
   else:clear=float(min(p[2]-.10001,5-abs(p[0]),5-abs(p[1]),5-p[2])) # native floor top=.1+reserve;finite declared open fixture
   point_clear=clear
   if last:
    dt=s-last[0]
    if dt<=0 or dt>.2:clear=-1.
    else:clear=min(clear,last[1])-.5*dt*(max(np.linalg.norm(v),last[2])+8*dt)
   status='PASS' if clear>.15 else 'FAIL'
   row=dict(monotonic=time.monotonic(),stamp=s,status=status,clearance_lower_bound_m=float(clear),position=p.tolist(),velocity=v.tolist(),interpretation='certified interpolant/tube under declared v/a feedback bounds;FAIL may mean collision or UNKNOWN')
   journal.append(dict(kind='truth',**row));atomic_json(out/'HEARTBEAT.json',row);last=(s,point_clear,float(np.linalg.norm(v)))
 sub=rospy.Subscriber('/p2/native/odom',Odometry,cb,queue_size=10);rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():rate.sleep()
 except rospy.ROSInterruptException:pass
 finally:journal.close([sub.unregister]);atomic_json(out/'RESULT.json',dict(status='PASS' if (run/'STOP_CONFIRMED.json').exists() else 'FAIL'))
if __name__=='__main__':main()
