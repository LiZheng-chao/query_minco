"""Native /mock_map evaluation-only collision observer;no geometry output to planner."""
from pathlib import Path
import sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--odom-topic',default='/p2/native/odom');a=p.parse_args()
 import rospy,numpy as np
 from scipy.spatial import cKDTree
 from sensor_msgs.msg import PointCloud2
 from nav_msgs.msg import Odometry
 from sensor_msgs import point_cloud2
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 run=Path(a.run);out=run/'truth_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('p7_independent_native_truth')
 cache={};previous=None
 def cloud(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='world':cache['error']='MAP_FRAME';return
   # Budget cap refuses oversize rather than silently subsampling away obstacles.
   if m.width*m.height>2000000:cache['error']='MAP_SIZE_BOUND';return
   points=np.asarray(list(point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)),float)
   if not len(points):cache['error']='EMPTY_MAP';return
   cache.update(tree=cKDTree(points),lo=points.min(0),hi=points.max(0),received=time.monotonic())
 def odom(m):
  nonlocal previous
  with journal.callback() as active:
   if not active:return
   s=m.header.stamp.to_sec();now=rospy.Time.now().to_sec();q=m.pose.pose.position;v=m.twist.twist.linear
   pos=np.array([q.x,q.y,q.z]);speed=float(np.linalg.norm([v.x,v.y,v.z]));clear=-1.;reason=cache.get('error')
   if not reason and 'tree' in cache and 0<=now-s<=.2:
    # Bound possible occupied voxel extent from point <= voxel diagonal0.1*sqrt3.
    if ((pos>=cache['lo'])&(pos<=cache['hi'])).all():
     distance,_=cache['tree'].query(pos);clear=float(distance-.1*np.sqrt(3))
     point_clear=clear
     if previous:
      dt=s-previous[0]
      clear=min(clear,previous[1])-.5*dt*(max(speed,previous[2])+8*dt) if 0<dt<=.2 else -1
   status='PASS' if clear>.15 and speed<=1+1e-6 else 'FAIL'
   row=dict(status=status,stamp=s,monotonic=time.monotonic(),position=pos.tolist(),clearance_lower_bound_m=clear,reason=reason,scope='independent evaluation only;not online routing')
   journal.append(dict(kind='truth',**row));atomic_json(out/'HEARTBEAT.json',row);previous=(s,point_clear if 'point_clear' in locals() else -1.,speed)
 subs=[rospy.Subscriber('/mock_map',PointCloud2,cloud,queue_size=1),rospy.Subscriber(a.odom_topic,Odometry,odom,queue_size=10)]
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():rospy.sleep(.01)
 finally:
  errors=journal.close([s.unregister for s in subs]);atomic_json(out/'RESULT.json',dict(status='PASS' if not errors and (run/'STOP_CONFIRMED.json').exists() else 'FAIL',errors=errors))
if __name__=='__main__':main()
