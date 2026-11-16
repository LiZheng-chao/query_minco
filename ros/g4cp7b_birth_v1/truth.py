"""Evaluation-only native forest observer;heavy map work separated from odom."""
from pathlib import Path
import sys,argparse,time,threading,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 import rospy,numpy as np
 from scipy.spatial import cKDTree
 from sensor_msgs.msg import PointCloud2
 from nav_msgs.msg import Odometry
 from sensor_msgs import point_cloud2
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 run=Path(a.run);out=run/'truth_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('p7b_native_truth')
 lock=threading.Lock();wake=threading.Event();stop=threading.Event();pending=[None];snapshot=[None];error=[None];previous=None;last_map=[None]
 def cloud(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='world' or m.width*m.height>2000000:
    with lock:error[0]='MAP_FRAME_OR_SIZE'
    return
   ident=hashlib.sha256(m.data).hexdigest()
   with lock:
    if last_map[0] is not None and last_map[0]['id']==ident:return
    if last_map[0] is not None:error[0]='STATIC_MAP_CHANGED'
    pending[0]=m;snapshot[0]=None # no stale geometry during new map validation
   wake.set()
 def work():
  while not stop.is_set():
   wake.wait(.1);wake.clear()
   with lock:m=pending[0];pending[0]=None;old=last_map[0]
   if m is None:continue
   try:
    ident=hashlib.sha256(m.data).hexdigest()
    if old is not None and old['id']==ident:new=old
    else:
     points=np.asarray(list(point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)),float)
     if not len(points) or not np.isfinite(points).all():raise ValueError('EMPTY_INVALID_MAP')
     new=dict(tree=cKDTree(points),lo=points.min(0),hi=points.max(0),id=ident)
    with lock:snapshot[0]=new;last_map[0]=new
   except Exception as e:
    with lock:error[0]=repr(e)
 def odom(m):
  nonlocal previous
  with journal.callback() as active:
   if not active:return
   with lock:geom=snapshot[0];fault=error[0]
   if geom is None and fault is None:return # owner never ARMs without heartbeat
   s=m.header.stamp.to_sec();q=m.pose.pose.position;v=m.twist.twist.linear
   pos=np.array([q.x,q.y,q.z]);speed=float(np.linalg.norm([v.x,v.y,v.z]));point_clear=-1.
   if geom is not None and not fault and m.header.frame_id=='world' and 0<=rospy.Time.now().to_sec()-s<=.2 and np.isfinite(pos).all() and np.isfinite(speed):
    if ((pos>=geom['lo'])&(pos<=geom['hi'])).all():point_clear=float(geom['tree'].query(pos)[0]-.1*np.sqrt(3))
   clear=point_clear
   if previous:
    dt=s-previous[0];clear=min(clear,previous[1])-.5*dt*(max(speed,previous[2])+8*dt) if 0<dt<=.2 else -1.
   previous=(s,point_clear,speed)
   fault_file=run/'FAULT.json'
   injected=fault_file.exists() and json.loads(fault_file.read_text()).get('case')=='truth_failure'
   status='PASS' if clear>.15 and speed<=1+1e-6 and not injected else 'FAIL'
   row=dict(status=status,stamp=s,monotonic=time.monotonic(),position=pos.tolist(),clearance_lower_bound_m=clear,map_id=geom['id'] if geom else None,fault=fault)
   journal.append(dict(kind='truth',**row));atomic_json(out/'HEARTBEAT.json',row)
 worker=threading.Thread(target=work,daemon=True);worker.start()
 subs=[rospy.Subscriber('/mock_map',PointCloud2,cloud,queue_size=1),rospy.Subscriber('/p2/native/odom',Odometry,odom,queue_size=10)]
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():rospy.sleep(.01)
 finally:
  cleanup=journal.close([s.unregister for s in subs]);stop.set();wake.set();worker.join(2)
  atomic_json(out/'RESULT.json',dict(status='PASS' if not cleanup and not worker.is_alive() and (run/'STOP_CONFIRMED.json').exists() else 'FAIL',errors=cleanup))
if __name__=='__main__':main()
