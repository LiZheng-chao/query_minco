"""Independent complete native-grid monitor;map truth never enters route selection."""
from pathlib import Path
import sys,time,argparse,threading,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from query_minco.integration.native_grid_truth_v1.grid import Grid
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 run=Path(a.run);out=run/'truth_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('fixed_native_truth')
 shared={};lock=threading.Lock();stop=threading.Event();previous=None;count=0;last=None
 def load():
  while not stop.is_set():
   try:
    g=Grid(run/'native_grid')
    with lock:shared['grid']=g
    return
   except (FileNotFoundError,json.JSONDecodeError):stop.wait(.1)
   except Exception as e:
    with lock:shared['error']=repr(e)
    return
 worker=threading.Thread(target=load,daemon=True);worker.start()
 def odom(m):
  nonlocal previous,count,last
  with journal.callback() as active:
   if not active:return
   with lock:g=shared.get('grid');error=shared.get('error')
   if g is None and error is None:return
   s=m.header.stamp.to_sec();p=m.pose.pose.position;v=m.twist.twist.linear;pos=np.array([p.x,p.y,p.z]);speed=float(np.linalg.norm([v.x,v.y,v.z]));clear=-1.
   try:
    if error:raise ValueError(error)
    if m.header.frame_id!='world' or not 0<=rospy.Time.now().to_sec()-s<=.2:raise ValueError('ODOM_SOURCE')
    clear=g.clearance(pos)
   except Exception as e:error=repr(e)
   point_clear=clear
   if previous:
    dt=s-previous[0];clear=min(clear,previous[1])-.5*dt*(max(speed,previous[2])+8*dt) if 0<dt<=.2 else -1.
   previous=(s,point_clear,speed)
   f=run/'FAULT.json';injected=f.exists() and json.loads(f.read_text()).get('case')=='truth_failure'
   status='PASS' if not error and clear>.15 and speed<=1+1e-6 and not injected else 'FAIL'
   last=dict(status=status,stamp=s,monotonic=time.monotonic(),position=pos.tolist(),clearance_lower_bound_m=clear,map_id=g.map_id if g else None,map_qualified=g is not None and not error,birth_clearance_pass=clear>=.60,truth_monitor_active=True,error=error)
   count+=1;journal.append(dict(kind='truth',**last));atomic_json(out/'HEARTBEAT.json',last)
 subs=[rospy.Subscriber('/p2/native/odom',Odometry,odom,queue_size=10)]
 try:
  while not rospy.is_shutdown() and not (run/'STOP_CONFIRMED.json').exists():rospy.sleep(.01)
 finally:
  errors=journal.close([s.unregister for s in subs]);stop.set();worker.join(2)
  with lock:qualified='grid' in shared and 'error' not in shared
  clean=not errors and not worker.is_alive() and (run/'STOP_CONFIRMED.json').exists()
  active=qualified and count>=5
  atomic_json(out/'RESULT.json',dict(status='PASS' if clean and active else 'FAIL',PROCESS_CLEANUP_PASS=clean,MAP_QUALIFIED=qualified,BIRTH_CLEARANCE_PASS=bool(last and last['birth_clearance_pass']),TRUTH_MONITOR_ACTIVE=active,PHYSICAL_SAFETY_PASS='NOT_EVALUATED_BY_PROCESS_CLEANUP',samples=count,errors=errors))
if __name__=='__main__':main()
