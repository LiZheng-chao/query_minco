"""Real-feedback deterministic MINCO reference. No GT imports,no optimizer."""
from pathlib import Path
import sys,argparse,json,time,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',choices=['C','D','E1','E2'],required=True);a=p.parse_args()
 import rospy,numpy as np,torch
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Imu,Image
 from std_msgs.msg import String
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 from query_minco.integration.yopo_execution_v4.trajectory import build
 from query_minco.integration.observed_space_v1.map import ObservedMap
 from query_minco.integration.observed_space_v1.space import MapSpace
 torch.set_num_threads(1);run=Path(a.run);out=run/'reference_data';out.mkdir();journal=Journal(out/'EVENTS.jsonl');rospy.init_node('p2_deterministic_reference')
 cache={};poses=collections.OrderedDict();anchor=None;active=None;plan=None;target=None;done=False;count=0;replans=0;last_plan=0.;errors=[];motion_start=None;ledger=ObservedMap(scene_id='frozen_forest3101_native_camera');observed_space=MapSpace(ledger)
 def store(k,m):
  with journal.callback() as okay:
   if not okay:return
   cache[k]=m
   if k=='depth':
    data=np.ndarray((m.height,m.width),dtype='<f4',buffer=m.data,strides=(m.step,4)).copy();path=out/('depth_%d_%09d.npz'%(m.header.stamp.secs,m.header.stamp.nsecs))
    if not path.exists():np.savez_compressed(path,depth=data,mask=np.isfinite(data)&(data>=.04)&(data<20),stamp=np.array(m.header.stamp.to_sec()))
   if k=='odom':
    q=m.pose.pose;v=m.twist.twist.linear;s=m.header.stamp.to_sec()
    poses[s]=dict(p=[q.position.x,q.position.y,q.position.z],v=[v.x,v.y,v.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w])
    while len(poses)>512:poses.popitem(last=False)
 subs=[rospy.Subscriber('/p2/odom',Odometry,lambda m:store('odom',m),queue_size=10),rospy.Subscriber('/p2/imu',Imu,lambda m:store('imu',m),queue_size=10),rospy.Subscriber('/p2/depth',Image,lambda m:store('depth',m),queue_size=2)]
 pub=rospy.Publisher('/p2/request',String,queue_size=1);rate=rospy.Rate(100)
 try:
  while not rospy.is_shutdown():
   if (run/'STOP_CONFIRMED.json').exists():break
   with journal.lock:c=dict(cache);pose_cache=dict(poses)
   if 'odom' not in c or 'imu' not in c:rate.sleep();continue
   m=c['odom'];s=m.header.stamp.to_sec();state=pose_cache[s];now=rospy.Time.now().to_sec()
   if now-s>.2 or abs(c['imu'].header.stamp.to_sec()-s)>.02:rate.sleep();continue
   acceleration=np.array([c['imu'].linear_acceleration.x,c['imu'].linear_acceleration.y,c['imu'].linear_acceleration.z]);pos=np.array(state['p'])
   if a.stage in ['E1','E2'] and 'depth' in c:
    image=c['depth'];ds=image.header.stamp.to_sec()
    if ds in pose_cache and ds>ledger.last_stamp:
     data=np.ndarray((image.height,image.width),dtype='<f4',buffer=image.data,strides=(image.step,4)).copy()
     ledger.add(data,pose_cache[ds],ds,ds,now)
   if anchor is None:anchor=pos.copy();target=anchor+np.array([.4 if a.stage in ['D','E1'] else 3.,0,0])
   hb=run/'supervisor_data/HEARTBEAT.json';h=json.loads(hb.read_text()) if hb.exists() else {}
   if h.get('mode')=='ARMED' and active is None:active=time.monotonic()
   pva=np.stack([anchor,np.zeros(3),np.zeros(3)])
   if active and time.monotonic()-active>=1. and a.stage in ['D','E1','E2']:
    try:
     if plan is None or (a.stage=='E2' and now-last_plan>=.2):
      begin=time.perf_counter();goal=target.copy();space=None
      if a.stage in ['E1','E2']:
       image=c.get('depth')
       if image is None or image.header.stamp.to_sec() not in pose_cache or now-image.header.stamp.to_sec()>.2:raise ValueError('NO_MATCHED_DEPTH')
       d=np.ndarray((image.height,image.width),dtype='<f4',buffer=image.data,strides=(image.step,4)).copy()
       space=observed_space
       goal=space.search_goal(pos,target)
      candidate=build(pos,np.array(state['v']),acceleration,state['q'],goal,s)
      if space:space.certify(candidate)
      elapsed=time.perf_counter()-begin
      if elapsed>.15 or rospy.Time.now().to_sec()-s>.2:raise ValueError('PLANNING_DEADLINE')
      plan=candidate;last_plan=now;replans+=1
      if motion_start is None:motion_start=plan.epoch
      journal.append(dict(kind='plan',monotonic=time.monotonic(),epoch=plan.epoch,plan=plan.record,planning_s=elapsed,maxima=plan.maxima,solver_errors=plan.errors,plan_id=plan.id,replans=replans))
     pva=plan.at(now)
     if a.stage in ['D','E1'] and now-plan.epoch>2.5:pva=np.stack([target,np.zeros(3),np.zeros(3)]);done=True
     if a.stage=='E2' and replans>=20 and pos[0]-anchor[0]>=2.:done=True
    except Exception as e:
     errors.append(repr(e));journal.append(dict(kind='planning_rejected',monotonic=time.monotonic(),reason=repr(e)))
     pub.publish(String(data=json.dumps(dict(frame='world',stamp=now,pva=pva.tolist(),kind='hold',fatal=repr(e)))))
     break
   fault=json.loads((run/'FAULT.json').read_text()) if (run/'FAULT.json').exists() else {}
   if fault.get('case')=='request_timeout':rate.sleep();continue
   if fault.get('case')=='nan_control':pva[0,0]=np.nan
   count+=1;env=dict(frame='world',stamp=now,sequence=count,pva=pva.tolist(),kind='trajectory' if plan and not done else 'hold')
   if env['kind']=='trajectory':env['plan']=plan.record
   pub.publish(String(data=json.dumps(env,allow_nan=fault.get('case')=='nan_control')))
   atomic_json(out/'STATUS.json',dict(monotonic=time.monotonic(),done=done,anchor=anchor.tolist(),target=target.tolist(),motion_start_epoch=motion_start,replans=replans,errors=errors,plan_id=plan.id if plan else None))
   rate.sleep()
 except rospy.ROSInterruptException:pass
 except BaseException as e:errors.append(repr(e))
 finally:
  cleanup=journal.close([s.unregister for s in subs]);atomic_json(out/'RESULT.json',dict(status='PASS' if not errors and not cleanup and (run/'STOP_CONFIRMED.json').exists() else 'FAIL',errors=errors,cleanup_errors=cleanup,replans=replans))
 if errors or cleanup:raise SystemExit(2)
if __name__=='__main__':main()
