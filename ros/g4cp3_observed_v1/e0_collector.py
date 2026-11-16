"""Live native forest sensor collector. It publishes no control or ARM messages."""
from pathlib import Path
import sys,argparse,time,json,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene-id',required=True);p.add_argument('--duration',type=float,default=10.);a=p.parse_args()
 if not 1<=a.duration<=15:raise ValueError('DURATION_CAP')
 import rospy,numpy as np
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo
 from std_msgs.msg import String
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 from query_minco.integration.observed_space_v1.map import ObservedMap,Config,UNKNOWN,KNOWN_FREE,OCCUPIED
 from query_minco.integration.observed_space_v1.space import bootstrap
 from query_minco.evaluation.mirror_holdout_v1 import sha
 run=Path(a.run);out=run/'sensor_data';out.mkdir();rospy.init_node('p3_e0_passive_collector')
 journal=Journal(out/'EVENTS.jsonl');poses=collections.OrderedDict();pending=collections.OrderedDict();camera=None;phase=None;errors=[];records=[];snap=None;boot=None;counts=0
 ledger=ObservedMap(scene_id=a.scene_id);start=time.monotonic();first_frame=None
 def odom(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='world':errors.append('ODOM_FRAME');return
   q=m.pose.pose;v=m.twist.twist.linear;s=m.header.stamp.to_sec()
   poses[s]=dict(frame='world',p=[q.position.x,q.position.y,q.position.z],q=[q.orientation.x,q.orientation.y,q.orientation.z,q.orientation.w],v=[v.x,v.y,v.z],source_stamp=s)
   while len(poses)>512:poses.popitem(last=False)
 def depth(m):
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='camera_link' or m.encoding!='32FC1' or m.is_bigendian or (m.width,m.height)!=(160,90):errors.append('DEPTH_CONTRACT');return
   pending[m.header.stamp.to_sec()]=m
   while len(pending)>4:
    old,_=pending.popitem(last=False);journal.append(dict(kind='excluded',reason='INPUT_QUEUE_BOUND',stamp=old))
 def cam(m):
  nonlocal camera
  with journal.callback() as active:
   if not active:return
   if m.header.frame_id!='camera_link' or list(m.K)!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:errors.append('CALIBRATION')
   else:camera=m
 def plant(m):
  nonlocal phase
  with journal.callback() as active:
   if not active:return
   phase=json.loads(m.data)
   if phase['phase']!='HELD' or phase['integration_steps']!=0:errors.append('UNAUTHORIZED_INTEGRATION')
 subs=[rospy.Subscriber('/p3/native/odom',Odometry,odom,queue_size=20),rospy.Subscriber('/p3/native/depth',Image,depth,queue_size=2),rospy.Subscriber('/p3/native/camera_info',CameraInfo,cam,queue_size=1),rospy.Subscriber('/p3/native/startup_state',String,plant,queue_size=10)]
 rate=rospy.Rate(10);timings=[]
 try:
  while not rospy.is_shutdown() and time.monotonic()-start<a.duration+8 and counts<200:
   if errors:break
   with journal.lock:
    matched=[s for s in pending if s in poses]
    if not matched or camera is None or phase is None:message=None
    else:
     stamp=max(matched);message=pending.pop(stamp);pose=poses[stamp].copy()
     # Older queued frames are explicitly excluded,never restamped.
     for old in list(pending):
      if old<stamp:pending.pop(old);journal.append(dict(kind='excluded',reason='SUPERSEDED_SOURCE_FRAME',stamp=old))
   if message is None:rate.sleep();continue
   now=rospy.Time.now().to_sec()
   if now-stamp>.2:journal.append(dict(kind='excluded',reason='STALE_SOURCE',stamp=stamp,now=now));rate.sleep();continue
   if first_frame is None:first_frame=time.monotonic()
   data=np.ndarray((90,160),dtype='<f4',buffer=message.data,strides=(message.step,4)).copy()
   begin=time.perf_counter();ledger.add(data,pose,stamp,pose['source_stamp'],now);snap=ledger.snapshot(now);elapsed=time.perf_counter()-begin;timings.append(elapsed)
   boot=bootstrap(snap,pose['p'],now);counts+=1
   path=out/('frame_%04d.npz'%counts);np.savez_compressed(path,depth=data,pose=np.asarray(pose['p']),quaternion=np.asarray(pose['q']),stamp=np.array(stamp))
   record=dict(frame=counts,stamp=stamp,pose_stamp=pose['source_stamp'],pose=pose,source=ledger.frames[-1].record,raw_file=str(path),raw_sha256=sha(path),snapshot=snap.id,valid_until=snap.valid_until,
    state_counts={str(k):int((snap.states==k).sum()) for k in [UNKNOWN,KNOWN_FREE,OCCUPIED]},bootstrap=boot,update_s=elapsed,plant_source=phase['state_source'])
   records.append(record);journal.append(dict(kind='observed_frame',monotonic=time.monotonic(),payload=record));atomic_json(out/'HEARTBEAT.json',record)
   if time.monotonic()-first_frame>=a.duration:break
   rate.sleep()
 except rospy.ROSInterruptException:errors.append('INTERRUPTED')
 except BaseException as e:errors.append(repr(e));journal.append(dict(kind='contract_failure',error=repr(e),monotonic=time.monotonic()))
 finally:
  cleanup=journal.close([s.unregister for s in subs])
  if snap is not None:np.savez_compressed(out/'FINAL_MAP.npz',states=snap.states,origin=snap.origin,resolution=np.array(snap.cfg.resolution))
  module_sha256={str(Path(__file__).resolve().parents[2]/('query_minco/integration/observed_space_v1/'+name)):sha(Path(__file__).resolve().parents[2]/('query_minco/integration/observed_space_v1/'+name)) for name in ['map.py','space.py']}
  result=dict(module_sha256=module_sha256,status='STATIC_PASS' if not errors and not cleanup and counts>=10 else 'FAIL',live_native_sensor=True,native_integration='NOT_RUN',max_integration_steps=phase['integration_steps'] if phase else None,
   source='native HELD pose and native CUDA-rendered depth;not integrated physical feedback',bootstrap_status=boot['status'] if boot else 'FAIL',bootstrap=boot,frames=counts,records=records,errors=errors,cleanup_errors=cleanup,
   timings_s=dict(max=max(timings or [0.]),mean=float(np.mean(timings)) if timings else None),new_network_updates=0)
  atomic_json(run/'E0_RESULT.json',result)
 if errors or cleanup or counts<10:raise SystemExit(2)
 # Successful sensor collection still does not authorize motion if its initial tube is unknown.
 if not boot or boot['status']!='STATIC_PASS':raise SystemExit(3)
if __name__=='__main__':main()
