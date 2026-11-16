"""Manual-only single low-speed reference from real physics feedback. No network."""
from pathlib import Path
import sys,argparse,time,json,threading
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import rospy,numpy as np,torch
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,CameraInfo,Imu
from quadrotor_msgs.msg import PositionCommand
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation,sample_world
from query_minco.integration.yopo_shadow_v1.command_contract import position_command
from query_minco.trajectory.state_goal_reference_v1 import reference_raw
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import norm_max
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hold','track'],required=True);p.add_argument('--fault',default='none');p.add_argument('--fault-at',type=float,default=float('inf'));p.add_argument('--fault-at-file');p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir();torch.set_num_threads(1);rospy.init_node('single_state_goal_reference')
 cache={};lock=threading.Lock()
 def store(k,m):
  with lock:cache[k]=m
 subscribers=[rospy.Subscriber('/g4b/odom',Odometry,lambda m:store('odom',m),queue_size=1),
  rospy.Subscriber('/g4b/imu',Imu,lambda m:store('imu',m),queue_size=1),rospy.Subscriber('/g4b/depth',Image,lambda m:store('depth',m),queue_size=1),rospy.Subscriber('/g4b/camera_info',CameraInfo,lambda m:store('camera',m),queue_size=1)]
 pub=rospy.Publisher('/g4b/request',PositionCommand,queue_size=1);
 with (out/'READY.json').open('x') as f:json.dump(dict(ready=True),f)
 rate=rospy.Rate(100);anchor=None;start=None;coeff=None;records=[];errors=[];done=False
 try:
  while not rospy.is_shutdown():
   if a.fault_at_file and Path(a.fault_at_file).exists():a.fault_at=json.loads(Path(a.fault_at_file).read_text())['monotonic']
   with lock:c=dict(cache)
   if len(c)<4:rate.sleep();continue
   m=c['odom'];imu=c['imu'];pose=m.pose.pose;pos=np.array([pose.position.x,pose.position.y,pose.position.z]);v=np.array([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]);acc=np.array([imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z]);now=rospy.Time.now()
   if any(now.to_sec()-c[k].header.stamp.to_sec()>.2 for k in ['odom','imu','depth']):rate.sleep();continue
   if anchor is None:anchor=pos.copy();start=time.monotonic()
   pva=np.stack([anchor,np.zeros(3),np.zeros(3)])
   if a.mode=='track' and time.monotonic()-start>.8:
    if coeff is None:
     planning_begin=time.perf_counter()
     image=c['depth'];d=np.frombuffer(image.data,'<f4').reshape(image.height,image.width).copy();q=pose.orientation
     x,state,rotation=observation(d,[q.x,q.y,q.z,q.w],v,acc,pos,anchor+np.array([.4,0,0]),list(c['camera'].K))
     with torch.no_grad():
      parameters=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1)(reference_raw(torch.from_numpy(state),'state_goal_quintic'),torch.from_numpy(state))
      tr=MincoAdapter(num_candidates=1)(parameters.head_pva,parameters.tail_pva,parameters.inner_pts,parameters.durations)
     if not tr.solve_valid.all():raise ValueError('MINCO_SOLVE')
     coeff=tr.coeffs[0,0].numpy();dur=tr.durations[0,0].numpy();origin=pos.copy();epoch=m.header.stamp.to_sec()
     maxima=[max(norm_max(poly,dt,k) for poly,dt in zip(coeff,dur)) for k in [1,2,3]]
     if any(z>limit for z,limit in zip(maxima,[1,8,100])):raise ValueError('ORIGINAL_TRAJECTORY_EXCEEDS_EXECUTION_LIMIT_NO_SCALING')
     with (out/'PLAN_IDENTITY.json').open('x') as f:json.dump(dict(planning_and_minco_s=time.perf_counter()-planning_begin,polynomial_maxima=maxima,T=2.,initial_pose=origin.tolist(),epoch=epoch,source='real physics Odom+world Imu+Depth/CameraInfo,no geometry or network'),f,indent=2)
     np.savez_compressed(out/'REFERENCE.npz',coeff=coeff,durations=dur,origin=origin,rotation=rotation,epoch=np.array(epoch))
    elapsed=now.to_sec()-epoch;pva=sample_world(coeff,dur,origin,rotation,min(elapsed,float(dur.sum())))
   else:elapsed=None
   if a.fault=='timeout' and time.monotonic()>=a.fault_at:rate.sleep();continue
   if a.fault=='nan' and time.monotonic()>=a.fault_at:pva[0,0]=np.nan
   if not np.isfinite(pva).all() and a.fault!='nan':raise ValueError('NONFINITE_PLAN')
   # Fault NaN is intentionally injected after constructing an otherwise valid command.
   msg=position_command(PositionCommand,np.nan_to_num(pva),now,0.,1);msg.kx=[0.,0.,0.];msg.kv=[0.,0.,0.]
   if a.fault=='nan' and time.monotonic()>=a.fault_at:msg.position.x=float('nan')
   pub.publish(msg)
   err=np.stack([pos,v,acc])-pva
   records.append(dict(stamp=now.to_sec(),odom_stamp=m.header.stamp.to_sec(),depth_stamp=c['depth'].header.stamp.to_sec(),target=pva.tolist(),actual=np.stack([pos,v,acc]).tolist(),error_norm=np.linalg.norm(err,axis=1).tolist(),trajectory_t=elapsed))
   if a.mode=='track' and elapsed is not None:
    if np.linalg.norm(err[0])>.25 or np.linalg.norm(v)>1+1e-6:raise ValueError('TRACKING_SAFETY_ABORT')
    if elapsed>2.4:done=True;break
   rate.sleep()
 except rospy.ROSInterruptException:pass
 except Exception as e:errors.append(repr(e));raise
 finally:
  status='NOT_RUN' if a.mode=='hold' else 'FAIL'
  if done and records:
   rr=[r for r in records if r['trajectory_t'] is not None and r['trajectory_t']<=2.]
   terminal=records[-1];passed=max(r['error_norm'][0] for r in rr)<.15 and terminal['error_norm'][0]<.1 and np.linalg.norm(np.array(terminal['actual'])[1])<.1 and np.array(terminal['actual'])[0,0]-anchor[0]>.25
   status='PASS' if passed else 'FAIL'
  with (out/'RESULT.json').open('x') as f:json.dump(dict(status=status,mode=a.mode,records=records,errors=errors,actual_physics_feedback=True,no_network=True,complete_single_trajectory=done),f,indent=2)
if __name__=='__main__':main()
