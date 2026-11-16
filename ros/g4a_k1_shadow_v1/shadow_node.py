"""ROS shadow planner. Publishes nav_msgs/Path only, never PositionCommand."""
from pathlib import Path
import sys,json,argparse,time,threading
from collections import deque
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch,rospy
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry,Path as RosPath
from geometry_msgs.msg import PoseStamped
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation,sample_world
from query_minco.audit.k1_signal_diagnostics_v1 import model_at
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.trajectory.state_goal_reference_v1 import reference_raw
from query_minco.evaluation.mirror_holdout_v1 import write
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30);args=p.parse_args()
 out=Path(args.output);out.mkdir();torch.set_num_threads(1);R=Path(__file__).resolve().parents[2]
 model,identity=model_at(R,128);before=state_sha(model);dec=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);minco=MincoAdapter(num_candidates=1)
 rospy.init_node('g4a_shadow_planner');lock=threading.Lock();odoms=deque(maxlen=200);images=deque(maxlen=1);infos=deque(maxlen=20);goals=deque(maxlen=1)
 def push(queue,msg):
  with lock:queue.append(msg)
 subs=[rospy.Subscriber('/g4a/shadow/odom_input',Odometry,lambda m:push(odoms,m),queue_size=100),
  rospy.Subscriber('/g4a/shadow/depth',Image,lambda m:push(images,m),queue_size=1),
  rospy.Subscriber('/g4a/shadow/camera_info',CameraInfo,lambda m:push(infos,m),queue_size=10),
  rospy.Subscriber('/g4a/shadow/goal',PoseStamped,lambda m:push(goals,m),queue_size=1)]
 methods=['fixed_forward','state_goal_quintic','K1_step128']
 pubs={m:rospy.Publisher('/g4a/shadow/path/'+m,RosPath,queue_size=1) for m in methods}
 records=[];exceptions=[];last=-1.;frames=0;start=time.monotonic();arrays={};rate=rospy.Rate(30)
 while frames<args.frames and not rospy.is_shutdown() and time.monotonic()-start<20:
  with lock:om=list(odoms);im=list(images);ci=list(infos);gl=list(goals)
  if len(om)<2 or not im or not ci or not gl:rate.sleep();continue
  image=im[-1];stamp=image.header.stamp.to_sec()
  if stamp==last:rate.sleep();continue
  matches=[(i,o) for i,o in enumerate(om) if abs(o.header.stamp.to_sec()-stamp)<1e-5]
  cameras=[c for c in ci if abs(c.header.stamp.to_sec()-stamp)<1e-5]
  if not matches or not cameras:rate.sleep();continue
  last=stamp;ix,odom=matches[-1];cam=cameras[-1]
  try:
   begin=time.perf_counter();age=(rospy.Time.now()-image.header.stamp).to_sec()
   if age<-.01 or age>.5:raise ValueError('STALE_DEPTH')
   if ix==0:raise ValueError('ACCELERATION_HISTORY_MISSING')
   prev=om[ix-1];dt=stamp-prev.header.stamp.to_sec()
   if not .001<=dt<=.2:raise ValueError('ODOMETRY_DT')
   if image.encoding!='32FC1' or image.height!=90 or image.width!=160 or image.step!=640 or image.is_bigendian:raise ValueError('IMAGE_MESSAGE_CONTRACT')
   if image.header.frame_id!='camera_link' or cam.header.frame_id!='camera_link' or odom.header.frame_id!='world' or gl[-1].header.frame_id!='world':raise ValueError('FRAME_ID')
   d=np.frombuffer(image.data,dtype='<f4').reshape(90,160).copy()
   v=np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z]);pv=np.array([prev.twist.twist.linear.x,prev.twist.twist.linear.y,prev.twist.twist.linear.z]);a=(v-pv)/dt
   pos=np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z]);goal=np.array([gl[-1].pose.position.x,gl[-1].pose.position.y,gl[-1].pose.position.z])
   q=odom.pose.pose.orientation;quat=[q.x,q.y,q.z,q.w]
   x,s,rot=observation(d,quat,v,a,pos,goal,list(cam.K));depth=torch.from_numpy(x);state=torch.from_numpy(s)
   frame=dict(frame=frames,stamp=stamp,depth_odom_delta_s=abs(odom.header.stamp.to_sec()-stamp),camera_delta_s=abs(cam.header.stamp.to_sec()-stamp),input_age_s=age,
    q_xyzw=quat,origin=pos.tolist(),velocity_world=v.tolist(),acceleration_estimate_world=a.tolist(),state12=s.tolist(),methods={})
   with torch.no_grad():
    for method in methods:
     tick=time.perf_counter();raw=model(depth,state) if method=='K1_step128' else reference_raw(state,method);forward=time.perf_counter()-tick
     tick=time.perf_counter();params=dec(raw,state);tr=minco(params.head_pva,params.tail_pva,params.inner_pts,params.durations);decode=time.perf_counter()-tick
     if not tr.solve_valid.all() or not torch.isfinite(tr.coeffs).all():raise ValueError('MINCO_INVALID')
     c=tr.coeffs[0,0].numpy();dur=tr.durations[0,0].numpy();head=sample_world(c,dur,pos,rot,0)
     np.testing.assert_allclose(head,np.stack([pos,v,a]),atol=2e-5,rtol=0)
     path=RosPath();path.header.stamp=image.header.stamp;path.header.frame_id='world'
     for t in np.linspace(0,2,41):
      pva=sample_world(c,dur,pos,rot,float(t));pose=PoseStamped();pose.header=path.header;pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,pva[0]);pose.pose.orientation.w=1;path.poses.append(pose)
     pubs[method].publish(path);key='frame%03d_%s'%(frames,method);arrays[key+'_coeff']=c;arrays[key+'_durations']=dur;arrays[key+'_raw21']=raw.numpy()
     frame['methods'][method]=dict(forward_s=forward,minco_s=decode,output_age_s=(rospy.Time.now()-image.header.stamp).to_sec(),raw21=raw.tolist(),initial_pva_world=head.tolist())
   frame['callback_pipeline_s']=time.perf_counter()-begin
   arrays['frame%03d_depth'%frames]=d;records.append(frame);frames+=1
  except Exception as e:exceptions.append(dict(stamp=stamp,error=repr(e)))
  rate.sleep()
 if state_sha(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('MODEL_MUTATION')
 np.savez_compressed(out/'SHADOW_ARRAYS.npz',**arrays)
 status='PASS' if frames>=args.frames else 'FAIL'
 write(out/'RESULT.json',dict(status=status,frames=frames,records=records,exceptions=exceptions,checkpoint=identity,network_updates=0,control_published=False,
  output_topics={m:'/g4a/shadow/path/'+m for m in methods},input_contract='native world linear velocity;finite-difference acceleration;full dynamic body rotation;no displayTF offset',
  elapsed_s=time.monotonic()-start,missing_or_skipped_images='latest-only queues;timestamp gaps in records permit observed inference-rate accounting'))
 print('SHADOW_RESULT',status,frames,len(exceptions),flush=True)
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
