"""Frozen C1 proposals only. Contains NO ROS Publisher or control message import."""
import argparse,json,time,threading,collections,pickle,traceback
from pathlib import Path
import numpy as np,torch
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.shadow import FROZEN,sha,decode_image,camera_check,make_state,rotation
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.trajectory_execution import Program,Measurement
from query_minco.inference.constrained_prefix import constrained_prefix
from query_minco.inference.committed_reference import prepare_reference
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope

def main():
 import rospy,message_filters,rosgraph
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo
 from std_msgs.msg import Time as TimeMessage
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--references',required=True);a=p.parse_args()
 run=Path(a.run);run.mkdir();root=run.parents[1]
 binding=next(b for b in json.loads(FROZEN.read_text()) if b['label']=='C1')
 adapter=InferenceAdapter(binding['checkpoint']['path'],binding['method_id'],binding['checkpoint']['sha256'])
 adapter.model.requires_grad_(False)
 with Path(a.references).open('rb') as f:bundles=pickle.load(f)
 nominal=bundles[1].prefix;epoch=[None];latest=[None];prev=[None];frames=collections.deque(maxlen=1)
 stats=[];rejects=collections.Counter();attempt=0;previous_frame=None;start=time.monotonic()
 pending=collections.deque();gated=[];stop=threading.Event()
 rospy.init_node('query_c1_proposal_shadow',disable_signals=True)
 master=rosgraph.Master('/query_c1_proposal_shadow')
 def odom(m):
  t=m.header.stamp.to_sec();v=m.twist.twist.linear;vv=np.array([v.x,v.y,v.z]);pp=m.pose.pose.position
  past=prev[0]
  if past is None:prev[0]=(t,vv);return
  dt=t-past[0]
  if dt<.005:return
  prev[0]=(t,vv)
  if dt>.05:return
  latest[0]=Measurement(t,np.stack([[pp.x,pp.y,pp.z],vv,(vv-past[1])/dt]))
 def synced(d,c,o):frames.append((d,c,o,time.monotonic()))
 def epoch_cb(m):epoch[0]=m.data.to_sec()
 def ref_at(t):return nominal.pva(max(0,min(t-epoch[0],nominal.duration)))
 class Known:
  def pva(self,t):return ref_at(t)
 known=Known()
 def gate_worker():
  while not stop.is_set() or pending:
   if not pending:stop.wait(.001);continue
   item=pending[0];now=rospy.Time.now().to_sec()
   if now<item['activation']:time.sleep(min(.001,item['activation']-now));continue
   pending.popleft();m=latest[0]
   row=dict(frame=item['frame'],query=item['query'],planned=item['activation'],actual=now,passed=False)
   try:
    if now-item['activation']>.01:raise ValueError('ACTIVATION_LATE')
    if m is None or not 0<=now-m.stamp<=.05:raise ValueError('NO_FRESH_ACTUAL_STATE')
    pred=m.predict(now);head=item['prefix'].pva(min(now-item['activation'],item['prefix'].duration))
    residual=np.linalg.norm(pred-head,axis=1)
    if (residual>[.02,.05,.1]).any():raise ValueError('ACTIVATION_STATE_TUBE')
    row.update(passed=True,residual=residual.tolist(),measurement_stamp=m.stamp,selected=not any(g.get('passed') and g['frame']==row['frame'] for g in gated))
   except ValueError as exc:row['reason']=str(exc)
   gated.append(row)
 worker=threading.Thread(target=gate_worker);worker.start()
 sub=rospy.Subscriber('/sim/odom',Odometry,odom,queue_size=20,tcp_nodelay=True)
 esub=rospy.Subscriber('/reference/nominal_start',TimeMessage,epoch_cb,queue_size=1)
 ss=[message_filters.Subscriber('/query_shadow/depth',Image,queue_size=2),message_filters.Subscriber('/query_shadow/camera_info',CameraInfo,queue_size=2),message_filters.Subscriber('/sim/odom',Odometry,queue_size=200)]
 sync=message_filters.TimeSynchronizer(ss,200);sync.registerCallback(synced)
 with (run/'READY.json').open('x') as f:json.dump(dict(model=binding,device='cpu',control=False,visualization=False),f)
 event=(run/'events.jsonl').open('x',buffering=1)
 def log(x):event.write(json.dumps(x,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
 try:
  while not rospy.is_shutdown() and time.monotonic()-start<48 and attempt<10:
   if (run/'STOP_COLLECTION').exists():break
   if not frames:time.sleep(.002);continue
   d,c,o,recv=frames.pop();frames.clear();now=rospy.Time.now().to_sec()
   if epoch[0] is None:continue
   t=d.header.stamp.to_sec();pose=o.pose.pose;q=pose.orientation;v=o.twist.twist.linear
   pos=np.array([pose.position.x,pose.position.y,pose.position.z]);vel=np.array([v.x,v.y,v.z]);R=rotation([q.x,q.y,q.z,q.w])
   if not 0<=now-t<=.25:rejects['SOURCE_AGE']+=1;continue
   if previous_frame is None:previous_frame=(t,vel);continue
   dt=t-previous_frame[0]
   if dt<=0:continue
   acc=(vel-previous_frame[1])/dt;previous_frame=(t,vel.copy())
   if not .005<=dt<=.3:rejects['SOURCE_ACCEL_DT']+=1;continue
   tilt=np.arccos(np.clip(R[2,2],-1,1))
   if np.linalg.norm(vel-[.8,0,0])>.03 or np.linalg.norm(acc)>.05 or tilt>np.deg2rad(.5):rejects['JOINT_STATE_BOUND']+=1;continue
   try:
    state,goal=make_state(pos,R,vel,acc,np.array([3,0,2]))
    camera_check(c.width,c.height,c.K,c.D,c.R,c.P)
    if d.header.stamp.to_nsec()!=c.header.stamp.to_nsec() or d.header.stamp.to_nsec()!=o.header.stamp.to_nsec():raise ValueError('SYNC')
    publishers=master.getSystemState()[0]
    if dict(publishers).get('/so3_control/pos_cmd')!=['/f3_reference_arbiter']:raise ValueError('AUTHORITY')
    if any('/query_c1_proposal_shadow' in ns and topic!='/rosout' for topic,ns in publishers):raise ValueError('UNEXPECTED_C1_PUBLISHER')
    depth=decode_image(d.data,d.encoding,d.width,d.height,d.step,d.is_bigendian)
   except ValueError as ex:rejects[str(ex)]+=1;continue
   index=attempt;attempt+=1;artifacts={};batch=dict(frame=index,source_stamp=t,arrival=recv,global_goal=[3,0,2],local_goal=goal,queries=[])
   prep_begin=time.perf_counter();adapter.inputs(depth[None],state[None],None)
   with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
    td=torch.from_numpy(depth[None].copy());ts=torch.from_numpy(state[None].copy());n0=time.perf_counter()
    raw=adapter.model(td,ts);n1=time.perf_counter();decoded=adapter.post(raw,ts,num_samples=65)
    out=adapter.export(decoded,raw,decoded.samples);n2=time.perf_counter()
   batch['network_ms']=(n1-n0)*1000;batch['decode_export_ms']=(n2-n1)*1000;batch['six_generation_ms']=(n2-n0)*1000
   if out['coefficients'].shape[1]!=6 or not out['solve_valid'].all():raise ValueError('MINCO_INVALID')
   planning_ros=rospy.Time.now().to_sec();activation=planning_ros+.06
   volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],planning_ros+120,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
   for qi in range(6):
    q0=time.perf_counter();row=dict(query=qi,passed=False);m=latest[0]
    try:
     checknow=rospy.Time.now().to_sec()
     if checknow>=activation-.02:raise ValueError('SHARED_EXECUTION_BUDGET_EXPIRED')
     if m is None:raise ValueError('NO_CURRENT_ODOM')
     world=Program(world_coefficients(out['coefficients'][0,qi],pos,R),out['durations'][0,qi])
     adaptation=constrained_prefix(world,m,activation,volume,checknow,active=known,source_stamp=t)
     artifacts['q%d_prefix_coefficients'%qi]=adaptation['prefix'].coefficients
     artifacts['q%d_prefix_durations'%qi]=adaptation['prefix'].durations
     artifacts['q%d_endpoint_brake_coefficients'%qi]=adaptation['brake'].coefficients
     artifacts['q%d_endpoint_brake_durations'%qi]=adaptation['brake'].durations
     row['connection_retime_endpoint_proof_ms']=adaptation['compute_s']*1000
     row.update(endpoint_prefix_kinematic_and_space_pass=True,connection_s=adaptation['connection_s'],connection_progress_m=float(np.linalg.norm(adaptation['prefix'].pva(adaptation['connection_s'])[0]-adaptation['prefix'].pva(0)[0])),
       retained_s=adaptation['retained_physical_s'],retained_progress_m=adaptation['retained_progress_m'],end_brake=adaptation['brake_bounds'],full_extrema=adaptation['full_extrema'])
     bank0=time.perf_counter()
     try:
      committed=prepare_reference(adaptation['prefix'],volume,checknow,time.monotonic())
     finally:row['all_phase_brake_bank_ms']=(time.perf_counter()-bank0)*1000
     ready=rospy.Time.now().to_sec();row['ready_ros']=ready
     if ready>activation-.02:raise ValueError('COMPLETE_BANK_MISSED_READY_DEADLINE')
     if ready-t>.25:raise ValueError('COMPLETE_BANK_STALE_SOURCE')
     pending.append(dict(frame=index,query=qi,activation=activation,prefix=adaptation['prefix']))
     row.update(passed=True,activation=activation)
    except ValueError as ex:row['reason']=str(ex)
    row['candidate_total_ms']=(time.perf_counter()-q0)*1000;batch['queries'].append(row)
   batch['all_six_processing_ms']=(time.perf_counter()-prep_begin)*1000
   io0=time.perf_counter()
   with (run/('frame_%03d.npz'%index)).open('xb') as f:np.savez_compressed(f,depth=depth,state=state,position_W=pos,rotation_W_B0=R,source_stamp=np.array(t),**{k:v[0] for k,v in out.items()},**artifacts)
   batch['evidence_IO_ms']=(time.perf_counter()-io0)*1000;batch['source_to_all_ready_ms']=(rospy.Time.now().to_sec()-t)*1000
   log(batch);stats.append(batch)
 finally:
  stop.set();worker.join(3);sub.unregister();esub.unregister()
  for s in ss:s.unregister()
  event.close()
  with (run/'RESULT.json').open('x') as f:json.dump(dict(mode='C1_READ_ONLY_PROPOSALS_NOT_CONTROL',models=binding,frames=len(stats),generated=6*len(stats),proposal_precommit_pass=sum(q['passed'] for b in stats for q in b['queries']),activation_gates=gated,
    joint_input_rejects=rejects,raw_scores_used=False,GT_Oracle_used=False,external_scope='EXTERNAL_SUPERVISOR',control=False,visualization=False),f,indent=2)
  rospy.signal_shutdown('boundedC1shadowcomplete')
if __name__=='__main__':main()
