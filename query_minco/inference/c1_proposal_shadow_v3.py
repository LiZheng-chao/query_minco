"""F3-E2/E3 instrumented native-worker frozen C1 readonly proposals. No Publisher/control message import.
Same native host and Controller; numerical preparation only. Disk work is in
an owned writer process, never the proposal or native 50Hz publication path.
"""
import argparse,collections,json,multiprocessing,pickle,threading,time,traceback,os,hashlib
from pathlib import Path
import numpy as np
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.shadow import FROZEN,decode_image,camera_check,make_state,rotation
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.trajectory_execution import Program,Measurement
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope,ErrorBudget
from query_minco.inference.candidate_feasibility import build_prefix
from query_minco.inference.proposal_worker import NativeWorker,curve_bytes
from query_minco.inference.dynamic_proposal import Inbox,metadata,packet

def evidence_writer(run,queue):
    with (run/'events.jsonl').open('x',buffering=1) as events:
        while True:
            item=queue.get()
            if item is None:break
            batch,arrays,banks=item;start=time.perf_counter()
            with (run/('frame_%03d.npz'%batch['frame'])).open('xb') as f:np.savez_compressed(f,**arrays)
            if banks:
                with (run/('banks_%03d.pkl'%batch['frame'])).open('xb') as f:pickle.dump(banks,f,protocol=4)
            batch['async_evidence_IO_ms']=(time.perf_counter()-start)*1000
            events.write(json.dumps(batch,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')

def main():
 import rospy,message_filters,rosgraph,torch
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo
 from std_msgs.msg import Time as TimeMessage
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--references',required=True);a=p.parse_args()
 run=Path(a.run);run.mkdir()
 ctx=multiprocessing.get_context('fork');queue=ctx.Queue(maxsize=8)
 writer=ctx.Process(target=evidence_writer,args=(run,queue));writer.start()
 binding=next(b for b in json.loads(FROZEN.read_text()) if b['label']=='C1')
 adapter=InferenceAdapter(binding['checkpoint']['path'],binding['method_id'],binding['checkpoint']['sha256']);adapter.model.requires_grad_(False)
 torch.set_num_threads(1)
 # Allocation/kernel warmup only. These are NOT sensor frames or valid proposals.
 with torch.inference_mode():
  td=torch.zeros((1,2,128,160));ts=torch.tensor([[.8,0,0,0,0,0,2.4,0,0,0,0,-1]],dtype=torch.float32)
  for _ in range(4):
   raw=adapter.model(td,ts);adapter.post(raw,ts,num_samples=65)
 with Path(a.references).open('rb') as f:bundles=pickle.load(f)
 receipt=json.loads(bundles[0].prefix_proof)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],receipt['valid_until'],receipt['evidence'][0]['cell']),),Scope.EXTERNAL_SUPERVISOR,ErrorBudget(**receipt['error_budget']))
 config=json.loads((run.parents[1]/'DYNAMIC_CONFIG.json').read_text())
 worker_native=NativeWorker(run.parents[1]/'runtime/libproposal_numerics.so')
 deadline=time.monotonic()+5
 while not (run.parent/'DYNAMIC_READY.json').exists():
  if time.monotonic()>deadline:raise ValueError('CPP_READONLY_INBOX_NOT_READY')
  time.sleep(.01)
 inbox=Inbox(json.loads((run.parent/'DYNAMIC_READY.json').read_text())['address'])
 nominal=bundles[1].prefix;epoch=[None];latest=[None];prev=[None];frames=collections.deque(maxlen=1);topology=[None]
 stats=[];rejects=collections.Counter();pending=collections.deque();gated=[];stop=threading.Event();previous_frame=None;start=time.monotonic()
 rospy.init_node('query_c1_proposal_shadow',disable_signals=True)
 master=rosgraph.Master('/query_c1_proposal_shadow')
 def graph_worker():
  while not stop.is_set():
   try:
    pubs=master.getSystemState()[0]
    ok=dict(pubs).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'] and not any('/query_c1_proposal_shadow' in ns and topic!='/rosout' for topic,ns in pubs)
    topology[0]=(time.monotonic(),ok,pubs)
   except Exception:topology[0]=None
   stop.wait(.1)
 graph=threading.Thread(target=graph_worker);graph.start()
 esub=rospy.Subscriber('/reference/nominal_start',TimeMessage,lambda m:epoch.__setitem__(0,m.data.to_sec()),queue_size=1)
 ss=[message_filters.Subscriber('/query_shadow/depth',Image,queue_size=2),message_filters.Subscriber('/query_shadow/camera_info',CameraInfo,queue_size=2),message_filters.Subscriber('/sim/odom',Odometry,queue_size=200)]
 sync=message_filters.TimeSynchronizer(ss,200);sync.registerCallback(lambda d,c,o:frames.append((d,c,o,time.monotonic())))
 with (run/'READY.json').open('x') as f:json.dump(dict(model=binding,device='cpu',control=False,visualization=False,warmup_frames_counted=0,pid=os.getpid(),inherited_cell_expiry=receipt['valid_until']),f)
 try:
  while not rospy.is_shutdown() and time.monotonic()-start<48 and len(stats)<40:
   if (run/'STOP_COLLECTION').exists():break
   if not frames:time.sleep(.001);continue
   d,c,o,recv=frames.pop();frames.clear();now=rospy.Time.now().to_sec()
   if epoch[0] is None:continue
   t=d.header.stamp.to_sec();pose=o.pose.pose;q=pose.orientation;v=o.twist.twist.linear
   pos=np.array([pose.position.x,pose.position.y,pose.position.z]);vel=np.array([v.x,v.y,v.z]);R=rotation([q.x,q.y,q.z,q.w])
   if not 0<=now-t<=.25:rejects['SOURCE_AGE']+=1;continue
   if previous_frame is None:previous_frame=(t,vel);continue
   dt=t-previous_frame[0];previous_frame=(t,vel.copy())
   if not .005<=dt<=.3:rejects['SOURCE_ACCEL_DT']+=1;continue
   acc=(vel-previous_frame[1])/dt;tilt=np.arccos(np.clip(R[2,2],-1,1))
   if np.linalg.norm(vel-[.8,0,0])>.03 or np.linalg.norm(acc)>.05 or tilt>np.deg2rad(.5):rejects['JOINT_STATE_BOUND']+=1;continue
   try:
    state,goal=make_state(pos,R,vel,acc,np.array([3,0,2]));camera_check(c.width,c.height,c.K,c.D,c.R,c.P)
    if not d.header.stamp.to_nsec()==c.header.stamp.to_nsec()==o.header.stamp.to_nsec():raise ValueError('SYNC')
    if topology[0] is None or not topology[0][1] or time.monotonic()-topology[0][0]>.5:raise ValueError('AUTHORITY')
    depth=decode_image(d.data,d.encoding,d.width,d.height,d.step,d.is_bigendian)
   except ValueError as ex:rejects[str(ex)]+=1;continue
   index=len(stats);batch=dict(frame=index,source_stamp=t,arrival_mono=recv,global_goal=[3,0,2],local_goal=goal,queries=[],selection='NONE',scope='EXTERNAL_SUPERVISOR',control=False)
   arrays=dict(depth=depth,state=state,position_W=pos,rotation_W_B0=R,source_stamp=np.array(t));banks={};critical=time.perf_counter()
   with torch.inference_mode():
    td=torch.from_numpy(depth[None].copy());ts=torch.from_numpy(state[None].copy());n0=time.perf_counter();nc0=time.thread_time_ns()
    raw=adapter.model(td,ts);n1=time.perf_counter();nc1=time.thread_time_ns();decoded=adapter.post(raw,ts,num_samples=65);data=adapter.export(decoded,raw,decoded.samples);n2=time.perf_counter();nc2=time.thread_time_ns()
   batch.update(network_ms=(n1-n0)*1000,decode_export_ms=(n2-n1)*1000,six_generation_ms=(n2-n0)*1000,network_thread_cpu_ms=(nc1-nc0)/1e6,decode_thread_cpu_ms=(nc2-nc1)/1e6)
   if data['coefficients'].shape[1]!=6 or not data['solve_valid'].all():raise ValueError('MINCO_INVALID')
   arrays.update({k:v[0] for k,v in data.items()})
   planning_ros=rospy.Time.now().to_sec();planning_mono=time.monotonic();activation=planning_ros+.06;activation_mono=planning_mono+.06;batch.update(planning_ros=planning_ros,activation_ros=activation)
   for qi in (5,4,0,1,2,3):
    row=dict(query=qi,passed=False);qt=time.perf_counter();qt_cpu=time.thread_time_ns()
    if batch['selection']!='NONE':
     row['reason']='NOT_CHECKED_AFTER_FIRST_COMPLETE_PROPOSAL';row['candidate_total_ms']=0.;batch['queries'].append(row);continue
    try:
     checknow=rospy.Time.now().to_sec()
     if activation-.02-checknow<.005:raise ValueError('INSUFFICIENT_REMAINING_PREPARATION_BUDGET')
     snapshot=inbox.snapshot();checknow=rospy.Time.now().to_sec();checkmono=time.monotonic()
     m=Measurement(float(snapshot['stamp']),np.array(list(map(float,snapshot['pva']))).reshape(3,3));mrecv=float(snapshot['recv_mono'])
     if not 0<=checknow-m.stamp<=.05 or not 0<=time.monotonic()-mrecv<=.05:raise ValueError('STALE_CURRENT_ODOM')
     row['cpp_snapshot_roundtrip_ms']=snapshot['client_wall_ms']
     if not 0<=checknow-t<=.25:raise ValueError('STALE_SOURCE')
     head=nominal.pva(activation-epoch[0]);residual=np.linalg.norm(m.predict(activation)-head,axis=1)
     if (residual>[.02,.05,.1]).any():raise ValueError('PREDICTED_TRACKING_TUBE')
     world=Program(world_coefficients(data['coefficients'][0,qi],pos,R),data['durations'][0,qi])
     if not .8-1e-6<=world.duration<=2.2+1e-6:raise ValueError('SOURCE_HORIZON')
     conn,ret,prefix=build_prefix(world,head,1.,.7,.5);cend=time.perf_counter()
     if np.linalg.norm(ret.pva(ret.duration)[0]-ret.pva(0)[0])<.005:raise ValueError('INSUFFICIENT_ORIGINAL_MINCO_PROGRESS')
     row['prefix_kinematic_stage_started']=True
     bank=worker_native.prepare(prefix,volume,checknow,checkmono,inherited_mono_expiry=float(snapshot['cell_expiry_mono']));bend=time.perf_counter()
     row.update(full_bank_and_sweep_pass=True,worker_timing=bank.timing)
     arrays['q%d_bank_body'%qi]=np.frombuffer(bank.body,dtype=np.uint8)
     envelope=metadata(bank,snapshot,index*6+qi+1,t,pos,R,activation,activation_mono,config)
     encode_end=time.perf_counter()
     acknowledgement=inbox.submit(envelope,bank.body)
     row.update(cpp_submission_pass=True,cpp_receipt=acknowledgement,packet_encode_ms=(encode_end-bend)*1000)
     ready=rospy.Time.now().to_sec()
     row.update(connection_ms=(cend-qt)*1000,complete_bank_ms=(bend-cend)*1000,ready_ros=ready,source_to_complete_ms=(ready-t)*1000,measurement_stamp=m.stamp,measurement_received_mono=mrecv,measurement_pva=m.pva.tolist(),planned_head_pva=head.tolist(),predicted_residual=residual.tolist())
     if ready>activation-.02:raise ValueError('COMPLETE_BANK_MISSED_READY_DEADLINE')
     if ready-t>.25:raise ValueError('COMPLETE_BANK_STALE_SOURCE')
     horizon=bank.horizon
     row.update(passed=True,connection_s=1.,source_phase=.7,source_rate=.5,retained_s=.1,bank_phases=bank.phase_count,complete_horizon_s=horizon,certificate_expiry=bank.valid_until_ros)
     batch['selection']=qi
     banks[qi]=dict(bank=bank,envelope=envelope)
     arrays.update({'q%d_prefix_coefficients'%qi:prefix.coefficients,'q%d_prefix_durations'%qi:prefix.durations,'q%d_retained_coefficients'%qi:ret.coefficients,'q%d_retained_durations'%qi:ret.durations})
    except ValueError as ex:row['reason']=str(ex)
    row['candidate_total_ms']=(time.perf_counter()-qt)*1000;row['candidate_thread_cpu_ms']=(time.thread_time_ns()-qt_cpu)/1e6;batch['queries'].append(row)
   batch.update(six_policy_processing_ms=(time.perf_counter()-critical)*1000,post_generation_ms=(time.monotonic()-planning_mono)*1000,source_to_decision_ms=(rospy.Time.now().to_sec()-t)*1000)
   queue.put_nowait((batch,arrays,banks));stats.append(batch)
 finally:
  stop.set();graph.join(3);esub.unregister()
  for subscriber in ss:subscriber.unregister()
  queue.put(None);writer.join(8)
  if writer.is_alive():raise RuntimeError('EVIDENCE_WRITER_DID_NOT_DRAIN')
  if writer.exitcode:raise RuntimeError('EVIDENCE_WRITER_FAILED')
  with (run/'RESULT.json').open('x') as f:json.dump(dict(mode='C1_READ_ONLY_PROPOSALS_NOT_CONTROL',model=binding,frames=len(stats),generated=6*len(stats),proposal_precommit_pass=sum(q['passed'] for b in stats for q in b['queries']),activation_gates=[],actual_gate_authority='../DYNAMIC_GATE.jsonl',joint_input_rejects=rejects,raw_scores_used=False,GT_Oracle_used=False,external_scope='EXTERNAL_SUPERVISOR',control=False,visualization=False),f,indent=2)
  rospy.signal_shutdown('bounded readonly C1 complete')
if __name__=='__main__':main()
