"""E4 uses the preserved v4 correction, real source receipts and explicit ledger.
No ROS Publisher. Frozen model/decoders unchanged. All writes are run evidence.
"""
import inspect,re
from query_minco.inference import c1_proposal_shadow_v4 as v4
from query_minco.inference.causal_source_state import SourceBoundInbox
def main_source():
 s=v4.corrected_main_source()
 s=s.replace("stats=[];rejects=collections.Counter();","input_records=[];capture_closed=[False];frame_lock=threading.Lock();input_counters=collections.Counter();ledger_context={}\n stats=[];rejects=collections.Counter();")
 s=re.sub(r"rejects\[(.*?)\]\+=1;continue",r"record_reject(\1);continue",s)
 needle=" sync=message_filters.TimeSynchronizer(ss,200);sync.registerCallback(lambda d,c,o:frames.append((d,c,o,time.monotonic())))"
 replacement=""" def record_reject(reason):
  rejects[reason]+=1;input_records.append(dict(ledger_context,kind='REJECT',reason=reason,decision_mono=time.monotonic()))
 def receive_frame(d,c,o):
  if capture_closed[0]:return
  received=time.monotonic();input_counters['synced_received']+=1
  with frame_lock:
   if frames:
    dropped=frames.popleft();input_counters['queue_replaced']+=1
    input_records.append(dict(kind='QUEUE_REPLACED',source_stamp=dropped[0].header.stamp.to_sec(),received_mono=dropped[3],replaced_mono=received))
   frames.append((d,c,o,received))
 sync=message_filters.TimeSynchronizer(ss,200);sync.registerCallback(receive_frame)"""
 assert s.count(needle)==1;s=s.replace(needle,replacement)
 old="""if not frames:time.sleep(.001);continue
   d,c,o,recv=frames.pop();frames.clear();now=rospy.Time.now().to_sec()
   if epoch[0] is None:continue"""
 new="""with frame_lock:item=frames.pop() if frames else None
   if item is None:time.sleep(.001);continue
   d,c,o,recv=item;now=rospy.Time.now().to_sec()
   input_counters['processed']+=1
   ledger_context.clear();ledger_context.update(source_stamp=d.header.stamp.to_sec(),received_mono=recv,processing_start_mono=time.monotonic(),processing_ros=now)
   if epoch[0] is None:record_reject('WAIT_NOMINAL_EPOCH');continue"""
 assert old in s;s=s.replace(old,new)
 s=s.replace("vel=source_pva[1].copy();acc=source_pva[2].copy()","vel=source_pva[1].copy();acc=source_pva[2].copy()\n    ledger_context.update(source_pva=source_pva.tolist(),source_received_mono=float(source['recv_mono']),source_previous_stamp=float(source['previous_stamp']),source_previous_velocity=list(map(float,source['previous_velocity'])))")
 s=s.replace("tilt=np.arccos(np.clip(R[2,2],-1,1))","tilt=np.arccos(np.clip(R[2,2],-1,1));ledger_context.update(speed_error=float(np.linalg.norm(vel-[.8,0,0])),acceleration_norm=float(np.linalg.norm(acc)),tilt_deg=float(np.rad2deg(tilt)))")
 old="index=len(stats);batch=dict("
 new="""input_sha=hashlib.sha256(depth.astype('<f4').tobytes()).hexdigest()
   if input_sha!=source['input_depth_mask_sha256']:record_reject('INDEPENDENT_DEPTH_MASK_HASH');continue
   index=len(stats)
   input_records.append(dict(ledger_context,kind='NETWORK_INPUT',frame=index,state12=state.tolist(),state_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest(),input_depth_mask_sha256=input_sha,network_start_mono=time.monotonic()))
   batch=dict("""
 assert old in s;s=s.replace(old,new)
 s=s.replace("arrays=dict(depth=depth,state=state,","arrays=dict(source_pva=source_pva,source_previous_stamp=float(source['previous_stamp']),source_previous_velocity=np.array(list(map(float,source['previous_velocity']))),depth=depth,state=state,")
 needle="envelope.update(source_state_version=2,source_pva=source_pva.reshape(-1).tolist(),network_state=state.tolist(),network_state_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest())"
 s=s.replace(needle,needle+"\n     envelope.update(input_depth_mask_sha256=input_sha,camera_info_stamp=c.header.stamp.to_sec(),camera_frame=c.header.frame_id,source_previous_stamp=float(source['previous_stamp']),source_previous_velocity=list(map(float,source['previous_velocity'])))")
 s=s.replace("stop.set();graph.join(3);esub.unregister()","capture_closed[0]=True;stop.set();graph.join(3);esub.unregister()")
 s=s.replace("with (run/'RESULT.json').open('x') as f:", "with (run/'INPUT_LEDGER.jsonl').open('x') as ledger:\n   for record in input_records:ledger.write(json.dumps(record)+'\\n')\n  with (run/'RESULT.json').open('x') as f:")
 s=s.replace("joint_input_rejects=rejects,","joint_input_rejects=rejects,input_counters=input_counters,")
 return s
def main():
 env=dict(vars(v4.original));env['SourceBoundInbox']=SourceBoundInbox
 exec(compile(main_source(),__file__+'[E4-source-receipt-overlay]','exec'),env);env['main']()
if __name__=='__main__':main()
