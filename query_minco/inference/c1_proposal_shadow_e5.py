"""CPU-prepared E5 ledger/fault fix, no new ROS trial performed.
Uses the exact E4 neural inputs, frozen inference and certified numerical worker.
"""
from query_minco.inference import c1_proposal_shadow_e4 as previous
from query_minco.inference.input_lifecycle import InputLifecycle,LatestInputMailbox,FaultLatch,SessionStopped
from query_minco.inference.session_inbox import GuardedSourceInbox
def main_source():
 s=previous.main_source()
 s=s.replace("input_records=[];capture_closed=[False];", "session=FaultLatch();lifecycle=InputLifecycle();mailbox=LatestInputMailbox(lifecycle,session);current_ticket=None\n input_records=[];capture_closed=[False];")
 old="inbox=SourceBoundInbox(json.loads((run.parent/'DYNAMIC_READY.json').read_text())['address'])"
 new="ready_identity=json.loads((run.parent/'DYNAMIC_READY.json').read_text());inbox=GuardedSourceInbox(ready_identity['address'],session,expected_peer_pid=int(ready_identity['pid']))"
 assert s.count(old)==1;s=s.replace(old,new)
 a=s.index(' def receive_frame(d,c,o):');b=s.index(' sync=message_filters.TimeSynchronizer',a)
 s=s[:a]+""" def receive_frame(d,c,o):
  if capture_closed[0]:return
  received=time.monotonic();input_counters['synced_received']+=1
  mailbox.receive((d,c,o,received),d.header.stamp.to_nsec(),int(received*1e9))
"""+s[b:]
 old="""with frame_lock:item=frames.pop() if frames else None
   if item is None:time.sleep(.001);continue
   d,c,o,recv=item;now=rospy.Time.now().to_sec()"""
 new="""taken=mailbox.take()
   if taken is None:time.sleep(.001);continue
   current_ticket,item=taken;session.require_live()
   d,c,o,recv=item;now=rospy.Time.now().to_sec()"""
 assert old in s;s=s.replace(old,new)
 s=s.replace("while not rospy.is_shutdown() and time.monotonic()-start<48", "while not session.stopped and not rospy.is_shutdown() and time.monotonic()-start<48")
 s=s.replace("rejects[reason]+=1;input_records.append", "rejects[reason]+=1\n  if current_ticket is not None:lifecycle.finish(current_ticket,'ABORTED' if session.stopped else 'REJECTED',reason)\n  input_records.append")
 s=s.replace("ledger_context.update(source_stamp=", "ledger_context.update(input_id=current_ticket.sequence,source_stamp=")
 s=s.replace("index=len(stats)\n   input_records.append", "session.require_live()\n   index=len(stats)\n   input_records.append")
 s=s.replace("batch=dict(frame=index,", "batch=dict(input_id=current_ticket.sequence,frame=index,")
 # Fault while an already-started inference is in flight: retain arrays/stages,
 # break candidate attempts and then produce exactly one ABORTED terminal.
 line="row['candidate_total_ms']=(time.perf_counter()-qt)*1000;row['candidate_thread_cpu_ms']=(time.thread_time_ns()-qt_cpu)/1e6;batch['queries'].append(row)"
 assert line in s;s=s.replace(line,line+"\n    if session.stopped:break")
 s=s.replace("queue.put_nowait((batch,arrays,banks));stats.append(batch)", "queue.put_nowait((batch,arrays,banks));stats.append(batch)\n   lifecycle.finish(current_ticket,'ABORTED' if session.stopped else 'COMPLETED' if batch['selection']!='NONE' else 'REJECTED',session.reason if session.stopped else 'DECISION_READY' if batch['selection']!='NONE' else 'NO_READY_PROPOSAL',frame=index,selection=batch['selection'])")
 s=s.replace("capture_closed[0]=True;stop.set();", "capture_closed[0]=True;mailbox.close(session.reason or 'COLLECTION_END');stop.set();")
 s=s.replace("with (run/'INPUT_LEDGER.jsonl').open('x') as ledger:", "terminal_snapshot=lifecycle.snapshot()\n  if terminal_snapshot['open'] or terminal_snapshot['received']!=len(terminal_snapshot['terminals']):raise RuntimeError('INPUT_TERMINALS_INCOMPLETE')\n  with (run/'INPUT_TERMINALS.jsonl').open('x') as terminal_file:\n   for record in terminal_snapshot['terminals']:terminal_file.write(json.dumps(record)+'\\n')\n  with (run/'INPUT_LEDGER.jsonl').open('x') as ledger:")
 s=s.replace("joint_input_rejects=rejects,", "input_terminal_counts=dict(received=terminal_snapshot['received'],terminal=len(terminal_snapshot['terminals']),open=terminal_snapshot['open'],mailbox_received=mailbox.received,mailbox_taken=mailbox.taken,queue_replaced=mailbox.replaced),session_fault=session.reason,joint_input_rejects=rejects,")
 s=s.replace(" finally:\n  capture_closed", " except SessionStopped as exc:\n  session.trip(str(exc))\n except Exception as exc:\n  session.trip(\'INPUT_PROCESSING_EXCEPTION:\'+type(exc).__name__);raise\n finally:\n  capture_closed")
 return s
def main():
 env=dict(vars(previous.v4.original));env.update(InputLifecycle=InputLifecycle,LatestInputMailbox=LatestInputMailbox,FaultLatch=FaultLatch,GuardedSourceInbox=GuardedSourceInbox,SessionStopped=SessionStopped)
 exec(compile(main_source(),__file__+'[terminal-session-overlay]','exec'),env);env['main']()
if __name__=='__main__':main()
