"""Read-only native command receipt observer. Cannot publish a control command."""
from pathlib import Path
import sys,time,json,threading,hashlib
import rospy,rosgraph
from quadrotor_msgs.msg import PositionCommand
sys.path.insert(0,'/home/lizheng/query_minco_mainline')
from query_minco.integration.pg2_stagec_v1.lease import Arbiter,Packet,State
out=Path(sys.argv[1]);rospy.init_node('query_lease_diagnostic',disable_signals=True);lock=threading.Lock();a=Arbiter(.1);a.shadow(rospy.Time.now().to_sec());last=None;counts={};logs=(out/'LEASE_DIAGNOSTIC.jsonl').open('x',buffering=1);master=rosgraph.Master(rospy.get_name())
def emit(d):logs.write(json.dumps(d)+'\n')
def cb(m):
 global last
 now=rospy.Time.now().to_sec();stamp=m.header.stamp.to_sec();rx=time.monotonic();flag=int(m.trajectory_flag)
 with lock:
  previous=last;last=dict(stamp=stamp,receipt_monotonic=rx,flag=flag,header_seq=m.header.seq,trajectory_id=m.trajectory_id)
  record=dict(kind='RECEIPT',ros_now=now,received_wall_ns=time.time_ns(),source_stamp=stamp,source_age_s=now-stamp,flag=flag,header_seq=m.header.seq,trajectory_id=m.trajectory_id,pva=[m.position.x,m.position.y,m.position.z,m.velocity.x,m.velocity.y,m.velocity.z,m.acceleration.x,m.acceleration.y,m.acceleration.z])
  if flag==1:
   try:
    # Only a diagnostic source-stamp order check; no invented native execution sequence contract.
    p=Packet('native_yopo_observed',m.header.stamp.to_nsec(),m.header.stamp.to_nsec(),stamp,stamp+.1,hashlib.sha256(m.serialize.__name__.encode()+repr(record['pva']).encode()).hexdigest())
    a.receive(p,now);record['lease']='READY_NO_AUTHORITY'
   except ValueError as ex:record['lease_rejected']=str(ex)
  else:record['legacy_hover']='OBSERVED_ONLY: reuse of old stamp is preserved, no authority or new motion reference inferred'
  emit(record)
sub=rospy.Subscriber('/so3_control/pos_cmd',PositionCommand,cb,queue_size=300);start=time.monotonic();last_graph=0;expired=False;expiry_events=[];graphs=[]
try:
 while time.monotonic()-start<40 and not rospy.is_shutdown():
  now=rospy.Time.now().to_sec()
  with lock:
   try:state=a.tick(now).value
   except ValueError:state=a.state.value
   if last:
    age=time.monotonic()-last['receipt_monotonic'];bad=age>.1
    if bad and not expired:
     event=dict(kind='RECEIPT_DEADLINE_EXCEEDED',received_wall_ns=time.time_ns(),receipt_gap_s=age,source_stamp=last['stamp'],source_age_s=now-last['stamp'],last_flag=last['flag'],state=state,physical_stop_verified=False);expiry_events.append(event);emit(event)
    expired=bad
   assert not a.query_permission
  if time.monotonic()-last_graph>.5:
   graph=master.getSystemState();pub=dict(graph[0]);assert set(t for t,nodes in graph[0] if rospy.get_name() in nodes)<={'/rosout'},'DIAGNOSTIC_CONTROL_PUBLISHER';graphs.append(dict(wall_ns=time.time_ns(),pos_cmd_owners=pub.get('/so3_control/pos_cmd',[])));last_graph=time.monotonic()
  time.sleep(.02)
finally:
 with lock:
  (out/'LEASE_RESULT.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',state=a.snapshot(),receipt_expiry_events=expiry_events,graphs=graphs,physical_fallback='BLOCKED_NOT_IMPLEMENTED',query_publications=0,sequence_status='NATIVE_REFERENCE_SEQUENCE_NOT_QUALIFIED; source stamp order observed only',poll_s=.02,ttl_s=.1),indent=2))
 logs.close();rospy.signal_shutdown('diagnostic observer ended')
