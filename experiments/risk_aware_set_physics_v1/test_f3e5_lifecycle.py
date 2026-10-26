"""CPU/local-UDS/standalone-session tests only. No ROS or Simulator compilation."""
import collections,dataclasses,json,os,socket,struct,subprocess,threading,time,unittest
from pathlib import Path
from query_minco.inference.input_lifecycle import *
from query_minco.inference.session_inbox import GuardedSourceInbox
from query_minco.inference.c1_proposal_shadow_e5 import main_source
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution009'
class Tests(unittest.TestCase):
 def test_exactly_one_terminal(self):
  l=InputLifecycle();t=l.receive(100,10);l.begin(t,11);l.finish(t,'COMPLETED','READY',12)
  with self.assertRaisesRegex(ValueError,'DUPLICATE'):l.finish(t,'REJECTED','bad',13)
  self.assertEqual(l.snapshot()['terminals'][0]['status'],'COMPLETED')
 def test_invalid_status_keeps_open(self):
  l=InputLifecycle();t=l.receive(100,10)
  with self.assertRaises(ValueError):l.finish(t,'INVALID','',11)
  self.assertEqual(l.snapshot()['open'],1);l.close(now_ns=12);self.assertEqual(len(l.snapshot()['terminals']),1)
 def test_forged_ticket_keeps_original(self):
  l=InputLifecycle();t=l.receive(100,10)
  with self.assertRaises(ValueError):l.finish(dataclasses.replace(t,source_stamp_ns=101),'COMPLETED',now_ns=11)
  self.assertEqual(l.snapshot()['open'],1);l.close(now_ns=12)
 def test_latest_queue_replacement(self):
  l=InputLifecycle();m=LatestInputMailbox(l,FaultLatch())
  m.receive('old',1,10);m.receive('new',2,11);t,p=m.take(12);self.assertEqual(p,'new')
  l.finish(t,'COMPLETED',now_ns=13);m.close(now_ns=14)
  self.assertEqual(collections.Counter(x['status'] for x in l.snapshot()['terminals']),{'DROPPED':1,'COMPLETED':1})
 def test_duplicate_source_never_replayed(self):
  l=InputLifecycle();m=LatestInputMailbox(l,FaultLatch());m.receive('one',100,10);m.receive('replay',100,11)
  t,p=m.take(12);self.assertEqual(p,'one');m.close(now_ns=13)
  self.assertEqual(sum(x['reason']=='DUPLICATE_OR_OLD_SOURCE' for x in l.snapshot()['terminals']),1)
 def test_clock_order_is_raw_abort_not_clamped(self):
  l=InputLifecycle();t=l.receive(100,20);self.assertFalse(l.begin(t,19))
  r=l.snapshot()['terminals'][0];self.assertFalse(r['clock_order_valid']);self.assertEqual(r['terminal_mono_ns'],19)
 def test_exception_closes_processing(self):
  l=InputLifecycle();m=LatestInputMailbox(l,FaultLatch());m.receive('input',100,10);m.take(11)
  try:raise RuntimeError('injected')
  except RuntimeError:m.close('EXCEPTION',12)
  self.assertEqual(l.snapshot()['open'],0);self.assertEqual(l.snapshot()['terminals'][0]['status'],'ABORTED')
 def test_fault_stops_rpc_and_inference_admission(self):
  latch=FaultLatch();sent=[];calls=[dict(passed=True),dict(passed=False,fault_latched=True,reason='SESSION_FAULT_LATCHED')]
  def transport(body):sent.append(body);return calls.pop(0)
  inbox=GuardedSourceInbox('mock',latch,transport=transport);inbox.source(1)
  with self.assertRaises(SessionStopped):inbox.source(2)
  with self.assertRaises(SessionStopped):inbox.source(3)
  self.assertEqual(len(sent),2);self.assertEqual(inbox.rpc_calls,2)
 def test_transient_reject_does_not_reuse_or_stop_session(self):
  latch=FaultLatch();answers=[dict(passed=False,reason='SOURCE_IMAGE_NOT_OBSERVED'),dict(passed=True)]
  inbox=GuardedSourceInbox('mock',latch,transport=lambda _:answers.pop(0))
  with self.assertRaisesRegex(ValueError,'SOURCE_IMAGE'):inbox.source(1)
  self.assertFalse(latch.stopped);self.assertTrue(inbox.source(2)['passed'])
 def test_inflight_preserved_then_fault_aborts_future(self):
  l=InputLifecycle();f=FaultLatch();m=LatestInputMailbox(l,f)
  t=m.receive('accepted',1,10);m.take(11);l.finish(t,'COMPLETED','READY',12);before=l.snapshot()['terminals'][0]
  f.trip('FAULT',13);m.receive('future',2,14);self.assertIsNone(m.take(15));m.close(now_ns=16)
  self.assertEqual(l.snapshot()['terminals'][0],before)
  self.assertEqual(l.snapshot()['terminals'][1]['status'],'ABORTED')
 def test_close_during_rpc_maps_to_abort(self):
  f=FaultLatch()
  def close(_):raise ConnectionRefusedError('unit closure')
  inbox=GuardedSourceInbox('mock',f,transport=close)
  with self.assertRaises(SessionStopped):inbox.source(1)
  self.assertTrue(f.stopped)
 def test_real_close_terminal_is_exactly_once(self):
  life=InputLifecycle();fault=FaultLatch();mailbox=LatestInputMailbox(life,fault)
  mailbox.receive('processed',1);taken=mailbox.take()
  def transport(_):raise EOFError('closed')
  inbox=GuardedSourceInbox('mock',fault,transport=transport)
  try:inbox.source(1)
  except SessionStopped:mailbox.close(fault.reason)
  state=life.snapshot();self.assertEqual(len(state['terminals']),1);self.assertEqual(state['terminals'][0]['status'],'ABORTED');self.assertEqual(state['open'],0)
 def test_local_uds_actual_protocol_no_ros(self):
  address='\0qm_e5_unit_'+str(os.getpid());server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(address);server.listen()
  seen=[];errors=[]
  def worker():
   try:
    for reply in [dict(passed=True),dict(passed=False,fault_latched=True,reason='SESSION_FINISHED')]:
     fd,_=server.accept()
     with fd:
      n=struct.unpack('<i',fd.recv(4))[0];body=b''
      while len(body)<n:body+=fd.recv(n-len(body))
      seen.append(body);encoded=json.dumps(reply).encode();fd.sendall(struct.pack('<i',len(encoded))+encoded)
   except Exception as e:errors.append(str(e))
   finally:server.close()
  thread=threading.Thread(target=worker);thread.start();f=FaultLatch();inbox=GuardedSourceInbox(address,f,expected_peer_pid=os.getpid())
  inbox.source(1)
  with self.assertRaises(SessionStopped):inbox.source(2)
  with self.assertRaises(SessionStopped):inbox.source(3)
  thread.join(2);self.assertFalse(thread.is_alive());self.assertEqual(len(seen),2);self.assertFalse(errors)
 def test_concurrent_received_all_have_one_terminal(self):
  life=InputLifecycle();f=FaultLatch();m=LatestInputMailbox(life,f)
  def producer():
   for i in range(300):m.receive(i,1000+i);time.sleep(.00001)
  p=threading.Thread(target=producer);p.start()
  while p.is_alive():
   taken=m.take()
   if taken:life.finish(taken[0],'COMPLETED','UNIT_READY')
   if m.received>100:f.trip('INJECTED_ASYNC_FAULT')
   time.sleep(.00001)
  p.join();m.close('END')
  state=life.snapshot();self.assertEqual(state['received'],300);self.assertEqual(state['open'],0);self.assertEqual(len(state['terminals']),300);self.assertEqual(len({r['input_id'] for r in state['terminals']}),300)
 def test_generated_source_has_terminal_and_no_control_publish(self):
  code=main_source();compile(code,'E5_MOCK_REVIEW','exec')
  self.assertIn('INPUT_TERMINALS.jsonl',code);self.assertIn('except SessionStopped',code);self.assertIn('session.require_live()',code)
  self.assertNotIn('rospy.Publisher',code);self.assertNotIn('PositionCommand',code)
 def test_cpp_standalone_session_guard(self):
  text=r'''#include "proposal_session_gate.hpp"
#include <cassert>
int main(){
 int physicalActive=7,physicalBackup=9;qm::ReadonlySessionGate a;
 assert(!a.check(0,true,false,false).allowed&&!a.latched());
 assert(a.check(1,true,true,false).allowed);
 assert(!a.check(6,true,true,false).allowed&&a.latched());
 assert(!a.check(1,true,true,false).allowed);
 qm::ReadonlySessionGate b;assert(!b.check(1,false,true,false).allowed);
 qm::ReadonlySessionGate c;assert(!c.check(1,true,false,false).allowed);
 qm::ReadonlySessionGate d;assert(!d.check(1,true,true,true).allowed);
 assert(physicalActive==7&&physicalBackup==9);
}'''
  p=OUT/'runtime/session_guard_unit.cpp';p.write_text(text);exe=OUT/'runtime/session_guard_unit'
  r=subprocess.run(['g++','-std=c++17','-pthread','-I'+str(ROOT/'query_minco/inference'),str(p),'-o',str(exe)],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(subprocess.run([str(exe)]).returncode,0)
def replay():
 old=OUT.parent/'f3_execution008';rows=[json.loads(x) for x in (old/'shadow_inputs001/c1/INPUT_LEDGER.jsonl').read_text().splitlines()]
 # Known old input identities only; old missing1 is NOT fabricated or repaired.
 entries=[]
 for r in rows:
  if r['kind']=='QUEUE_REPLACED':entries.append((r['received_mono'],r['source_stamp'],r))
  elif r['kind'] in ('REJECT','NETWORK_INPUT'):entries.append((r['received_mono'],r['source_stamp'],r))
 entries.sort();fault=json.loads((old/'CONTROL_FAILURE_CAUSAL_REVIEW.json').read_text())['first_event42_ros']
 events=__import__('numpy').genfromtxt(old/'shadow_inputs001/EVENT.csv',delimiter=',',names=True)
 fault_mono=float(events[events['code']==42][0]['mono'])
 l=InputLifecycle();f=FaultLatch();m=LatestInputMailbox(l,f);model_calls=0;admissions=[]
 for recv,stamp,r in entries:
  now=int(recv*1e9)
  if recv>=fault_mono:f.trip('HISTORICAL_FAULT_INJECTED_CPU_REPLAY',now)
  t=m.receive(r,int(round(stamp*1e9)),now)
  if f.stopped:continue
  taken=m.take(now)
  if not taken:continue
  if r['kind']=='NETWORK_INPUT':
   f.require_live();admissions.append(dict(receive_mono=recv,fault_latched=f.stopped));model_calls+=1;l.finish(t,'COMPLETED','HISTORICAL_PROCESSING_ONLY_NOT_NEW_PROPOSAL',now)
  else:l.finish(t,'DROPPED' if r['kind']=='QUEUE_REPLACED' else 'REJECTED',r.get('reason','QUEUE_REPLACED'),now)
 m.close('REPLAY_END',int((entries[-1][0]+1)*1e9));snapshot=l.snapshot()
 assert snapshot['received']==len(entries)==472 and not snapshot['open'] and len(snapshot['terminals'])==472
 (OUT/'analysis/CPU_REPLAY_TERMINALS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in snapshot['terminals']))
 return dict(mode='COUNTERFACTUAL_CPU_INTERFACE_REPLAY_NOT_NATIVE',known_old_received_identities=472,unique_terminal_count=472,open=0,model_stub_admissions=model_calls,new_admissions_after_fault_latch=sum(a['fault_latched'] for a in admissions),historical_missing_processed_input=1,historical_gap_not_backfilled=True,old_native_qualification_unchanged=True)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 report=dict(cpu_tests=result.testsRun,success=result.wasSuccessful(),ROS_runs=0,Simulator_patch_applied=False,Simulator_patch_compiled=False,replay=replay())
 (OUT/'CPU_INTERFACE_RESULTS.json').write_text(json.dumps(report,indent=2));assert report['success']
 print(json.dumps(report,indent=2))
