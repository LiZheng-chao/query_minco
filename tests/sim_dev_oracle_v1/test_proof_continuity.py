"""Targeted proof continuity and executed-output scope. No ROS/Plant."""
from pathlib import Path
import copy,json,unittest,importlib.util
import numpy as np
from experiments.sim_dev_oracle_v1.state_sync import StatePairs,StateInvalid
from experiments.sim_dev_oracle_v1.guardian import Guardian
from experiments.sim_dev_oracle_v1.evidence import assess
from query_minco.integration.yopo_execution_v4.trajectory import build,Plan
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/sim_dev_proof_continuity_fix_20261118_v1'
CASE=R/'research_outputs/sim_dev_goal_arm_cleanup_fix_20261118_v1/runtime_bundle/runs/UNQUALIFIED_SIM_DEV_DEBUG_08M_20261118_181250_7375/none'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
host=load('engine_method_host',Path(__file__).with_name('test_state_sync.py'))
fixture=load('evidence_fixture',Path(__file__).with_name('test_evidence.py'))
class Pulse:
 def __init__(self):self.calls=0
 def wait(self,delay):self.calls+=1;return self.calls>1
class IPC:
 def __init__(self,guardian,before_reply=None):self.guardian=guardian;self.before_reply=before_reply
 def send(self,data):
  try:self.reply=dict(ok=True,proof=self.guardian.check(**data))
  except ValueError as ex:self.reply=dict(ok=False,reason=str(ex))
 def poll(self,timeout):return True
 def recv(self):
  if self.before_reply:self.before_reply()
  return self.reply
class Tests(unittest.TestCase):
 def test_late_closer_imu_retains_original_fresh_pair(self):
  b,c=host.bridge(100.);host.feed(b,'imu',99.99);host.feed(b,'odom',100.);p=b.state({},0,0);before=b.engine.actual
  host.feed(b,'imu',100.);q=b.state({},0,0)
  self.assertEqual(q['status'],'STATE_PAIR_DUPLICATE');self.assertEqual(q['state_pair_id'],p['state_pair_id']);self.assertIs(b.engine.actual,before);self.assertFalse(b.engine.state_pending);self.assertEqual(q['imu_stamp'],99.99)
 def test_retained_pair_does_not_mask_latest_invalid_or_expiry(self):
  b,c=host.bridge();host.feed(b,'imu',99.99);host.feed(b,'odom',100.);b.state({},0,0);host.feed(b,'imu',100.);c[0]=100.201
  with self.assertRaisesRegex(StateInvalid,'STALE'):b.state({},0,0)
  self.assertIsNone(b.engine.actual)
 @unittest.skipUnless((CASE/'supervisor_data/EVENTS.jsonl').exists(),'Recorded flight not available')
 def test_actual_pending_reply_replay_and_raw_output_scope(self):
  rows=[json.loads(x) for x in (CASE/'supervisor_data/EVENTS.jsonl').read_text().splitlines()];aud=[e['payload'] for e in rows if e.get('kind')=='t2_planner_audit'];pending=next(e for e in aud if e['event']=='STATE_PAIR_PENDING' and e['monotonic']>33087.)
  request=max((e for e in aud if e['event']=='ORACLE_PREFIX_REQUESTED' and e['monotonic']<pending['monotonic']),key=lambda e:e['monotonic']);reply=next(e for e in aud if e['event']=='ORACLE_PREFIX_RETURNED' and e.get('issued')==request['issued']);state=max((e for e in aud if e['event']=='STATE_UPDATED' and e['monotonic']<pending['monotonic']),key=lambda e:e['monotonic'])
  old=load('old_pair_impl',O/'source_before/state_sync.py');old_pairs=old.StatePairs();new_pairs=StatePairs()
  for e in sorted((e for e in aud if e['event']=='STATE_SENSOR_RECEIVED' and e['monotonic']<pending['monotonic']),key=lambda e:e['monotonic']):
   for pairer in [old_pairs,new_pairs]:pairer.add(e['kind'],e['data'],e['source_stamp'],e['receipt_ros'])
  for pairer in [old_pairs,new_pairs]:pairer.last_pair=(state['odom_stamp'],state['imu_stamp'])
  self.assertEqual(old_pairs.select(request['issued'])['status'],'STATE_PAIR_PENDING')
  fixed=new_pairs.select(request['issued']);self.assertEqual(fixed['status'],'STATE_PAIR_DUPLICATE');self.assertEqual(fixed['imu_stamp'],state['imu_stamp'])
  plans={e['plan_id']:Plan(e['plan']) for e in aud if e['event']=='CANDIDATE_GENERATED'};b,c=host.bridge(request['issued']);b.pairs=new_pairs;e=b.engine;e.nonce=json.loads((CASE/'RUNTIME_PARAMETERS.json').read_text())['run_nonce'];e.state(state['state'],state['odom_stamp'],state['imu_stamp'],state['state_pair']);e.last_stamp=request['source_stamp'];e.revision=request['revision'];e.set_goal([3.25,-.25,2.],request['issued']);e.proposed=dict(plan=plans[request['plan_id']],planner={});e.guardian_ready=True;e.stop=Pulse();e.conn=IPC(Guardian(str(CASE/'native_grid'),e.nonce),before_reply=lambda:b.state({},0,0));e.guard_loop()
  self.assertIsNotNone(e.committed);self.assertEqual(e.committed['proof']['expires'],reply['expires']);self.assertEqual(e.committed['plan'].id,request['plan_id']);self.assertFalse(e.state_pending)
  refs=[dict(kind='plan',plan=p.record,plan_id=p.id) for p in plans.values()];controller=[json.loads(x) for x in (CASE/'controller/CONSUMPTION.jsonl').read_text().splitlines()];identity=json.loads((CASE/'CONTROLLER_IDENTITY.json').read_text());stop=json.loads((CASE/'STOP_CONFIRMED.json').read_text());start=min(x['monotonic'] for x in rows if x.get('kind')=='plant' and x.get('accepted') and x.get('payload',{}).get('integration_steps',0)>0)
  verdict=assess(rows,start,stop['monotonic'],refs,controller,e.nonce,identity)
  self.assertEqual(verdict['reason'],'TWO_COMPLETE_PLAN_CHAINS_AND_SWITCH_REQUIRED');self.assertGreaterEqual(len(verdict['chains']),3);self.assertEqual(verdict['status'],'FAIL')
  report=dict(scope='RECORDED_CONTEXT_COUNTERFACTUAL_NONPHYSICAL_NOT_FLIGHT_PASS',original_pending=pending,guardian_request=request,guardian_return=reply,original_state=state,old_selection='STATE_PAIR_PENDING',new_selection=fixed,computed_guardian_proof=e.committed['proof'],corrected_evidence_verdict=verdict,historical_result_unchanged=True)
  with (O/'REPLAY.json').open('w') as f:json.dump(report,f,indent=2)
 def test_continuous_new_proofs_execute_past_many_original_expiries(self):
  b,c=host.bridge();e=b.engine;plan=build(np.array(host.ODOM['p']),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.65,-.25,2.]),100.);e.guardian_ready=True;e.conn=IPC(Guardian(host.Empty(),e.nonce));e.revision=1;e.proposed=dict(plan=plan,planner={});sent=[];ids=[]
  for i in range(26):
   c[0]=100.+i*.04;pva=plan.at(c[0]);odom=dict(frame='world',p=pva[0].tolist(),v=pva[1].tolist(),q=[0,0,0,1]);imu=dict(frame='world',a_world=pva[2].tolist());host.feed(b,'odom',c[0],odom);host.feed(b,'imu',c[0],imu);b.state({},0,0);e.last_stamp=c[0];e.set_goal([3.25,-.25,2.],c[0]);e.stop=Pulse();e.guard_loop();self.assertIsNotNone(e.committed);self.assertAlmostEqual(e.committed['proof']['expires']-e.committed['proof']['issued'],.1);ids.append(e.committed['proof']['identity'])
   for delta in [0.,.01,.02,.03]:
    now=c[0]+delta;self.assertIsNotNone(e.with_current(now,sent.append))
  self.assertEqual(len(set(ids)),26);self.assertEqual(len(sent),104)
  # Missing response never creates or extends a proof; fresh state does not rescue expiry.
  proof=copy.deepcopy(e.committed['proof']);c[0]=proof['expires']+.001;pva=plan.at(c[0]);host.feed(b,'odom',c[0],dict(frame='world',p=pva[0].tolist(),v=pva[1].tolist(),q=[0,0,0,1]));host.feed(b,'imu',c[0],dict(frame='world',a_world=pva[2].tolist()));b.state({},0,0);e.last_stamp=c[0];e.set_goal([3.25,-.25,2.],c[0]);called=[];self.assertIsNone(e.with_current(c[0],called.append));self.assertEqual(called,[]);self.assertEqual(e.committed['proof'],proof)
  gate=load('actual_expired_stop',Path(__file__).with_name('test_arm_handoff.py'));gate.Tests('test_ready_allows_once_and_expiry_invokes_real_stop_branch').test_ready_allows_once_and_expiry_invokes_real_stop_branch()
 def expired_extra(self,forward=False):
  f=fixture.fixture();events,refs,logs,identity,nonce=f;extra=copy.deepcopy(logs[-2]);extra.update(index=5,output_index=3,output_ns=102150000000,monotonic=2.15);logs.insert(-1,extra);logs[-1]['records']=5
  if forward:
   event=copy.deepcopy(events[4]);event.update(monotonic=2.151,stamp=102.15);event['payload']['stamp_ns']=extra['output_ns'];events.append(event)
  return f
 def test_unforwarded_expired_raw_output_cannot_count_but_is_not_execution(self):
  result=fixture.check(self.expired_extra());self.assertEqual(result['status'],'PASS',result);self.assertEqual(len(result['chains']),2)
 def test_forwarded_expired_output_still_fails(self):
  result=fixture.check(self.expired_extra(True));self.assertEqual(result['reason'],'EXPIRED_AUTHORIZATION')
 def test_expired_consumption_still_fails_without_output_forward(self):
  f=self.expired_extra();f[2][3]['received_ros']=102.15;f[2][4]['received_ros']=102.15;f[2][5]['received_ros']=102.15;result=fixture.check(f);self.assertEqual(result['reason'],'EXPIRED_AUTHORIZATION')
if __name__=='__main__':unittest.main()
