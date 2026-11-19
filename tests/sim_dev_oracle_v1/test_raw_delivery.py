"""Three targeted raw-delivery tests; replayed C++ records, no ROS/Plant."""
import ast,json,copy,unittest,importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
from experiments.sim_dev_oracle_v1.raw_delivery import LatestRaw
from experiments.sim_dev_oracle_v1.prepare_runtime import raw_latest_adapter,goal_queue_adapter,arm_handshake_adapter,state_pair_adapter
from query_minco.integration.yopo_execution_v4.readiness import Readiness
from query_minco.integration.yopo_execution_v4.trajectory import Plan
R=Path(__file__).resolve().parents[2]
CASE=R/'research_outputs/sim_dev_proof_continuity_fix_20261118_v1/runtime_bundle/runs/UNQUALIFIED_SIM_DEV_DEBUG_08M_20261118_210425_7375/none'
spec=importlib.util.spec_from_file_location('real_engine_host',Path(__file__).with_name('test_state_sync.py'));host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
def message(ns,record=None):
 d=record or dict(force=[0,0,9.6138],orientation=[0,0,0,1],kr=[1,1,1],kom=[.1,.1,.1],aux=[0,0,0,0,1,0,0])
 return NS(header=NS(stamp=NS(to_sec=lambda:ns//1000000000+(ns%1000000000)/1e9,to_nsec=lambda:ns),frame_id='/quadrotor'),force=NS(**dict(zip('xyz',d['force']))),orientation=NS(**dict(zip('xyzw',d['orientation']))),kR=d['kr'],kOm=d['kom'],aux=NS(current_yaw=d['aux'][0],kf_correction=d['aux'][1],angle_corrections=d['aux'][2:4],enable_motors=bool(d['aux'][4]),use_external_yaw=bool(d['aux'][5])),_connection_header={'callerid':'/f0_audit_position_controller'})
def runtime(b,c):
 source=raw_latest_adapter(goal_queue_adapter(arm_handshake_adapter(state_pair_adapter((R/'ros/g4cp7b_fix2_v1/supervisor.py').read_text()))))
 compile(source,'<actual generated supervisor>','exec')
 main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
 raw=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='raw')
 drain=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr=='service' for stmt in n.body))
 class Journal:
  def __init__(self):self.events=[]
  @contextmanager
  def callback(self):yield True
  def append(self,event):self.events.append(event)
 journal=Journal();stops=[];published=[];readiness=Readiness(True,True,'equilibrium_hover');readiness.mode='ARMED';readiness.counts={k:5 for k in ['odom','imu','plant','request','depth','raw']}
 ns=dict(journal=journal,mode='RUNNING',forest_bridge=b,rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:c[0]))),r=readiness,abort=stops.append,final=NS(publish=published.append),log=lambda kind,stamp,data:journal.append(dict(kind=kind,stamp=stamp,payload=data)))
 ns['accept']=lambda kind,stamp,data:readiness.observe(kind,stamp,c[0],data)
 exec(compile(ast.Module(body=[raw],type_ignores=[]),'<actual raw callback>','exec'),ns)
 def seed():
  for k in ['odom','imu','depth','request']:readiness.samples[k]=(c[0],c[0],{})
  readiness.samples['request']=(c[0],c[0],dict(pva=[[.25,-.25,2.],[0,0,0],[0,0,0]]))
  readiness.samples['plant']=(c[0],c[0],dict(phase='ARMED',integration_steps=10))
 def service(slot):
  seed();ns['raw_slot']=slot
  exec(compile(ast.Module(body=[drain],type_ignores=[]),'<actual generated drain/STOP>','exec'),ns)
 return service,stops,published,journal,source
class Tests(unittest.TestCase):
 @unittest.skipUnless((CASE/'controller/CONSUMPTION.jsonl').exists(),'Recorded C++ input unavailable')
 def test_recorded_positioncommand_controller_raw_final_latest_integration(self):
  events=[json.loads(x) for x in (CASE/'supervisor_data/EVENTS.jsonl').read_text().splitlines()];logs=[json.loads(x) for x in (CASE/'controller/CONSUMPTION.jsonl').read_text().splitlines()];aud=[e['payload'] for e in events if e.get('kind')=='t2_planner_audit'];commands={e['payload']['command_key']['trajectory_id']:e['payload'] for e in events if e.get('kind')=='controller_command' and e.get('payload',{}).get('plan_id')};consumed={e['trajectory_id']:e for e in logs if e.get('kind')=='COMMAND_CONSUMED'};plans={e['plan_id']:Plan(e['plan']) for e in aud if e['event']=='CANDIDATE_GENERATED'};commits=[e for e in aud if e['event']=='ORACLE_PREFIX_COMMITTED'];states=[e for e in aud if e['event']=='STATE_UPDATED'];goals=[e for e in aud if e['event']=='GOAL_RECEIVED'];outputs=[]
  for out in logs:
   if out.get('kind')!='CONTROL_OUTPUT' or out['trajectory_id'] not in commands:continue
   d=commands[out['trajectory_id']];valid=[e for e in commits if e['plan_id']==d['plan_id'] and e['monotonic']<=out['monotonic']]
   if not valid:continue
   commit=valid[-1];t=out['output_ns']//1000000000+(out['output_ns']%1000000000)/1e9
   if not commit['oracle']['issued']<=t+.002<=commit['oracle']['expires']:continue
   state=max((e for e in states if e['monotonic']<=out['monotonic']),key=lambda e:e['monotonic']);goal=max((e for e in goals if e['monotonic']<=out['monotonic']),key=lambda e:e['monotonic']);outputs.append((out,d,commit,state,goal))
  self.assertGreaterEqual(len(outputs),10)
  b,c=host.bridge();service,stops,published,journal,source=runtime(b,c);slot=LatestRaw(outputs[0][2]['oracle']['nonce']);plan_ids=set()
  for index,(out,d,commit,state,goal) in enumerate(outputs):
   self.assertEqual(consumed[out['trajectory_id']]['input_ns'],d['command_key']['input_ns']);self.assertEqual(consumed[out['trajectory_id']]['pva'],[x for row in d['pva'] for x in row]);m=message(out['output_ns'],out);t=m.header.stamp.to_sec();slot.offer(m,t+.001,out['monotonic']+.001)
   if index%2==0 and index+1<len(outputs):continue
   c[0]=t+.002;b.engine.nonce=commit['oracle']['nonce'];b.engine.state(state['state'],state['odom_stamp'],state['imu_stamp'],state['state_pair']);b.engine.set_goal(goal['goal'],goal['source_stamp']);b.engine.last_stamp=commit['oracle']['source_stamp'];b.engine.committed=dict(plan=plans[d['plan_id']],proof=commit['oracle'],planner={},state_pair=commit['state_pair']);before=len(published);service(slot)
   self.assertEqual(stops,[]);self.assertEqual(len(published),before+1);self.assertIs(published[-1],m);self.assertEqual(m.header.stamp.to_nsec(),out['output_ns']);plan_ids.add(d['plan_id'])
  self.assertGreaterEqual(len(plan_ids),2);self.assertTrue(any(any('superseded_by_receive_sequence' in rec for rec in e.get('receipts',[])) for e in journal.events));self.assertIn('receive_raw if forest_bridge else raw',source)
 def test_old_stamp_or_service_backlog_latches_stop(self):
  for at_receipt in [True,False]:
   with self.subTest(at_receipt=at_receipt):
    b,c=host.bridge(100.201);service,stops,published,journal,_=runtime(b,c);slot=LatestRaw('SYNTHETIC_NONPHYSICAL');m=message(100000000000);slot.offer(m,100.201 if at_receipt else 100.001,1.)
    if at_receipt:slot.offer(message(100201000000),100.201,1.001)
    service(slot);self.assertEqual(stops,['raw_STALE']);self.assertEqual(published,[]);self.assertTrue(journal.events[-1]['error'])
 def test_valid_raw_cannot_forward_without_current_proof(self):
  for expired in [False,True]:
   with self.subTest(expired=expired):
    b,c=host.bridge(100.);host.feed(b,'odom',100.);host.feed(b,'imu',100.);host.request(b)
    if expired:host.commit(b)
    c[0]=100.101;host.feed(b,'odom',c[0]);host.feed(b,'imu',c[0]);host.request(b);b.engine.last_stamp=c[0]
    service,stops,published,journal,_=runtime(b,c);slot=LatestRaw('SYNTHETIC_NONPHYSICAL');slot.offer(message(100100000000),100.101,1.);service(slot);self.assertEqual(published,[]);self.assertEqual(stops,['T2_NO_CURRENT_PROOF'])
if __name__=='__main__':unittest.main()
