"""Three targeted nonphysical checks; never imports ROS or starts Plant."""
import ast
import copy
import json
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
from collections import Counter
from experiments.sim_dev_oracle_v1.bridge import Engine
from experiments.sim_dev_oracle_v1.guardian import Guardian
from experiments.sim_dev_oracle_v1.prepare_runtime import state_pair_adapter,arm_handshake_adapter
from query_minco.integration.yopo_execution_v4.trajectory import Plan
R=Path(__file__).resolve().parents[2]
O=R/'research_outputs/sim_dev_no_proof_fix_20261118_v1'
CASE=R/'research_outputs/sim_dev_state_pair_fix_20261118_v1/runtime_bundle/runs/SIM_08M_20261118_171939_7375/none'
spec=importlib.util.spec_from_file_location('real_engine_host',Path(__file__).with_name('test_state_sync.py'));host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
class Pulse:
 def __init__(self):self.count=0
 def wait(self,delay):self.count+=1;return self.count>1
class Tests(unittest.TestCase):
 def ready(self):
  b,c=host.bridge();host.feed(b,'odom',100.);host.feed(b,'imu',100.);host.request(b)
  return b,c
 def test_no_proof_or_old_proof_cannot_arm(self):
  b,c=self.ready();e=b.engine;sent=[];e.request_arm(e.nonce);self.assertIsNone(e.with_arm_ready(100.,sent.append));self.assertEqual(sent,[])
  host.commit(b);c[0]=100.01;e.arm_request_stamp=None;e.request_arm(e.nonce)
  self.assertIsNone(e.with_arm_ready(c[0],sent.append));self.assertEqual(sent,[])
  with self.assertRaisesRegex(ValueError,'SIM_ARM_NONCE'):e.request_arm('wrong')
 def test_ready_allows_once_and_expiry_invokes_real_stop_branch(self):
  b,c=self.ready();e=b.engine;e.request_arm(e.nonce);proof=host.commit(b);sent=[]
  self.assertIsNotNone(e.with_arm_ready(100.,sent.append));self.assertEqual(len(sent),1);self.assertIsNone(e.with_arm_ready(100.,sent.append))
  c[0]=proof['expires']+.001
  # Execute the ACTUAL generated Supervisor raw callback (no ROS runtime).
  src=arm_handshake_adapter(state_pair_adapter((R/'ros/g4cp7b_fix2_v1/supervisor.py').read_text()))
  fn=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name=='raw')
  class Journal:
   @contextmanager
   def callback(self):yield True
  stops=[]
  msg=NS(header=NS(stamp=NS(to_sec=lambda:c[0],to_nsec=lambda:int(c[0]*1e9)),frame_id='world'),aux=NS(current_yaw=0.,kf_correction=0.,angle_corrections=[0.,0.],enable_motors=True,use_external_yaw=False),force=NS(x=0.,y=0.,z=9.6138),orientation=NS(x=0.,y=0.,z=0.,w=1.),kR=[1.,1.,1.],kOm=[.1,.1,.1])
  ns=dict(journal=Journal(),mode='RUNNING',accept=lambda *a:True,r=NS(mode='ARMED',forward_allowed=lambda now:True),forest_bridge=b,rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:c[0]))),abort=stops.append)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<real SIM raw callback>','exec'),ns);ns['raw'](msg)
  self.assertEqual(stops,['T2_NO_CURRENT_PROOF'])
 @unittest.skipUnless((CASE/'supervisor_data/EVENTS.jsonl').exists(),'Recorded input unavailable: replay UNKNOWN')
 def test_original_timestamp_replay(self):
  rows=[json.loads(x) for x in (CASE/'supervisor_data/EVENTS.jsonl').read_text().splitlines()];fault=next(e for e in rows if e.get('kind')=='fault');cut=rows[:rows.index(fault)];aud=[e['payload'] for e in cut if e.get('kind')=='t2_planner_audit'];counts=Counter(x['event'] for x in aud);plans={x['plan_id']:Plan(x['plan']) for x in aud if x['event']=='CANDIDATE_GENERATED'};old=next(x for x in reversed(aud) if x['event']=='ORACLE_PREFIX_COMMITTED');latest_state=next(x for x in reversed(aud) if x['event']=='STATE_UPDATED');new=next(x for x in reversed(aud) if x['event']=='CANDIDATE_GENERATED')
  latest=lambda kind:next(e for e in reversed(cut) if e.get('kind')==kind and e.get('accepted'))
  arm=next(e for e in cut if e.get('mode')=='ARMED');b,c=host.bridge(fault['ros_now']);e=b.engine;e.nonce=old['oracle']['nonce'];e.state(latest_state['state'],latest_state['odom_stamp'],latest_state['imu_stamp'],latest_state['state_pair']);e.last_stamp=latest('depth')['stamp'];e.set_goal([3.25,-.25,2.],latest('request')['stamp']);e.committed=dict(plan=plans[old['plan_id']],proof=old['oracle'],planner={},state_pair=old['state_pair'])
  self.assertIsNone(e.command(c[0]));self.assertGreater(c[0],old['oracle']['expires'])
  # Handoff request uses a REAL recorded ARMED observation clock; not an invented delivery timestamp.
  e.arm_request_stamp=arm['ros_now'];e.arm_sent=False;sent=[];self.assertIsNone(e.with_arm_ready(c[0],sent.append))
  monitor=Guardian(str(CASE/'native_grid'),e.nonce)
  class LocalGuardian:
   def send(self,data):
    try:self.result=dict(ok=True,proof=monitor.check(**data))
    except ValueError as ex:self.result=dict(ok=False,reason=str(ex))
   def poll(self,timeout):return True
   def recv(self):return self.result
  e.guardian_ready=True;e.stop=Pulse();e.conn=LocalGuardian();e.revision=sum(x.get('kind')=='depth' and x.get('accepted',False) for x in cut);e.proposed=dict(plan=plans[new['plan_id']],planner=new['planner']);e.guard_loop()
  self.assertEqual(e.committed['plan'].id,new['plan_id']);self.assertIsNotNone(e.with_arm_ready(c[0],sent.append));self.assertEqual(len(sent),1)
  result=dict(scope='REAL_TIMESTAMPS_COUNTERFACTUAL_NO_ROS_NO_PLANT_NO_FLIGHT_QUALIFICATION',event_counts=dict(counts),last_original_proof=old['oracle'],last_state_pair=latest_state['state_pair'],latest_depth_stamp=latest('depth')['stamp'],latest_goal_stamp_inferred_from_actual_request_path=latest('request')['stamp'],first_supervisor_armed_ros=arm['ros_now'],original_fault_ros=fault['ros_now'],expiry_before_fault_s=fault['ros_now']-old['oracle']['expires'],original_command_none=True,original_handoff_blocked=True,fresh_guardian_check_passed=True,new_candidate_plan_id=new['plan_id'],counterfactual_proof=e.committed['proof'],counterfactual_permit_callbacks=len(sent),limitations=['Exact Guardian rejected-request timestamps/payloads were not logged originally.','Fresh proof is recomputed offline by the unchanged Guardian on the original Depth-selected candidate and scene; it is not original flight evidence.','No new flight/scheduling/Controller consumption success is claimed.'])
  (O/'replay.raw.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':unittest.main()
