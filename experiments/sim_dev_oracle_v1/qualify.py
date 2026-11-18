"""Derive NONPHYSICAL interface preflight only from actual raw test results."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 binding=json.loads((O/'SIM_SOURCE_BINDING.json').read_text())['sha256']
 for name,h in binding.items():
  if not Path(name).is_file() or sha(name)!=h:raise ValueError('QUALIFICATION_SOURCE_DRIFT '+name)
 controller=json.loads((O/'controller_probe_v3/RESULT.json').read_text());cleanup=json.loads((O/'controller_probe_v3/COMMANDS.json').read_text());runtime=json.loads((O/'NONDRIVING_RUNTIME_REPLAY.json').read_text());synthetic=json.loads((O/'SYNTHETIC_FOLLOW_RESULT.json').read_text());tests=json.loads((O/'CHECK_COMMANDS.json').read_text());native=json.loads((O/'SIM_REPLAY.json').read_text())
 assert controller['status']=='PASS_NONDRIVING_CONTROLLER_ONLY' and controller['distinct_consumed_plan_ids']==2 and controller['switch_proven']
 assert all(x['exit_code']==0 for x in cleanup) and runtime['guardian_process_exit']==0 and synthetic['guardian_exit']==0
 assert runtime['steady_valid_fraction']>=.99 and runtime['steady_longest_invalid_s']<=.1 and not runtime['evidence_overflow']
 assert synthetic['reason']=='SYNTHETIC_TARGET_REACHED' and synthetic['steady_valid_fraction']>=.99 and synthetic['progress_m']>=.8 and synthetic['distinct_simulated_command_ids']>=2
 assert native['runtime_diagnostic_candidates']==69 and native['runtime_diagnostic_prefix_accepts']==69 and all(x['exit_code']==0 for x in tests)
 evidence=[O/n for n in ['controller_probe_v3/RESULT.json','controller_probe_v3/CONSUMPTION.jsonl','controller_probe_v3/COMMANDS.json','NONDRIVING_RUNTIME_REPLAY.json','SYNTHETIC_FOLLOW_RESULT.json','CHECK_COMMANDS.json','SIM_REPLAY.json']]
 result=dict(status='PASS',mode='SIM_DEV_ORACLE',scope='NONPHYSICAL_INTERFACE_ONLY_REQUIRES_NEW_S_AND_C7_BEFORE_PLANT_ARM',binding=binding,controller_consumer_test=True,stop_cleanup_test=True,stop_cleanup_scope='NONDRIVING_NODELET_MASTER_AND_GUARDIAN_CLEANUP_ONLY_PHYSICAL_STOP_NOT_RUN',strict_depth_qualified=False,physical_qualified=False,Plant_steps=0,evidence=[dict(path=str(p),sha256=sha(p)) for p in evidence])
 p=O/'runtime_v2/SIM_DEV_RUNTIME_QUALIFICATION.json'
 with p.open('x') as f:json.dump(result,f,indent=2)
 print('SIM_DEV_NONPHYSICAL_INTERFACE_PASS_NO_ROS_NO_ARM; NEW_S_C7_PHYSICAL_PENDING')
if __name__=='__main__':main()
