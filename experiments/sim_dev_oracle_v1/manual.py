"""Explicit researcher-only SIM entry. Check-only never imports ROS or starts motion."""
from pathlib import Path
import sys,argparse,json,hashlib,importlib.util
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';MODE='SIM_DEV_ORACLE'
def qualification(root,binding):
 p=Path(root)/'SIM_DEV_RUNTIME_QUALIFICATION.json'
 if not p.exists():raise PermissionError('SIM_DEV_NONDRIVING_RUNTIME_QUALIFICATION_PENDING')
 d=json.loads(p.read_text())
 if d.get('mode')!=MODE or d.get('binding')!=binding or d.get('status')!='PASS' or d.get('controller_consumer_test') is not True or d.get('stop_cleanup_test') is not True:raise PermissionError('SIM_DEV_RUNTIME_QUALIFICATION_NOT_PROVEN')
 # Qualification must reference real raw test artifacts, not just booleans.
 for row in d.get('evidence',[]):
  path=Path(row['path'])
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise PermissionError('SIM_DEV_QUALIFICATION_DRIFT')
 if len(d.get('evidence',[]))<2:raise PermissionError('SIM_DEV_RAW_RUNTIME_EVIDENCE_REQUIRED')
 return d

def main():
 if Path.cwd().resolve()!=R:raise ValueError('EXPECTED_REPOSITORY_CWD')
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=[MODE],required=True);p.add_argument('--stage',choices=['S','C',MODE],default=MODE);p.add_argument('--check-only',action='store_true');p.add_argument('--allow-physics');p.add_argument('--prior');p.add_argument('--output');a=p.parse_args()
 identity=json.loads((O/'SIM_SOURCE_BINDING.json').read_text());binding=identity['sha256']
 if json.loads((O/'runtime_v2/SOURCE_BINDING.json').read_text())!=binding:raise ValueError('SIM_RUNTIME_BINDING_MISMATCH')
 for name,expected in binding.items():
  if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('SIM_SOURCE_DRIFT '+name)
 if a.check_only:
  if a.stage==MODE:
   try:qualification(O/'runtime_v2',binding)
   except PermissionError as ex:print('PREFLIGHT_SOURCE_PASS_PREREQUISITE_PENDING_NO_ROS_NO_ARM',str(ex));return
  if a.stage==MODE and a.prior:
   from query_minco.integration.birth_execution_v1.evidence import predecessor
   predecessor(a.prior,'C_CAMPAIGN',binding);print('PREFLIGHT_PASS_NO_ROS_NO_ARM SIM_DEV_ORACLE_ONLY');return
  if a.stage=='S':
   from query_minco.integration.birth_execution_v1.evidence import legacy_window
   legacy_window(O/'runtime_v2/LEGACY_B_PHYSICAL_WINDOW.json')
  print('PREFLIGHT_SOURCE_PASS_PREREQUISITE_PENDING_NO_ROS_NO_ARM SIM_NONPHYSICAL_INTERFACE_PASS_NEW_S_C7_PENDING');return
 if a.allow_physics!='RESEARCHER_MANUAL_'+a.stage+'_ONLY':raise PermissionError('RESEARCHER_SEPARATE_PHYSICAL_AUTHORIZATION_REQUIRED')
 if not a.output:raise ValueError('UNIQUE_OUTPUT_REQUIRED')
 if a.stage==MODE:qualification(O/'runtime_v2',binding)
 generated=O/'generated_v2/manual.py';spec=importlib.util.spec_from_file_location('sim_dev_generated_manual',generated);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 # Original owner still enforces same-binding S -> C7, per-episode ROS master,
 # Plant STOP, nonce/PID ownership, Controller identity and full cleanup.
 sys.argv=[str(generated),'--stage',a.stage,'--allow-physics',a.allow_physics,'--output',a.output]+(['--prior',a.prior] if a.prior else [])
 module.main(config_root=O/'runtime_v2',binding_name='SOURCE_BINDING.json',audit_build=R/'research_outputs/f0_controller_consumption_20261118_v1')
if __name__=='__main__':main()
