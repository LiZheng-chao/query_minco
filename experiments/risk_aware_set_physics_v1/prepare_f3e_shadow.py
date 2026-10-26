"""Prepare one exclusive readonly Shadow; no simulator start in this script.
Old candidate receipts remain expired. New non-Query reference certification
is recomputed from immutable coefficients against this new run's static scene,
not an extension, field edit, or reuse of old safety certificates.
"""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import PreparedReference,StopEntry
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from experiments.risk_aware_set_physics_v1.export_native_reference import export,thaw
def main():
 root=Path('/home/lizheng/query_minco_mainline');old=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution005';out=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution006'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for file in ('PRIVATE_STARTUP_BINDING.json','E_BINDING.json'):
  for path,expected in json.loads((old/file).read_text()).items():assert sha(Path(path))==expected,(path,'BINDING_CHANGED')
 assert json.loads((out/'offline/END_TO_END_CPU_BENCHMARK.json').read_text())['passed']==24
 assert json.loads((out/'offline/SUMMARY.json').read_text())['complete_cpu_pass']==48
 (out/'scene').mkdir()
 for name in ('empty_room.ply','SCENE_VALIDATION.json'):shutil.copyfile(old/'scene'/name,out/'scene'/name)
 (out/'scene/sensor.yaml').write_text((old/'scene/sensor.yaml').read_text().replace('f3_execution005','f3_execution006'))
 for name in ('native_reference_shadow','private_sim_exec.sh','controlled_startup_shadow.launch'):
  target=out/'runtime'/name
  if name=='native_reference_shadow':shutil.copy2(old/'runtime'/name,target)
  else:
   target.write_text((old/'runtime'/name).read_text().replace('f3_execution005','f3_execution006'))
   target.chmod(0o755 if name.endswith('.sh') else 0o644)
 host=(old/'runtime/E_INPUT_HOST.py').read_text().replace('f3_execution005','f3_execution006').replace('query_minco.inference.c1_proposal_shadow','query_minco.inference.c1_proposal_shadow_v2')
 (out/'runtime/E_INPUT_HOST.py').write_text(host)
 prereg=dict(run_id='f3_execution006/shadow_inputs001',kind='ONCE_ONLY_REAL_C1_READONLY_PROPOSALS',
  controlled_startup='Previously authorized private gate, same binary/Controller/dynamics',
  scene='EXTERNAL_SUPERVISOR same independent verified 3m empty room',goal_W=[3,0,2],local_goal_m=2.4,
  CPU_entry={'48_complete_offline':True,'24_bounded_CPU_checks_pass':True},
  maximum_host_s=70,per_child_timeout_s=85,maximum_C1_frames=10,maximum_C1_s=48,
  limits=dict(speed=1,acceleration=1.5,command_jerk=10,soft_brake_jerk=.35,tracking_PVA=[.02,.05,.1]),
  time_contract=dict(activation_lead_s=.06,ready_guard_s=.02,post_generation_budget_s=.04,source_age_s=.25,current_odom_age_s=.05,activation_late_s=.01,control_interval_s=.04),
  deterministic_policy='Generate all six; q5 first then q4/0/1/2/3 if >=25ms remains. Fixed connection1s sourcephase.7 rate.5 retain.1s. First complete proof only; remaining NOT_CHECKED, never unsafe fallback.',
  proof_scope='Conditional float64 Bernstein full-body containment, EXTERNAL_SUPERVISOR; native depth safety still UNKNOWN.',
  evidence='Inherited new run cell remaining lifetime only. No old candidate receipt renewal. Async owned writer process.',
  resources=dict(windows_available_kb=1048576,WSL_available_kb=3145728,GPU_free_start_mib=512,GPU_free_runtime_mib=256,no_OOM=True,no_sustained_swap=True),
  sole_physical_publisher='/f3_reference_arbiter',Query_control=False,Query_dynamic_submission=False,training=False,
  real_pass='At least one full bank proposal before unchanged deadline, plus causal actual activation state gate. All original native tracking/time/resource/isolation requirements retained.',
  failure_policy='One run only. Any native control/resource contract failure stops progression. No retry for PASS.')
 (out/'SHADOW_PREREG.json').write_text(json.dumps(prereg,indent=2))
 (out/'shadow_preparation').mkdir()
 with (old/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:base=pickle.load(f)
 now=time.time();mono=time.monotonic();until=now+120.
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],until,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 result={};checked=0
 for key,b in base.items():
  prefix=thaw(b.prefix).validate(MotionLimits());entries=[]
  for row in b.stops:
   stop=thaw(row.polynomial);full=stop if row.phase==0 else concatenate(trim(prefix,0,row.phase),stop)
   full.validate(MotionLimits())
   proof=volume.certify(full.coefficients,full.durations,now,required_until=until)
   assert proof['status']=='CERTIFIED_FREE'
   assert row.phase+stop.duration<120.
   entries.append(StopEntry(row.phase,row.polynomial,json.dumps(proof,sort_keys=True).encode()));checked+=1
  proof=volume.certify(prefix.coefficients,prefix.durations,now,required_until=until)
  assert proof['status']=='CERTIFIED_FREE'
  result[key]=PreparedReference(b.prefix,tuple(entries),json.dumps(proof,sort_keys=True).encode(),until,mono+120.,'EXTERNAL_SUPERVISOR',now,mono)
 export(result,out/'shadow_preparation/NATIVE_REFERENCES.bin')
 with (out/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(result,f,protocol=4)
 (out/'shadow_preparation/NEW_REFERENCE_CERTIFICATION.json').write_text(json.dumps(dict(checked_complete_prefix_stops=checked,issued_ros=now,expiry_ros=until,source_numeric_programs=str(old/'shadow_preparation/TRANSPORT_BUNDLES.pkl'),old_certificates_reused=False,Query_candidate_receipts_renewed=False,remaining_s_at_finish=until-time.time()),indent=2))
 paths=[out/'runtime'/n for n in ('native_reference_shadow','private_sim_exec.sh','controlled_startup_shadow.launch','E_INPUT_HOST.py')]
 paths += [root/'query_minco/inference/candidate_feasibility.py',root/'query_minco/inference/c1_proposal_shadow_v2.py',out/'shadow_preparation/NATIVE_REFERENCES.bin',out/'shadow_preparation/TRANSPORT_BUNDLES.pkl',out/'SHADOW_PREREG.json',out/'scene/sensor.yaml',out/'scene/empty_room.ply']
 (out/'REAL_SHADOW_BINDING.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2))
 print('PREPARED',checked,'remaining',until-time.time(),flush=True)
if __name__=='__main__':main()
