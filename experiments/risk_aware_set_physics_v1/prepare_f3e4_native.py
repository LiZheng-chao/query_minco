"""E4 independent warmup contract; exact controller/dynamics and readonly core."""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,quintic,concatenate,trim
from query_minco.inference.execution_clock_v3 import cruise_segment
from query_minco.inference.committed_reference import PreparedReference,StopEntry,prepare_reference
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.proposal_worker import curve_bytes
from query_minco.inference.dynamic_proposal import CALIBRATION
from experiments.risk_aware_set_physics_v1.export_native_reference import export,thaw
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008'
def main():
 old=OUT.parent/'f3_execution007';base=OUT.parent/'f3_execution005';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert json.loads((OUT/'CPU_INPUT_ACCEPTANCE.json').read_text())['passed']
 assert json.loads((OUT/'CONTROL_RISK_REVIEW_BEFORE_RUN.json').read_text())['status']=='RISK_LOCATED_UNIQUE_ROOT_CAUSE_UNPROVEN'
 for path,h in json.loads((base/'PRIVATE_STARTUP_BINDING.json').read_text()).items():assert sha(Path(path))==h,path
 (OUT/'scene').mkdir()
 for name in ('empty_room.ply','SCENE_VALIDATION.json'):shutil.copyfile(base/'scene'/name,OUT/'scene'/name)
 (OUT/'scene/sensor.yaml').write_text((base/'scene/sensor.yaml').read_text().replace('f3_execution005','f3_execution008'))
 for name in ('private_sim_exec.sh','controlled_startup_shadow.launch'):
  p=OUT/'runtime'/name;p.write_text((base/'runtime'/name).read_text().replace('f3_execution005','f3_execution008'));p.chmod(.0 and 0o644 or (0o755 if name.endswith('.sh') else 0o644))
 shutil.copy2(old/'runtime/libproposal_numerics.so',OUT/'runtime/libproposal_numerics.so')
 host=(old/'runtime/E_INPUT_HOST.py').read_text().replace('f3_execution007','f3_execution008').replace('native_reference_dynamic','native_reference_dynamic_e4').replace('c1_proposal_shadow_v3','c1_proposal_shadow_e4')
 host=host.replace("'query_minco.inference.shadow_camera_low_speed','--host'","'query_minco.inference.shadow_camera_e4','--query-hz','30','--host'")
 (OUT/'runtime/E_INPUT_HOST.py').write_text(host)
 prereg=dict(run='f3_execution008/shadow_inputs001',mode='F3_E4_TRUE_SOURCE_READONLY_SINGLE_TRIAL',Query_control=False,
 model='frozen C1 step2880',global_goal=[3,0,2],local_goal_m=2.4,initial_position=[-1.6,0,2],
 independent_nonQuery_warmup=dict(label='WARMUP_CRUISE_EXTENDED_V1_NOT_OLD_REFERENCE',retreat_target_x=-2.24,retreat_duration_s=5.8,acceleration_duration_s=3.,cruise_start_x=-1.04,cruise_duration_s=1.3,cruise_velocity=[.8,0,0],instant_handover_x=0.,terminal_duration_s=7.5,nominal_total_s=17.6,full_stop_phase_period_s=.02,full_nominal_phases=881,reason='Explicitly extend physically controlled warmup/cruise for20 opportunities; keep global/local goal and velocity/attitude/limits. All new nonQuery trajectory phases require fresh complete brake and body proof.'),
 input=dict(native_config_fps=33,query_target_hz=30,exact_integer_rays=True,shape=[2,128,160],camera_K=[[100,0,79.5],[0,100,63.5],[0,0,1]],source_acceleration='trusted100Hz previous Odom velocity difference, previous stamp strictly older; no future sample or forced zero',source_age_s=.25,current_state_age_s=.05,speed_radius=.03,source_acc_max=.05,tilt_deg=.5,goal_slope=.175),
 time=dict(activation_lead=.06,ready_guard=.02,prepare_budget=.04,activation_lateness=.01,command_gap=.04),
 limits=dict(v=1,a=1.5,j=10,soft_brake_j=.35,tracking=[.02,.05,.1]),
 opportunities=dict(minimum_valid_decisions=20,max_valid_source_gap_s=.1,minimum_window_s=.5,max_network_frames=40,max_C1_wall_s=48,predicted_supported_window_s=1.3,predicted_raw_30Hz_opportunities=39,minimum_end_to_end_fraction=.95),
 evidence='Every synced input/dropped frame/reject/actual12D+digest/source previous stamp+velocity saved. Receiver independently binds depth-mask hash, camera, source history,12D and all proofs. No old proposal reuse.',
 control_risk='Fixed-step/wall timestamp mechanism remains unproved unique cause. Dedicated worker ROS queue only, no bottom patch; strict original control gates and precertified abort stop.',
 resources=dict(Windows_min_kb=1048576,WSL_min_kb=3145728,GPU_start_free_mib=512,GPU_runtime_free_mib=256,no_OOM=True,no_sustained_swap=True),
 duration=dict(host_max_s=70,child_timeout_s=85,outer_max_s=100),
 failure='Exactly one native run; stop progression on control/resource contract failure; no retries to PASS. Query never changes physical active/pending/backup.')
 (OUT/'SHADOW_PREREG.json').write_text(json.dumps(prereg,indent=2))
 with (base/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:previous=pickle.load(f)
 # Same prescribed initial state; explicitly new independent nonQuery warmup.
 head=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float);back=head.copy();back[0,0]=-2.24
 top=back.copy();top[0,0]=-1.04;top[1,0]=.8
 pre=concatenate(quintic(head,back,5.8),quintic(back,top,3.))
 pre=concatenate(pre,cruise_segment(top,1.3))
 mid=np.array([[0,0,2],[.8,0,0],[0,0,0]],float);goal=np.array([[3,0,2],[0,0,0],[0,0,0]],float)
 nominal=concatenate(pre,quintic(mid,goal,7.5)).validate(MotionLimits())
 assert abs(nominal.duration-17.6)<1e-9
 # CPU preparation before release; trusted scene must later match live exact map.
 now=time.time();mono=time.monotonic();until=now+120
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],until,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 start=time.perf_counter();newnominal=prepare_reference(nominal,volume,now,mono);prepare_ms=(time.perf_counter()-start)*1000
 result={1:newnominal};checked=0
 for bid,b in previous.items():
  if bid==1:continue
  prefix=thaw(b.prefix).validate(MotionLimits());entries=[]
  for row in b.stops:
   stop=thaw(row.polynomial);full=stop if row.phase==0 else concatenate(trim(prefix,0,row.phase),stop)
   full.validate(MotionLimits());proof=volume.certify(full.coefficients,full.durations,now,required_until=until)
   assert proof['status']=='CERTIFIED_FREE'
   entries.append(StopEntry(row.phase,row.polynomial,json.dumps(proof,sort_keys=True).encode()));checked+=1
  proof=volume.certify(prefix.coefficients,prefix.durations,now,required_until=until);assert proof['status']=='CERTIFIED_FREE'
  result[bid]=PreparedReference(b.prefix,tuple(entries),json.dumps(proof,sort_keys=True).encode(),until,mono+120,'EXTERNAL_SUPERVISOR',now,mono)
 result=dict(sorted(result.items()));assert len(newnominal.stops)==881
 (OUT/'shadow_preparation').mkdir()
 with (OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(result,f,protocol=4)
 export(result,OUT/'shadow_preparation/NATIVE_REFERENCES.bin')
 (OUT/'shadow_preparation/WARMUP_CPU_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,nominal_extrema=nominal.extrema(),nominal_phases=len(newnominal.stops),unchanged_other_phases=checked,new_nominal_prepare_ms=prepare_ms,expiry_ros=until,expiry_mono=mono+120,minimum_lifetime_remaining_s=until-time.time(),certificate_scope='EXTERNAL_SUPERVISOR',Controller_changed=False,physics_changed=False),indent=2))
 assert until-time.time()>65,'INSUFFICIENT_FRESH_REFERENCE_LIFETIME'
 scene=json.loads((OUT/'scene/SCENE_VALIDATION.json').read_text())
 config=dict(mode='READONLY_ONLY',owner_epoch='f3_execution008_reference_owner_v1',geometry=scene['quantised_xyz_sha256'],calibration=CALIBRATION,parent_digest=hashlib.sha256(curve_bytes(nominal)).hexdigest(),worker_pid_file=str(OUT/'WORKER_PID.txt'),cell_expiry_ros=until,cell_expiry_mono=mono+120)
 (OUT/'DYNAMIC_CONFIG.json').write_text(json.dumps(config,indent=2))
 paths=[OUT/'runtime'/n for n in ('native_reference_dynamic_e4','native_reference_dynamic_e4.cpp','libproposal_numerics.so','E_INPUT_HOST.py','controlled_startup_shadow.launch','private_sim_exec.sh')]
 paths+=[ROOT/'query_minco/inference'/n for n in ('c1_proposal_shadow_e4.py','shadow_camera_e4.py','neural_input_binding_e4.hpp','c1_proposal_shadow_v4.py','source_state_binding.hpp','proposal_worker.py')]
 paths+=[OUT/'SHADOW_PREREG.json',OUT/'DYNAMIC_CONFIG.json',OUT/'shadow_preparation/NATIVE_REFERENCES.bin',OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl',OUT/'scene/sensor.yaml',OUT/'scene/empty_room.ply']
 (OUT/'REAL_SHADOW_BINDING.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2))
 print('CPU warmup PASS, new881 complete stops; lifetime',until-time.time(),flush=True)
if __name__=='__main__':main()
