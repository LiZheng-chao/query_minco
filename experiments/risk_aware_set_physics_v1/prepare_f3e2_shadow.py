"""Prepare ONE native readonly E2/E3 run; exact old control core and trajectory."""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import PreparedReference,StopEntry
from query_minco.inference.trajectory_execution import MotionLimits,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.proposal_worker import curve_bytes
from query_minco.inference.dynamic_proposal import CALIBRATION
from experiments.risk_aware_set_physics_v1.export_native_reference import export,thaw
def main():
 root=Path('/home/lizheng/query_minco_mainline');base=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution005';out=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution007'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for name in ('PRIVATE_STARTUP_BINDING.json','E_BINDING.json'):
  for file,h in json.loads((base/name).read_text()).items():assert sha(Path(file))==h,file
 assert json.loads((out/'DYNAMIC_GATE_CPU_RESULTS.json').read_text())['passed']==31
 assert json.loads((out/'offline/WORKER_CPU_REGRESSION.json').read_text())['full_candidates_pass']==48
 assert json.loads((out/'offline/REAL_INPUT_REPLAY.json').read_text())['passed']==9
 (out/'scene').mkdir()
 for name in ('empty_room.ply','SCENE_VALIDATION.json'):shutil.copyfile(base/'scene'/name,out/'scene'/name)
 (out/'scene/sensor.yaml').write_text((base/'scene/sensor.yaml').read_text().replace('f3_execution005','f3_execution007'))
 for name in ('private_sim_exec.sh','controlled_startup_shadow.launch'):
  p=out/'runtime'/name;p.write_text((base/'runtime'/name).read_text().replace('f3_execution005','f3_execution007'));p.chmod(0o755 if name.endswith('.sh') else 0o644)
 host=(base/'runtime/E_INPUT_HOST.py').read_text().replace('f3_execution005','f3_execution007').replace('query_minco.inference.c1_proposal_shadow','query_minco.inference.c1_proposal_shadow_v3').replace("OUT/'runtime/native_reference_shadow'","OUT/'runtime/native_reference_dynamic'")
 host=host.replace("PYTHON='/home/lizheng/anaconda3/envs/yopo/bin/python'","os.environ['QM_DYNAMIC_CONFIG']=str(OUT/'DYNAMIC_CONFIG.json')\nPYTHON='/home/lizheng/anaconda3/envs/yopo/bin/python'")
 needle="   rospy.set_param('/quadrotor_simulator_so3/startup_release_session','f3_execution007_shadow_inputs001')"
 replacement="""   ready=json.loads((HOST/'c1/READY.json').read_text())
   code,msg,nodepid=xmlrpc.client.ServerProxy(master.lookupNode('/query_c1_proposal_shadow')).getPid('/f3_native_observer')
   assert code==1 and int(nodepid)==ready['pid']
   identity=proc(int(nodepid));assert identity and 'c1_proposal_shadow_v3' in identity['cmdline']
   assert ready['model']['checkpoint']['sha256']=='2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1'
   with (OUT/'WORKER_PID.txt').open('x') as pidfile:pidfile.write(str(nodepid))
   dump('TRUSTED_WORKER_IDENTITY.json',dict(identity=identity,method=ready['model'],readonly_scope=True))
"""+needle
 assert needle in host;host=host.replace(needle,replacement)
 (out/'runtime/E_INPUT_HOST.py').write_text(host)
 prereg=dict(kind='F3_E2_E3_ONCE_NATIVE_READONLY_RELIABILITY',run='f3_execution007/shadow_inputs001',initial_position=[-1.6,0,2],reference='Identical frozen nonQuery control prefix/stop coefficients as execution005/006. No trajectory, Controller, start gate or global target alteration.',global_goal=[3,0,2],local_goal_m=2.4,
 input_support=dict(speed_target=.8,speed_radius=.03,acceleration_max=.05,tilt_deg=.5,goal_slope_max=.175),
 limits=dict(v=1,a=1.5,j=10,soft_brake_j=.35,tracking=[.02,.05,.1]),
 deadlines=dict(activation_lead_s=.06,ready_guard_s=.02,post_generation_s=.04,source_age_s=.25,current_state_age_s=.05,late_activation_s=.01,control_max_interval_s=.04),
 opportunity_plan=dict(camera_fps=33,original_predicted_moving_supported_window_s=[.72,1.04],predicted_source_frames=[23,34],minimum_joint_valid_opportunities=20,minimum_opportunity_window_s=.5,maximum_processed_frames=40,max_source_gap_s=.1,
 success_criterion='>=20 joint-valid unique source frames spanning>=.5s, all six generated in each; >=95% of opportunities have complete full-bank+whole-body proof+ready deadline+C++ readonly verify+actual causal activation gate; accepted source gaps<=.1s. No CPU fill-in or posthoc threshold changes.'),
 native_limits=dict(host_max_s=70,child_timeout_s=85,outer_timeout_s=100,Windows_available_kb=1048576,WSL_available_kb=3145728,GPU_free_start_mib=512,GPU_free_runtime_mib=256,no_sustained_swap=True,no_OOM=True),
 policy='Generate all six, fixed q5 first, then q4/0/1/2/3 if >=5ms remains. Stop after first complete success. Connection1s/phase.7/rate.5/retained.1s. Full56 phases, same whole-body error budget. Untested candidates remain untested.',
 trusted_submission='C++ worker attached to unchanged reference-only main. SO_PEERCRED PID bound by native supervisor through read-only config; trusted Odom snapshot token; independent whole-bank bounds and geometry; immutable READONLY latch only. No write to physical references.',
 timing='Every native numerical stage wall/threadCPU/Linux schedstat wait. Record network/decode/connection/snapshot/packet/C++ receive and full E2E; asynchronous evidence process.',
 failure_policy='Exactly one native run; fail closed and stop progression on resource/control violation. Missing20 opportunities is insufficient evidence, not relaxed.',
 Query_control=False,training=False,visualization=False)
 (out/'SHADOW_PREREG.json').write_text(json.dumps(prereg,indent=2))
 (out/'shadow_preparation').mkdir()
 with (base/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:old=pickle.load(f)
 now=time.time();mono=time.monotonic();until=now+120.;volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],until,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 result={};checked=0
 for bid,b in old.items():
  prefix=thaw(b.prefix).validate(MotionLimits());entries=[]
  for s in b.stops:
   stop=thaw(s.polynomial);full=stop if s.phase==0 else concatenate(trim(prefix,0,s.phase),stop)
   full.validate(MotionLimits());proof=volume.certify(full.coefficients,full.durations,now,required_until=until)
   assert proof['status']=='CERTIFIED_FREE' and s.phase+stop.duration<120
   entries.append(StopEntry(s.phase,s.polynomial,json.dumps(proof,sort_keys=True).encode()));checked+=1
  proof=volume.certify(prefix.coefficients,prefix.durations,now,required_until=until);assert proof['status']=='CERTIFIED_FREE'
  result[bid]=PreparedReference(b.prefix,tuple(entries),json.dumps(proof,sort_keys=True).encode(),until,mono+120,'EXTERNAL_SUPERVISOR',now,mono)
 export(result,out/'shadow_preparation/NATIVE_REFERENCES.bin')
 with (out/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(result,f,protocol=4)
 scene=json.loads((out/'scene/SCENE_VALIDATION.json').read_text())
 config=dict(mode='READONLY_ONLY',owner_epoch='f3_execution007_native_reference_owner_v1',geometry=scene['quantised_xyz_sha256'],calibration=CALIBRATION,
 parent_digest=hashlib.sha256(curve_bytes(thaw(result[1].prefix))).hexdigest(),worker_pid_file=str(out/'WORKER_PID.txt'),cell_expiry_ros=until,cell_expiry_mono=mono+120)
 (out/'DYNAMIC_CONFIG.json').write_text(json.dumps(config,indent=2))
 (out/'shadow_preparation/NEW_REFERENCE_CERTIFICATION.json').write_text(json.dumps(dict(checked=checked,issued_ros=now,expiry_ros=until,old_evidence_reused=False,coefficients_unchanged=True,remaining_s=until-time.time()),indent=2))
 paths=[out/'runtime'/n for n in ('native_reference_dynamic','libproposal_numerics.so','E_INPUT_HOST.py','controlled_startup_shadow.launch','private_sim_exec.sh')]
 paths+=[root/'query_minco/inference'/n for n in ('proposal_numerics.hpp','proposal_numerics.cpp','proposal_worker.py','dynamic_proposal_gate.hpp','dynamic_proposal.py','native_reference_dynamic.cpp','c1_proposal_shadow_v3.py')]
 paths += [out/'SHADOW_PREREG.json',out/'DYNAMIC_CONFIG.json',out/'shadow_preparation/NATIVE_REFERENCES.bin',out/'shadow_preparation/TRANSPORT_BUNDLES.pkl',out/'scene/sensor.yaml',out/'scene/empty_room.ply']
 (out/'REAL_SHADOW_BINDING.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2))
 print('PREPARED',checked,'fresh reference cert remaining',until-time.time(),flush=True)
if __name__=='__main__':main()
