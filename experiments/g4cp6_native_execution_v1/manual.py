"""Researcher-only physical C/D/E launcher. Never invoked by qualification scripts."""
from pathlib import Path
import sys,json,os,time,uuid,socket,signal,subprocess,argparse,xmlrpc.client
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.lifecycle import Lifecycle,atomic_json
from query_minco.integration.yopo_execution_v4.metrics import summarize
from query_minco.integration.yopo_execution_v4.owned import family,alive,kill_family,stop_plant
from query_minco.integration.yopo_execution_v4.evidence import CASES,expected_fault,predecessor,legacy_window
from query_minco.integration.yopo_control_supervisor_v1.lease import OwnedProcessStop,parent_death_preexec
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp6_native_execution_20261112_v1';PY='/home/lizheng/anaconda3/envs/yopo/bin/python'
def claim(path):
 p=Path(path).resolve()
 if (R/'research_outputs').resolve() not in p.parents:raise ValueError('OUTPUT_ROOT')
 p.parent.mkdir(parents=True,exist_ok=True);p.mkdir();return p
def episode(root,stage,case,binding,env,scene):
 out=claim(root);life=Lifecycle();owned=[];plant=None;guard=None;terminal=None;reason=None;qualification=False;injected=None;armed_at=None;anchor=None;last_hb=None;owned_identities={}
 def launch(name,argv):
  folder=out/name;folder.mkdir();so=(folder/'stdout.log').open('xb');se=(folder/'stderr.log').open('xb')
  redacted={k:('[REDACTED]' if any(w in k.upper() for w in ['TOKEN','PASSWORD','SECRET','CREDENTIAL','API_KEY']) else v) for k,v in env.items()}
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),environment=redacted,started_unix=time.time()))
  try:child=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=parent_death_preexec())
  except BaseException as error:
   so.close();se.close();write(folder/'LAUNCH_ERROR.json',dict(error=repr(error)));raise
  owned.append((name,folder,child,so,se,time.monotonic()));return child
 nonce=uuid.uuid4().hex;children={};physics_steps_at_qualification=0
 try:
  controller=launch('controller',['/opt/ros/noetic/bin/roslaunch',str(R/'ros/g4cp6_native_execution_v1/native_position_controller.launch'),'prefix:=/p2'])
  sensor=launch('sensor',[scene['sensor_binary'],'__name:=p2_sensor','/g4b/native/odom:=/p2/native/odom','/g4b/native/depth:=/p2/native/depth','/g4b/native/camera_info:=/p2/native/camera_info'])
  relay=launch('relay',[PY,'-B','ros/g4cp2_execution_v4/fault_relay.py','--run',str(out)])
  supervisor=launch('supervisor',[PY,'-B','ros/g4cp2_execution_v4/supervisor.py','--run',str(out)]+(['--forest'] if stage=='E' else []))
  reference=launch('reference',[PY,'-B','ros/g4cp2_execution_v4/reference.py','--run',str(out),'--stage',stage])
  truth=launch('truth',[PY,'-B','ros/g4cp2_execution_v4/truth_monitor.py','--run',str(out)]+(['--map-folder',scene['map_folder']] if stage=='E' else []))
  args=[scene['plant_binary'],'__name:=p2_native_plant','_run_nonce:='+nonce,'_motor_initial_state:=equilibrium_hover','_rate/odom:=100.0']
  args+=['_simulator/init_state_'+axis+':='+str(v) for axis,v in zip('xyz',scene['spawn'])]
  args+=['~odom:=/p2/native/odom','~imu:=/p2/native/imu','~cmd:=/p2/so3_cmd','~startup_state:=/p2/startup_state','~prepare_permit:=/p2/prepare_permit','~start_permit:=/p2/start_permit']
  plant=launch('plant',args);guard=OwnedProcessStop(plant);begin=time.monotonic();prepare_sent=False;arm_sent=False
  while time.monotonic()-begin<(70 if stage=='E' else 35):
   for _,_,child,_,_,_ in owned:
    for identity in family(child.pid):owned_identities[(identity['pid'],identity['start_ticks'])]=identity
   now=time.monotonic();hb=out/'supervisor_data/HEARTBEAT.json'
   if hb.exists():
    h=json.loads(hb.read_text());last_hb=h
    if now-h['monotonic']>.25:reason='SUPERVISOR_HEARTBEAT_TIMEOUT';break
    if h['mode']=='ABORT':reason=h['reason'];break
    if h['mode']=='PREPARE' and h['plant']['phase']=='HELD' and not prepare_sent:
     launch('prepare_permit',['/opt/ros/noetic/bin/rostopic','pub','-1','/p2/prepare_permit','std_msgs/String',json.dumps(dict(data=nonce))]);prepare_sent=True
    if h['mode']=='ACK' and not arm_sent:
     if len(set(h['ack_stamps']))<5 or h['plant']['integration_steps']!=0:raise ValueError('INVALID_ACK')
     launch('arm_permit',['/opt/ros/noetic/bin/rostopic','pub','-1','/p2/start_permit','std_msgs/String',json.dumps(dict(data=nonce))]);arm_sent=True
    if h['mode']=='ARMED':
     if not arm_sent or h['plant']['state_source']!='DYNAMICS_INTEGRATED' or h['plant']['integration_steps']<=0:raise ValueError('NONPHYSICAL_FEEDBACK')
     if armed_at is None:armed_at=now;anchor=h['state']['p']
     if stage=='C' and injected is None and now-armed_at>=1.:
      injected=now;write(out/'FAULT.json',dict(case=case,monotonic=now,ros_time=time.time(),armed=True,steps=h['plant']['integration_steps']))
      if case=='reference_exit':write(out/'FAULT_PROCESS_TARGETS.json',dict(case=case,targets=kill_family(reference.pid),monotonic=time.monotonic()))
      if case=='controller_exit':write(out/'FAULT_PROCESS_TARGETS.json',dict(case=case,targets=kill_family(controller.pid),monotonic=time.monotonic()))
     status=out/'reference_data/STATUS.json'
     if stage in ['D','E'] and status.exists() and json.loads(status.read_text())['done']:
      qualification=True;life.move('QUALIFIED',now);physics_steps_at_qualification=h['plant']['integration_steps'];break
   truth_hb=out/'truth_data/HEARTBEAT.json'
   if truth_hb.exists():
    t=json.loads(truth_hb.read_text())
    if t['status']!='PASS' or now-t['monotonic']>.2:reason='TRUTH_COLLISION_UNKNOWN_OR_STALE';break
   elif now-begin>8:reason='TRUTH_NO_FEEDBACK';break
   for name,folder,p,_,_,_ in owned:
    if name.endswith('_permit') and p.poll()==0:continue
    if p.poll() is not None:reason=name+'_EXIT';break
   if reason:break
   if injected is not None and now-injected>.5:reason='FAULT_NOT_DETECTED';break
   time.sleep(.01)
  else:reason='EPISODE_DEADLINE'
  if reason:
   life.fault(reason,time.monotonic())
   if stage=='C' and injected is not None and expected_fault(case,reason):
    qualification=True;life.move('QUALIFIED',time.monotonic());physics_steps_at_qualification=last_hb['plant']['integration_steps'] if last_hb and last_hb.get('plant') else 0
 except BaseException as e:reason=repr(e);life.fault(reason,time.monotonic())
 finally:
  life.move('TERMINATING',time.monotonic());stop_begin=time.monotonic()
  terminal,stop_errors=stop_plant(plant,guard,reason or 'QUALIFIED_STAGE_FINISH')
  stop_time=time.monotonic()
  if terminal and terminal['terminal_state']!='STOP_NOT_CONFIRMED':
   terminal.update(stop_requested_monotonic=stop_begin,stop_confirmed_monotonic=stop_time,exit_observed_monotonic=stop_time)
   atomic_json(out/'STOP_CONFIRMED.json',dict(monotonic=stop_time,ros_time=time.time(),terminal=terminal))
  # Let callback-owning Python nodes observe the confirmed plant stop and close journals themselves.
  for name,folder,p,so,se,began in reversed(owned):
   if p is not plant and p.poll() is None:
    if name in ['supervisor','reference','truth']:
     try:p.wait(timeout=2.)
     except subprocess.TimeoutExpired:pass
    if p.poll() is None:
     try:os.killpg(p.pid,signal.SIGINT)
     except ProcessLookupError:pass
   try:code=p.wait(timeout=5.)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
   so.close();se.close();row=dict(pid=p.pid,actual_exit_code=code,elapsed_s=time.monotonic()-began,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log'))
   write(folder/'PROCESS_RESULT.json',row);children[name]=row
  orphaned=[row for row in owned_identities.values() if alive(row)]
  for row in orphaned:
   if alive(row):
    try:os.kill(row['pid'],signal.SIGKILL)
    except ProcessLookupError:pass
  life.move('TERMINATED',time.monotonic())
 events_path=out/'supervisor_data/EVENTS.jsonl';events=[json.loads(x) for x in events_path.read_text().splitlines()] if events_path.exists() else []
 accepted=[e for e in events if e.get('kind')=='plant' and e.get('accepted') and e['payload']['integration_steps']>0]
 start=min([e['monotonic'] for e in accepted] or [stop_time])
 full_run=summarize(events,end=stop_time,anchor=anchor)
 first_state_stamp=min([e['stamp'] for e in accepted] or [float('inf')])
 physical_events=[e for e in events if e.get('stamp') is not None and e['stamp']>=first_state_stamp]
 metrics=summarize(physical_events,end=stop_time,anchor=anchor)
 active_fault_end=injected if stage=='C' and injected else stop_time
 healthy=summarize(events,begin=start,end=active_fault_end,anchor=anchor)
 truth_path=out/'truth_data/EVENTS.jsonl'
 truth_rows=[json.loads(x) for x in truth_path.read_text().splitlines()] if truth_path.exists() else []
 truth_bad=[x for x in truth_rows if x.get('kind')=='truth' and x.get('monotonic',0)<=stop_time and x.get('status')!='PASS']
 if not truth_rows or truth_bad:qualification=False
 if stage!='C' and any(e.get('mode')=='ABORT' and e.get('monotonic',0)<=stop_time for e in events):qualification=False
 if full_run['maxima']['speed']>1+1e-6 or full_run['maxima']['acceleration']>8+1e-6 or full_run['maxima']['actual_jerk_fd_diagnostic']>100:qualification=False
 if healthy['maxima']['actual_jerk_fd_diagnostic']>100:qualification=False
 if healthy['maxima']['speed']>1+1e-6 or healthy['maxima']['acceleration']>8+1e-6:qualification=False
 # Every active stream must remain fresh;duplicates do not extend liveness.
 for values in healthy['intervals'].values():
  if (values['max_source_gap_s'] or 0)>.2 or (values['max_receipt_gap_s'] or 0)>.2 or values['max_age_s']>.2 or (values['end_receipt_age_s'] or 0)>.2 or (values['end_source_age_s'] or 0)>.2:qualification=False
 enough_feedback=all(healthy['intervals'].get(k,{}).get('n',0)>=5 for k in ['odom','imu','depth','request','raw','plant']) and metrics['max_integration_steps']>=500
 expected=bool(enough_feedback and stage=='C' and injected is not None and reason and expected_fault(case,reason) and stop_time-injected<=.35 and accepted)
 if stage=='C' and not expected:qualification=False
 motion={}
 if stage in ['D','E'] and accepted:
  positions=[e['payload'] for e in events if e.get('kind')=='odom' and e.get('accepted') and start<=e['monotonic']<=stop_time]
  status=out/'reference_data/STATUS.json';ref=json.loads(status.read_text()) if status.exists() else {}
  if positions and ref:
   import numpy as np
   last=positions[-1];progress=float(last['p'][0]-ref['anchor'][0]);error=float(np.linalg.norm(np.array(last['p'])-ref['target']));speed=float(np.linalg.norm(last['v']))
   motion=dict(progress_m=progress,terminal_position_error_m=error,terminal_speed=speed,replans=ref['replans'])
   if stage=='D' and not(progress>=.25 and error<=.10 and speed<=.10 and metrics['maxima']['tracking_error']<=.15):qualification=False
   if stage=='E' and not(progress>=2 and ref['replans']>=20 and metrics['maxima']['tracking_error']<=.15):qualification=False
  else:qualification=False
 life.qualified=life.qualified and qualification
 expected_children=['plant']+(['reference'] if case=='reference_exit' else [])+(['controller'] if case=='controller_exit' else [])
 # A native watchdog exit3 is expected only in a matching C fault.
 if expected and children.get('plant',{}).get('actual_exit_code')==3:children['plant']['accepted_native_watchdog_exit']=3;expected_children.append('plant')
 outcome=life.result(children,terminal,expected_killed=expected_children,expected_fault=expected)
 # Node lifecycle files must exist and report clean journal teardown unless intentionally killed.
 node_results={}
 for name,folder in [('supervisor','supervisor_data'),('reference','reference_data'),('truth','truth_data')]:
  p=out/folder/'RESULT.json'
  if name=='reference' and case=='reference_exit':continue
  node_results[name]=json.loads(p.read_text()) if p.exists() else dict(status='FAIL',reason='MISSING_NODE_FINAL')
  if node_results[name]['status']!='PASS':outcome['cleanup_status']='FAIL';outcome['status']='FAIL'
 if stop_errors:outcome.update(status='FAIL',cleanup_status='FAIL',stop_errors=stop_errors)
 if orphaned:outcome.update(status='FAIL',cleanup_status='FAIL',orphaned_owned_processes=orphaned)
 outcome.update(stage=stage,case=case,source_binding=binding,physical_feedback=bool(accepted),fault_injected_monotonic=injected,terminal=terminal,healthy_window=healthy,
  full_available_run=full_run,physical_start_state_stamp=first_state_stamp if accepted else None,full_physical_window=metrics,truth_bad=truth_bad,motion=motion,children=children,node_results=node_results,physics_steps_at_qualification=physics_steps_at_qualification,events_sha256=sha(events_path) if events_path.exists() else None,network_updates=0)
 write(out/'RESULT.json',outcome);return outcome
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['C','D'],required=True);p.add_argument('--output',required=True);p.add_argument('--allow-physics');p.add_argument('--prior');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.allow_physics!='RESEARCHER_MANUAL_'+a.stage+'_ONLY':raise PermissionError('MANUAL_STAGE_AUTHORITY_REQUIRED')
 if a.stage=='E' and json.loads((O/'FOREST_SENSOR_ROUTE_QUALIFICATION.json').read_text())['status']!='PASS':raise PermissionError('E_INITIAL_OBSERVED_ROUTE_NOT_QUALIFIED')
 binding=json.loads((O/'SOURCE_BINDING.json').read_text())
 for path,h in binding.items():
  if sha(path)!=h:raise ValueError('SOURCE_ASSET_DRIFT '+path)
 if a.stage=='C':legacy_window(O/'LEGACY_B_PHYSICAL_WINDOW.json')
 elif not a.prior:raise PermissionError('PREVIOUS_STAGE_REQUIRED')
 else:predecessor(a.prior,'C_CAMPAIGN' if a.stage=='D' else 'D',binding)
 if a.check_only:
  json.loads((O/'SCENES.json').read_text());print('PREFLIGHT_PASS_NO_ROS_NO_ARM');return
 out=claim(a.output);write(out/'RUN_IDENTITY.json',dict(argv=sys.argv,cwd=str(R),stage=a.stage,source_binding=binding,prior=a.prior,prior_sha256=sha(a.prior) if a.prior else None));scene=json.loads((O/'SCENES.json').read_text())['forest' if a.stage=='E' else 'open']
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11359',ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',
  CMAKE_PREFIX_PATH='/home/lizheng/YOPO/Controller/devel:/opt/ros/noetic',ROS_PACKAGE_PATH='/home/lizheng/YOPO/Controller/src:/opt/ros/noetic/share',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages',
  LD_LIBRARY_PATH='/home/lizheng/YOPO/Controller/devel/lib:/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 # Match roscore reuse semantics without accepting a live listener or killing another run.
 with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',11359))
 master_dir=out/'master';master_dir.mkdir();so=(master_dir/'stdout.log').open('xb');se=(master_dir/'stderr.log').open('xb')
 argv=['/opt/ros/noetic/bin/roscore','-p','11359'];write(master_dir/'PROCESS_START.json',dict(argv=argv,cwd=str(R),started_unix=time.time()))
 master=subprocess.Popen(argv,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=parent_death_preexec());results=[]
 try:
  rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    response=rpc.getPid('/p2_manual')
    if master.poll() is None and response[0]==1 and response[2] in [row['pid'] for row in family(master.pid)]:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('MASTER_TIMEOUT')
  rpc.setParam('/p2_manual','/use_sim_time',False)
  for case in CASES if a.stage=='C' else ['none']:
   result=episode(out/case,a.stage,case,binding,env,scene);results.append(dict(case=case,status=result['status'],result_path=str(out/case/'RESULT.json'),result_sha256=sha(out/case/'RESULT.json')))
   if result['status']!='PASS':break
 finally:
  if master.poll() is None:os.killpg(master.pid,signal.SIGINT)
  try:code=master.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(master.pid,signal.SIGKILL);code=master.wait()
  so.close();se.close();write(master_dir/'PROCESS_RESULT.json',dict(pid=master.pid,actual_exit_code=code,stdout_sha256=sha(master_dir/'stdout.log'),stderr_sha256=sha(master_dir/'stderr.log')))
 if a.stage=='C':
  status='PASS' if len(results)==6 and all(r['status']=='PASS' for r in results) and code==0 else 'FAIL'
  final=dict(stage='C_CAMPAIGN',status=status,source_binding=binding,episodes=results)
 else:
  final=json.loads((Path(results[0]['result_path'])).read_text()) if results else dict(status='FAIL',stage=a.stage)
  if code!=0:final.update(status='FAIL',master_cleanup='FAIL')
 write(out/'RESULT.json',final);print('MANUAL_RESULT',a.stage,final['status'])
 if final['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
