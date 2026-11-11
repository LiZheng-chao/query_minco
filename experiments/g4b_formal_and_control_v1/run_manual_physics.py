"""MANUAL ONLY:owned physics,real native sensors,and independent simulation fail-stop."""
from pathlib import Path
import sys,json,os,time,subprocess,signal,socket,xmlrpc.client,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_control_supervisor_v1.lease import OwnedProcessStop,parent_death_preexec
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];OWNER=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
CASES=['timeout','depth_loss','odom_delay','nan','planner_stop','controller_fault']
def verify_evidence(path,expected_inputs):
 d=json.loads(Path(path).read_text())
 if d.get('schema')!='g4b_actual_physics_terminal_safety_v1' or d.get('status')!='PASS' or d.get('actual_physics') is not True or d.get('source_binding')!=expected_inputs or [x['fault'] for x in d['cases']]!=CASES or not all(x['status']=='PASS' for x in d['cases']):raise PermissionError('REAL_PHYSICS_SAFETY_EVIDENCE_REQUIRED')
 for row in d['cases']:
  if not(row.get('real_feedback') and row.get('active_feedback_loop_observed') and row.get('terminal',{}).get('stop_state_observed') and row['terminal'].get('terminal_state')=='OWNED_PROCESS_TERMINATED'):raise PermissionError('UNVERIFIED_PHYSICS_STOP')
  episode=Path(path).parent/row['fault']/'EPISODE.json'
  if sha(episode)!=d['episodes_sha256'][row['fault']] or json.loads(episode.read_text())!=row:raise PermissionError('EPISODE_IDENTITY')
  proc=json.loads((episode.parent/'physics/PROCESS_RESULT.json').read_text())
  if proc['pid']!=row['terminal']['pid'] or proc['actual_exit_code']!=row['terminal']['actual_exit_code']:raise PermissionError('PHYSICS_PID_EXIT_IDENTITY')
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['safety','track'],required=True);p.add_argument('--output',required=True);p.add_argument('--allow-physics');p.add_argument('--safety-evidence');a=p.parse_args()
 if a.allow_physics!='RESEARCHER_MANUAL_SIMULATION_ONLY':raise PermissionError('MANUAL_PHYSICS_AUTHORIZATION_REQUIRED')
 binding=json.loads((OWNER/'MANUAL_PHYSICS_SOURCE_BINDING.json').read_text())
 for file,h in binding.items():
  if sha(file)!=h:raise ValueError('MANUAL_SOURCE_DRIFT '+file)
 if a.stage=='track':
  if not a.safety_evidence:raise PermissionError('SAFETY_EVIDENCE_REQUIRED')
  verify_evidence(a.safety_evidence,binding)
 out=Path(a.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11343',ROS_IP='127.0.0.1',ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),
  CMAKE_PREFIX_PATH=str(R.parent/'YOPO/Controller/devel')+':/opt/ros/noetic',ROS_PACKAGE_PATH=str(R.parent/'YOPO/Controller/src')+':/opt/ros/noetic/share',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:'+str(R.parent/'YOPO/Controller/devel/lib/python3/dist-packages'),
  LD_LIBRARY_PATH=str(R.parent/'YOPO/Controller/devel/lib')+':/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:'+str(R.parent/'YOPO/Simulator/devel/lib'),
  PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'])
 with socket.socket() as sock:sock.bind(('127.0.0.1',11343))
 py='/home/lizheng/anaconda3/envs/yopo/bin/python';owned=[]
 def launch(folder,argv):
  folder.mkdir(parents=True);write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),started_unix=time.time(),env={k:env[k] for k in ['ROS_MASTER_URI','ROS_HOME','ROS_LOG_DIR','CMAKE_PREFIX_PATH','PYTHONPATH','LD_LIBRARY_PATH']}))
  stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb');proc=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=parent_death_preexec())
  item=(folder,proc,stdout,stderr,time.monotonic());owned.append(item);return item
 def cleanup(items):
  for folder,proc,stdout,stderr,start in reversed(items):
   if proc.poll() is None:
    try:os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=proc.wait(timeout=6)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   stdout.close();stderr.close();write(folder/'PROCESS_RESULT.json',dict(pid=proc.pid,actual_exit_code=code,elapsed_s=time.monotonic()-start,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log')))
 results=[];master=launch(out/'roscore',['/opt/ros/noetic/bin/roscore','-p','11343'])
 try:
  rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    if rpc.getPid('/g4b_manual')[0]==1:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('MASTER_UNAVAILABLE')
  time.sleep(1);rpc.setParam('/g4b_manual','/use_sim_time',False)
  build=json.loads((OWNER/'open_sensor_build/BUILD_IDENTITY.json').read_text())
  cases=CASES if a.stage=='safety' else ['none']
  for fault in cases:
   folder=out/fault;folder.mkdir();base=len(owned);fault_at=time.monotonic()+7.
   controller=launch(folder/'controller',['/opt/ros/noetic/bin/roslaunch',str(R/'ros/g4b_control_supervisor_v1/controller_only.launch'),'prefix:=/g4b'])
   sensor=launch(folder/'native_sensor',[build['binary'],'__name:=g4b_open_sensor'])
   supervisor=launch(folder/'supervisor',[py,'-B','ros/g4b_control_supervisor_v1/supervisor_node.py','--prefix','/g4b','--output',str(folder/'lease')])
   relay=launch(folder/'relay',[py,'-B','ros/g4b_control_supervisor_v1/sensor_relay.py','--fault',fault,'--fault-at-file',str(folder/'FAULT_SCHEDULE.json')])
   reference=launch(folder/'reference',[py,'-B','ros/g4b_control_supervisor_v1/physical_reference.py','--mode','hold' if a.stage=='safety' else 'track','--fault',fault,'--fault-at-file',str(folder/'FAULT_SCHEDULE.json'),'--output',str(folder/'reference_records')])
   monitor=launch(folder/'monitor',[py,'-B','ros/g4b_control_supervisor_v1/feedback_monitor.py','--output',str(folder/'feedback')])
   # All Python/controller/render processes must be ready BEFORE any physics integration.
   ready=False
   for _ in range(1500):
    heartbeat=folder/'lease/HEARTBEAT.json';rp=folder/'reference_records/READY.json'
    pubs=rpc.getSystemState('/g4b_preflight')[2][0];topics={x[0] for x in pubs}
    if heartbeat.exists() and rp.exists() and {'/g4b/native/depth','/g4b/so3_raw','/g4b/odom'}<=topics:
     lease=json.loads(heartbeat.read_text())
     if lease['mode']=='WAIT' and time.monotonic()-lease['monotonic']<.25:ready=True;break
    if any(item[1].poll() is not None for item in [controller,sensor,supervisor,relay,reference]):raise RuntimeError('SERVICE_DIED_BEFORE_PHYSICS')
    time.sleep(.01)
   if not ready:raise RuntimeError('PRE_PHYSICS_READINESS_TIMEOUT')
   fault_at=time.monotonic()+3.;write(folder/'FAULT_SCHEDULE.json',dict(monotonic=fault_at,registered_before_physics=True))
   # This is the only physics launch site;unreachable without explicit manual flag.
   physics=launch(folder/'physics',[str(R.parent/'YOPO/Controller/devel/lib/so3_quadrotor_simulator/quadrotor_simulator_so3'),
    '__name:=g4b_physics','_rate/odom:=100.0','_simulator/init_state_x:=0.0','_simulator/init_state_y:=0.0','_simulator/init_state_z:=2.0',
    '~odom:=/g4b/native/odom','~imu:=/g4b/native/imu','~cmd:=/g4b/so3_cmd','~force_disturbance:=/g4b/force_disturbance','~moment_disturbance:=/g4b/moment_disturbance'])
   interlock=OwnedProcessStop(physics[1]);began=time.monotonic();injected=False;active=False;terminal=None;reason=None
   try:
    while time.monotonic()-began<16:
     now=time.monotonic();heartbeat=folder/'lease/HEARTBEAT.json'
     if heartbeat.exists():
      lease=json.loads(heartbeat.read_text());active|=lease['mode']=='ACTIVE'
      if now-lease['monotonic']>.25:reason='SUPERVISOR_HEARTBEAT_TIMEOUT'
      if lease['mode']=='ABORT':reason=lease['reason']
     elif now-began>5:reason='SUPERVISOR_STARTUP_TIMEOUT'
     if now-began>5 and not active:reason='NO_ACTIVE_PLANNING_LEASE'
     if a.stage=='safety' and now>=fault_at and not injected:
      if fault=='planner_stop':os.killpg(reference[1].pid,signal.SIGKILL)
      if fault=='controller_fault':os.killpg(controller[1].pid,signal.SIGINT)
      injected=True
     for name,item in [('supervisor',supervisor),('controller',controller),('reference',reference),('sensor',sensor),('relay',relay)]:
      if item[1].poll() is not None:
       done=folder/'reference_records/RESULT.json'
       if name=='reference' and a.stage=='track' and done.exists() and json.loads(done.read_text())['status']=='PASS':reason='REFERENCE_TRACK_COMPLETED'
       else:reason=name+'_EXIT'
     if physics[1].poll() is not None:reason='PHYSICS_EARLY_EXIT'
     if reason:terminal=interlock.stop(reason);break
     time.sleep(.01)
    if terminal is None:reason='MANUAL_RUN_WALL_TIMEOUT';terminal=interlock.stop(reason)
   finally:
    if physics[1].poll() is None:terminal=interlock.stop('GUARDIAN_FINALLY_STOP')
    cleanup(owned[base:]);del owned[base:]
   feedback=json.loads((folder/'feedback/RESULT.json').read_text()) if (folder/'feedback/RESULT.json').exists() else {}
   has_feedback=len(feedback.get('records',{}).get('odom',[]))>=20 and len(feedback.get('records',{}).get('raw_controller',[]))>=20
   if a.stage=='safety':
    allowed={'timeout':['request_TIMEOUT','request_TIMEOUT_LATE'],'depth_loss':['depth_TIMEOUT','depth_TIMEOUT_LATE'],'odom_delay':['odom_STAMP','odom_TIMEOUT_LATE'],'nan':['request_NAN'],'planner_stop':['reference_EXIT','request_TIMEOUT'],'controller_fault':['controller_EXIT','controller_TIMEOUT']}
    passed=active and has_feedback and injected and reason in allowed[fault] and terminal is not None and terminal['stop_state_observed'] and terminal['terminal_state']=='OWNED_PROCESS_TERMINATED' and 0<=now-fault_at<=.4
   else:passed=reason=='REFERENCE_TRACK_COMPLETED' and has_feedback and active
   rec=dict(fault=fault,status='PASS' if passed else 'FAIL',active_feedback_loop_observed=active,real_feedback=has_feedback,reason=reason,terminal=terminal,actual_physics=True)
   write(folder/'EPISODE.json',rec);results.append(rec)
   if not passed:break
 finally:cleanup(owned)
 status='PASS' if all(x['status']=='PASS' for x in results) and len(results)==len(CASES if a.stage=='safety' else ['none']) else 'FAIL'
 write(out/'RESULT.json',dict(schema='g4b_actual_physics_terminal_safety_v1' if a.stage=='safety' else 'g4b_first_real_state_goal_tracking_v1',
  status=status,actual_physics=True,source_binding=binding,cases=results,episodes_sha256={x['fault']:sha(out/x['fault']/'EPISODE.json') for x in results},emergency_mode='owned simulation pause+terminate,not certified hover/real-aircraft braking',new_network_updates=0))
 print('MANUAL_PHYSICS_RESULT',a.stage,status)
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
