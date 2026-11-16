"""Manual stages A/B only. No training,no trajectories,no forest flight."""
from pathlib import Path
import sys,argparse,json,os,time,uuid,socket,signal,subprocess,xmlrpc.client,math,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_control_supervisor_v1.lease import parent_death_preexec,OwnedProcessStop
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4cp1_startup_control_20261111_v1'
def prior_held(path,binding,motor_initial_state):
 d=json.loads(Path(path).read_text())
 stamps=d.get('ack_stamps',[])
 if not isinstance(stamps,list) or len(stamps)<5 or any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):raise PermissionError('STRICTLY_UNIQUE_HELD_ACK_REQUIRED')
 if d.get('motor_initial_state')!=motor_initial_state or d.get('status')!='PASS' or d.get('stage')!='A_HELD' or d.get('source_binding')!=binding or d.get('max_integration_steps')!=0 or len(d.get('ack_stamps',[]))<5:raise PermissionError('QUALIFIED_HELD_EVIDENCE_REQUIRED')
 proc=Path(d['plant_process_result'])
 if sha(proc)!=d['plant_process_result_sha256'] or json.loads(proc.read_text())['pid']!=d['plant_pid']:raise PermissionError('HELD_PROCESS_IDENTITY')
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['held','hover'],default='held');p.add_argument('--output',required=True)
 p.add_argument('--allow-held-processes',required=True);p.add_argument('--allow-first-integration');p.add_argument('--prior-held');p.add_argument('--motor-initial-state',choices=['zero','equilibrium_hover'],default='zero');p.add_argument('--accept-equilibrium-initial-condition');a=p.parse_args()
 if a.allow_held_processes!='RESEARCHER_MANUAL_HELD_ONLY':raise PermissionError('EXPLICIT_HELD_AUTHORITY_REQUIRED')
 if a.stage=='hover' and(a.allow_first_integration!='RESEARCHER_MANUAL_HOVER_ONLY' or not a.prior_held):raise PermissionError('SEPARATE_HOVER_AUTHORITY_AND_HELD_EVIDENCE_REQUIRED')
 if a.motor_initial_state=='equilibrium_hover' and a.accept_equilibrium_initial_condition!='RESEARCHER_ACCEPT_PRESPUN_INITIAL_CONDITION':raise PermissionError('EXPLICIT_PRESPUN_INITIAL_CONDITION_REQUIRED')
 if a.stage=='hover' and a.motor_initial_state=='zero':raise PermissionError('COLD_START_ACCELERATION_CONTRACT_NO_GO')
 binding=json.loads((O/'MANUAL_SOURCE_BINDING.json').read_text())
 for f,h in binding.items():
  if sha(f)!=h:raise ValueError('SOURCE_BINARY_ASSET_DRIFT '+f)
 predecessor=prior_held(a.prior_held,binding,a.motor_initial_state) if a.stage=='hover' else None
 out=Path(a.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
 if predecessor:write(out/'HELD_EVIDENCE_COPY.json',predecessor)
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11347',ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),
  ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',CMAKE_PREFIX_PATH='/home/lizheng/YOPO/Controller/devel:/opt/ros/noetic',
  ROS_PACKAGE_PATH='/home/lizheng/YOPO/Controller/src:/opt/ros/noetic/share',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages',
  LD_LIBRARY_PATH='/home/lizheng/YOPO/Controller/devel/lib:/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib',
  PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with socket.socket() as sock:sock.bind(('127.0.0.1',11347))
 owned=[];nonce=uuid.uuid4().hex;status='FAIL';reason=None;history=[];terminal=None;physics=None;interlock=None;ack=[]
 def launch(name,argv):
  folder=out/name;folder.mkdir();so=(folder/'stdout.log').open('xb');se=(folder/'stderr.log').open('xb')
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),environment={k:(dict(redacted=True,value_sha256=hashlib.sha256(v.encode()).hexdigest()) if any(t in k.upper() for t in ['TOKEN','PASSWORD','SECRET','CREDENTIAL','API_KEY']) else v) for k,v in env.items()},started_unix=time.time()))
  proc=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=parent_death_preexec())
  owned.append((folder,proc,so,se,time.monotonic()));return proc
 try:
  launch('master',['/opt/ros/noetic/bin/roscore','-p','11347']);rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    if rpc.getPid('/g4cp1_manual')[0]==1:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('MASTER_TIMEOUT')
  rpc.setParam('/g4cp1_manual','/use_sim_time',False)
  launch('controller',['/opt/ros/noetic/bin/roslaunch',str(R/'ros/g4b_control_supervisor_v1/controller_only.launch'),'prefix:=/g4cp1'])
  sensor=json.loads((R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/BUILD_IDENTITY.json').read_text())
  launch('sensor',[sensor['binary'],'__name:=g4cp1_sensor','/g4b/native/odom:=/g4cp1/odom','/g4b/native/depth:=/g4cp1/depth','/g4b/native/camera_info:=/g4cp1/camera_info'])
  argv=['/home/lizheng/anaconda3/envs/yopo/bin/python','-B','ros/g4cp1_startup_v3/prepare_hold_node.py','--prepare-authorized','--output',str(out/'preparation')]
  if a.stage=='hover':argv+=['--arm-authorized']
  argv+=['--motor-initial-state',a.motor_initial_state]
  launch('prepare',argv)
  build=json.loads((O/'build_equilibrium_v3/BUILD_IDENTITY.json').read_text())
  physics=launch('plant',[build['binary'],'__name:=g4cp1_native_plant','_run_nonce:='+nonce,'_motor_initial_state:='+a.motor_initial_state,'_rate/odom:=100.0',
   '_simulator/init_state_x:=0.0','_simulator/init_state_y:=0.0','_simulator/init_state_z:=2.0',
   '~odom:=/g4cp1/odom','~imu:=/g4cp1/imu','~cmd:=/g4cp1/so3_cmd','~startup_state:=/g4cp1/startup_state','~prepare_permit:=/g4cp1/prepare_permit','~start_permit:=/g4cp1/start_permit'])
  interlock=OwnedProcessStop(physics)
  begin=time.monotonic();prepared=False;arm_sent=False;active_start=None;sent_prepare=False;last_stamp=0.
  while time.monotonic()-begin<40:
   for folder,proc,_,_,_ in owned:
    if folder.name.endswith('_permit') and proc.poll()==0:continue
    if proc.poll() is not None:raise RuntimeError('OWNED_COMPONENT_EXIT '+str(folder.name))
   hb=out/'preparation/HEARTBEAT.json'
   if hb.exists():
    d=json.loads(hb.read_text())
    if time.monotonic()-d['monotonic']>.25:raise RuntimeError('SUPERVISOR_HEARTBEAT_TIMEOUT')
    if d['mode']=='ABORT':raise RuntimeError('SUPERVISOR_ABORT '+str(d['reason']))
    if d['odom_stamp']!=last_stamp:history.append(d);last_stamp=d['odom_stamp']
    if d['mode']=='PREPARE' and d['plant']['phase']=='HELD' and not sent_prepare:
     launch('prepare_permit',['/opt/ros/noetic/bin/rostopic','pub','-1','/g4cp1/prepare_permit','std_msgs/String',json.dumps(dict(data=nonce))]);sent_prepare=True
    if d['mode']=='ACK' and not prepared:
     if d['plant']['phase']!='ACK' or d['plant']['integration_steps']!=0 or len(d['ack_stamps'])<5:raise RuntimeError('INVALID_FINAL_ACK')
     prepared=True;ack=d['ack_stamps'];write(out/'ACK_EVIDENCE.json',d)
     if a.stage=='held':status='PASS';break
     launch('arm_permit',['/opt/ros/noetic/bin/rostopic','pub','-1','/g4cp1/start_permit','std_msgs/String',json.dumps(dict(data=nonce))]);arm_sent=True
    if d['mode']=='ARMED':
     if not arm_sent or d['plant']['state_source']!='DYNAMICS_INTEGRATED' or d['plant']['integration_steps']<=0:raise RuntimeError('INVALID_PHYSICAL_FEEDBACK')
     if active_start is None:active_start=time.monotonic()
     acc=d['imu']['a_world']
     if math.sqrt(sum(x*x for x in acc))>8+1e-6:raise RuntimeError('MEASURED_ACCELERATION_OVER_8_STARTUP_NOT_EXEMPT')
     if time.monotonic()-active_start>=5:
      physical=[x for x in history if x['mode']=='ARMED'];last=physical[-1]
      position_error=math.sqrt(sum((x-y)**2 for x,y in zip(last['state']['p'],last['request']['pva'][0])))
      speed=math.sqrt(sum(x*x for x in last['state']['v']))
      if len(physical)<200 or position_error>.10 or speed>.15:raise RuntimeError('HOVER_ACCEPTANCE_NOT_MET')
      status='PASS';break
   time.sleep(.01)
  else:raise RuntimeError('PHASE_TIMEOUT')
 except BaseException as e:reason=repr(e)
 finally:
  if physics is not None:
   if interlock is not None:terminal=interlock.stop(reason or 'MANUAL_STAGE_COMPLETED')
   elif physics.poll() is not None:terminal=dict(terminal_state='OWNED_PROCESS_ALREADY_EXITED',actual_exit_code=physics.returncode,stop_state_observed=False)
   else:
    os.killpg(physics.pid,signal.SIGKILL);terminal=dict(terminal_state='OWNED_PROCESS_TERMINATED',actual_exit_code=physics.wait(timeout=3),stop_state_observed=False)
  for folder,proc,so,se,start in reversed(owned):
   if proc.poll() is None:
    try:os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=proc.wait(timeout=1.)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   so.close();se.close();write(folder/'PROCESS_RESULT.json',dict(pid=proc.pid,actual_exit_code=code,elapsed_s=time.monotonic()-start,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log')))
  result=dict(motor_initial_state=a.motor_initial_state,status=status,stage='A_HELD' if a.stage=='held' else 'B_HOVER',reason=reason,source_binding=binding,ack_stamps=ack,history=history,terminal=terminal,
   max_integration_steps=max([x['plant']['integration_steps'] for x in history if x.get('plant')]+[0]),network_updates=0,actual_integrated_feedback_observed=any(x['mode']=='ARMED' and x['plant']['integration_steps']>0 for x in history))
  if physics is not None:
   path=out/'plant/PROCESS_RESULT.json';result.update(plant_process_result=str(path),plant_process_result_sha256=sha(path),plant_pid=physics.pid)
  write(out/('ACCEPTANCE_A.json' if a.stage=='held' else 'ACCEPTANCE_B.json'),result)
 print('MANUAL_STAGE',a.stage,status,reason)
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
