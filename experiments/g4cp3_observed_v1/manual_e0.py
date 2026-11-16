"""Researcher-only native forest sensor collection;no control publisher,no ARM site."""
from pathlib import Path
import sys,json,os,time,uuid,signal,socket,subprocess,argparse,xmlrpc.client
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.owned import family,stop_plant
from query_minco.integration.yopo_control_supervisor_v1.lease import OwnedProcessStop,parent_death_preexec
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--allow-sensors');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.allow_sensors!='RESEARCHER_MANUAL_E0_SENSORS_NO_ARM':raise PermissionError('EXPLICIT_SENSOR_ONLY_AUTHORITY_REQUIRED')
 binding=json.loads((O/'E0_SOURCE_BINDING.json').read_text())
 for path,h in binding.items():
  if sha(path)!=h:raise ValueError('SOURCE_OR_ASSET_DRIFT '+path)
 # Existing successful D is reviewed as historical v4 evidence,not as certification of a new execution chain.
 review=json.loads((O/'D_PHYSICAL_REVIEW.json').read_text())
 if review['status']!='PHYSICAL_PASS' or sha(review['original_D'])!=review['original_D_sha256']:raise PermissionError('D_EVIDENCE')
 if a.check_only:print('E0_SENSOR_PREFLIGHT_STATIC_PASS_NO_ROS_NO_ARM');return
 out=Path(a.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
 scenes=json.loads((R/'research_outputs/g4cp2_execution_20261111_v1/SCENES.json').read_text());scene=scenes['forest']
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11351',ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',
  CMAKE_PREFIX_PATH='/home/lizheng/YOPO/Controller/devel:/opt/ros/noetic',ROS_PACKAGE_PATH='/home/lizheng/YOPO/Controller/src:/opt/ros/noetic/share',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages',
  LD_LIBRARY_PATH='/home/lizheng/YOPO/Controller/devel/lib:/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],
  CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',11351))
 owned=[];plant=None;guard=None;error=None;terminal=None;stop_errors=[];results={}
 def launch(name,argv):
  folder=out/name;folder.mkdir();so=(folder/'stdout.log').open('xb');se=(folder/'stderr.log').open('xb')
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),started_unix=time.time(),environment={k:('[REDACTED]' if any(w in k.upper() for w in ['TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY']) else v) for k,v in env.items()}))
  try:child=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=parent_death_preexec())
  except BaseException:so.close();se.close();raise
  owned.append((name,folder,child,so,se,time.monotonic()));return child
 try:
  master=launch('master',['/opt/ros/noetic/bin/roscore','-p','11351']);rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    response=rpc.getPid('/p3_e0')
    if master.poll() is None and response[0]==1 and response[2] in [r['pid'] for r in family(master.pid)]:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('OWNED_MASTER_TIMEOUT')
  rpc.setParam('/p3_e0','/use_sim_time',False)
  collector=launch('collector',['/home/lizheng/anaconda3/envs/yopo/bin/python','-B','ros/g4cp3_observed_v1/e0_collector.py','--run',str(out),'--scene-id','frozen_forest3101_native_camera','--duration','10'])
  launch('sensor',[scene['sensor_binary'],'__name:=p3_forest_sensor','/g4b/native/odom:=/p3/native/odom','/g4b/native/depth:=/p3/native/depth','/g4b/native/camera_info:=/p3/native/camera_info'])
  argv=[scene['plant_binary'],'__name:=p3_e0_held_pose','_run_nonce:='+uuid.uuid4().hex,'_motor_initial_state:=equilibrium_hover',
    '~odom:=/p3/native/odom','~imu:=/p3/native/imu','~startup_state:=/p3/native/startup_state','~cmd:=/p3/unused_final_command']
  argv+=['_simulator/init_state_'+axis+':='+str(v) for axis,v in zip('xyz',scene['spawn'])]
  plant=launch('held_native_pose',argv);guard=OwnedProcessStop(plant)
  began=time.monotonic()
  while collector.poll() is None:
   if plant.poll() is not None:raise RuntimeError('HELD_NATIVE_PRODUCER_EXIT')
   if time.monotonic()-began>25:raise RuntimeError('E0_SENSOR_WALL_DEADLINE')
   time.sleep(.02)
  if collector.returncode not in [0,3]:raise RuntimeError('E0_COLLECTOR_CONTRACT_FAILURE')
 except BaseException as e:error=repr(e)
 finally:
  terminal,stop_errors=stop_plant(plant,guard,error or 'E0_COLLECTION_FINISHED')
  for name,folder,child,so,se,began in reversed(owned):
   if child.poll() is None:
    try:os.killpg(child.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   so.close();se.close();row=dict(pid=child.pid,actual_exit_code=code,elapsed_s=time.monotonic()-began,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log'));write(folder/'PROCESS_RESULT.json',row);results[name]=row
 result=json.loads((out/'E0_RESULT.json').read_text()) if (out/'E0_RESULT.json').exists() else dict(status='FAIL',bootstrap_status='FAIL')
 unexpected={k:v for k,v in results.items() if v['actual_exit_code'] not in ([-9] if k=='held_native_pose' else [0,3] if k=='collector' else [0])}
 write(out/'OWNER_RESULT.json',dict(status='STATIC_PASS' if not error and not stop_errors and not unexpected and result['status']=='STATIC_PASS' else 'FAIL',
  source_binding=binding,source_scope='LIVE_NATIVE_SENSORS_WITH_HELD_NONINTEGRATED_POSE',bootstrap_status=result['bootstrap_status'],physical_integration='NOT_RUN',error=error,stop_errors=stop_errors,terminal=terminal,children=results))
 print('E0_SENSOR_RESULT',result['status'],'BOOTSTRAP',result['bootstrap_status'],'PHYSICAL_INTEGRATION_NOT_RUN')
 if error or stop_errors or unexpected or result['status']!='STATIC_PASS':raise SystemExit(2)
 if result['bootstrap_status']!='STATIC_PASS':raise SystemExit(3)
if __name__=='__main__':main()
