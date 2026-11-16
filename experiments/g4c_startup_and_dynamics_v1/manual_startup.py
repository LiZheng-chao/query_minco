"""MANUAL ONLY. Default held-readiness;explicit separate flag allows short physical hover."""
from pathlib import Path
import sys,argparse,json,os,time,uuid,signal,subprocess,socket,xmlrpc.client
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_control_supervisor_v1.lease import parent_death_preexec
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--allow-held-processes',required=True)
 p.add_argument('--stage',choices=['held-readiness','armed-hover'],default='held-readiness')
 p.add_argument('--allow-first-integration');a=p.parse_args()
 if a.allow_held_processes!='RESEARCHER_MANUAL_HELD_STARTUP_ONLY':raise PermissionError('MANUAL_AUTHORITY_REQUIRED')
 if a.stage=='armed-hover' and a.allow_first_integration!='RESEARCHER_MANUAL_SHORT_HOVER_ONLY':raise PermissionError('SEPARATE_INTEGRATION_AUTHORITY_REQUIRED')
 binding=json.loads((O/'MANUAL_STARTUP_SOURCE_BINDING.json').read_text())
 for f,h in binding.items():
  if sha(f)!=h:raise ValueError('SOURCE_OR_BINARY_DRIFT '+f)
 out=Path(a.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11345',ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),
  ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',CMAKE_PREFIX_PATH='/home/lizheng/YOPO/Controller/devel:/opt/ros/noetic',
  ROS_PACKAGE_PATH='/home/lizheng/YOPO/Controller/src:/opt/ros/noetic/share',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages',
  LD_LIBRARY_PATH='/home/lizheng/YOPO/Controller/devel/lib:/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib',
  PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with socket.socket() as sock:sock.bind(('127.0.0.1',11345))
 owned=[];nonce=uuid.uuid4().hex;status='FAIL';reason=None;history=[]
 def launch(name,argv):
  folder=out/name;folder.mkdir();so=(folder/'stdout.log').open('xb');se=(folder/'stderr.log').open('xb')
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),env=env,started_unix=time.time()))
  proc=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=parent_death_preexec())
  owned.append((folder,proc,so,se,time.monotonic()));return proc
 try:
  launch('master',['/opt/ros/noetic/bin/roscore','-p','11345'])
  rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    if rpc.getPid('/g4c_manual')[0]==1:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('MASTER_TIMEOUT')
  rpc.setParam('/g4c_manual','/use_sim_time',False)
  launch('controller',['/opt/ros/noetic/bin/roslaunch',str(R/'ros/g4b_control_supervisor_v1/controller_only.launch'),'prefix:=/g4c'])
  sensor=json.loads((R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/BUILD_IDENTITY.json').read_text())
  launch('sensor',[sensor['binary'],'__name:=g4c_open_sensor','/g4b/native/odom:=/g4c/odom','/g4b/native/depth:=/g4c/depth','/g4b/native/camera_info:=/g4c/camera_info'])
  py='/home/lizheng/anaconda3/envs/yopo/bin/python'
  launch('prepare',[py,'-B','ros/g4c_startup_v2/prepare_hold_node.py','--authorize-bootstrap-hold','--output',str(out/'preparation')])
  binary=json.loads((O/'startup_build_v2/BUILD_IDENTITY.json').read_text())['binary']
  physics=launch('gated_plant',[binary,'__name:=g4c_gated_plant','_run_nonce:='+nonce,'_rate/odom:=100.0',
   '_simulator/init_state_x:=0.0','_simulator/init_state_y:=0.0','_simulator/init_state_z:=2.0',
   '~odom:=/g4c/odom','~imu:=/g4c/imu','~cmd:=/g4c/so3_cmd','~startup_state:=/g4c/startup_state','~start_permit:=/g4c/start_permit'])
  begin=time.monotonic();armed_at=None;prepared=False
  while time.monotonic()-begin<30:
   for folder,proc,_,_,_ in owned:
    if folder.name=='manual_permit' and proc.poll()==0:continue
    if proc.poll() is not None:raise RuntimeError('OWNED_COMPONENT_EXIT '+str(proc.pid))
   hb=out/'preparation/HEARTBEAT.json'
   if hb.exists():
    d=json.loads(hb.read_text());history.append(d)
    if time.monotonic()-d['monotonic']>.25:raise RuntimeError('SUPERVISOR_HEARTBEAT_STALE')
    if d['mode']=='ABORT':raise RuntimeError('LATCHED_ABORT '+str(d['reason']))
    if d['mode']=='PREPARED_HOLD_ACK' and not prepared:
     if d['plant']['integration_steps']!=0:raise RuntimeError('UNAUTHORIZED_INTEGRATION')
     prepared=True;write(out/'PREPARED_HOLD.json',d)
     if a.stage=='held-readiness':status='HELD_READY_NOT_PHYSICS_VALIDATED';break
     # The only arm-permit site. Unreachable in the default held-readiness stage.
     permit=launch('manual_permit',['/opt/ros/noetic/bin/rostopic','pub','-1','/g4c/start_permit','std_msgs/String',json.dumps(dict(data=nonce))])
     armed_at=time.monotonic()
    if armed_at is not None and d['mode']=='ARMED_HOLD' and time.monotonic()-armed_at>2:
     if d['plant']['integration_steps']<500:raise RuntimeError('INSUFFICIENT_REAL_FEEDBACK')
     status='SHORT_HOVER_OBSERVED_NOT_FAULT_CERTIFIED';break
   time.sleep(.01)
  else:raise RuntimeError('CONTENT_READINESS_TIMEOUT')
 except BaseException as e:reason=repr(e);raise
 finally:
  # Stop the owned physics process FIRST;stopping publishers is not braking.
  plant=[x for x in owned if x[0].name=='gated_plant']
  others=[x for x in reversed(owned) if x[0].name!='gated_plant']
  for folder,proc,so,se,start in plant+others:
   if proc.poll() is None:
    try:os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=proc.wait(timeout=1.)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   so.close();se.close();write(folder/'PROCESS_RESULT.json',dict(pid=proc.pid,actual_exit_code=code,elapsed_s=time.monotonic()-start,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log')))
  write(out/'RESULT.json',dict(status=status,reason=reason,stage=a.stage,source_binding=binding,history=history,
   physical_execution_requested=a.stage=='armed-hover',terminal='OWNED_PROCESSES_TERMINATED',aircraft_braking_certification=False,network_updates=0))
if __name__=='__main__':main()
