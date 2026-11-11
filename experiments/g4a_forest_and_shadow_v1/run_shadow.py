"""Owned isolated ROS master + native forest sensor + non-control shadow test."""
from pathlib import Path
import sys,json,os,subprocess,signal,time,socket,xmlrpc.client,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];OWNER=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30);p.add_argument('--port',type=int,default=11341);args=p.parse_args()
 out=Path(args.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.mkdir();build=json.loads((OWNER/'ros_build/BUILD_IDENTITY.json').read_text())
 for path,h in dict(build['sources'],**{build['binary']:build['sha256'],build['config']:build['config_sha256'],build['cloud']:build['cloud_sha256']}).items():
  if sha(path)!=h:raise ValueError('NATIVE_IDENTITY_DRIFT '+path)
 with socket.socket() as s:
  s.bind(('127.0.0.1',args.port))
 env=dict(os.environ,CMAKE_PREFIX_PATH='/opt/ros/noetic',ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],ROS_MASTER_URI='http://127.0.0.1:'+str(args.port),ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),
  ROS_PACKAGE_PATH='/opt/ros/noetic/share:'+str(R.parent/'YOPO/Controller/src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
  LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:'+str(R.parent/'YOPO/Simulator/devel/lib'),
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:'+str(R.parent/'YOPO/Controller/devel/lib/python3/dist-packages'),
  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 python='/home/lizheng/anaconda3/envs/yopo/bin/python';owned=[]
 def launch(name,argv):
  folder=out/name;folder.mkdir()
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),environment_contract={k:env[k] for k in ['ROS_MASTER_URI','ROS_IP','ROS_HOME','ROS_LOG_DIR','PYTHONPATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']},started_unix=time.time()))
  stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
  proc=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True);owned.append((name,proc,stdout,stderr,time.monotonic()));return proc
 success=False
 try:
  master=launch('roscore',['/opt/ros/noetic/bin/roscore','-p',str(args.port)])
  rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    if rpc.getPid('/g4a_auditor')[0]==1:break
   except Exception:pass
   if master.poll() is not None:raise RuntimeError('MASTER_EARLY_EXIT')
   time.sleep(.1)
  else:raise RuntimeError('MASTER_NOT_READY')
  time.sleep(1.)
  if master.poll() is not None:raise RuntimeError('MASTER_DIED_DURING_STARTUP')
  native=launch('native_sensor',[build['binary'],'__name:=g4a_native_sensor'])
  pose=json.loads((R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101/pose0/POSE_CERTIFICATE.json').read_text())
  receiver=launch('receiver',[python,'-B','ros/g4a_k1_shadow_v1/receiver.py','--output',str(out/'receiver_data')])
  fixture=launch('odometry_fixture',[python,'-B','ros/g4a_k1_shadow_v1/odometry_fixture.py','--origin']+[str(x) for x in pose['position_world']])
  planner=launch('shadow_planner',[python,'-B','ros/g4a_k1_shadow_v1/shadow_node.py','--output',str(out/'planner'),'--frames',str(args.frames)])
  code=planner.wait(timeout=40)
  receiver_code=receiver.wait(timeout=20)
  if receiver_code:raise RuntimeError('INDEPENDENT_RECEIVER_FAILED')
  state=rpc.getSystemState('/g4a_auditor');types=rpc.getTopicTypes('/g4a_auditor')
  write(out/'LIVE_ROS_GRAPH.json',dict(system_state=state,topic_types=types))
  assert not any('pos_cmd' in row[0] or 'so3_cmd' in row[0] for row in state[2][0]),'CONTROL_PUBLISHER'
  write(out/'RESULT.json',dict(status='PASS' if code==0 else 'FAIL',shadow_exit_code=code,real_native_sensor=True,
   odometry_source='scripted kinematic ROS fixture,not dynamics or aircraft execution',controller_started=False,control_published=False,network_updates=0))
  success=code==0
 except BaseException as error:
  write(out/'STOP.json',dict(error=repr(error),control_published=False));raise
 finally:
  for name,proc,stdout,stderr,started in reversed(owned):
   if proc.poll() is None:
    try:os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=proc.wait(timeout=8)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   stdout.close();stderr.close()
   write(out/name/'PROCESS_RESULT.json',dict(pid=proc.pid,real_exit_code=code,elapsed_s=time.monotonic()-started,shutdown='owned process group SIGINT;SIGKILL only after8s',
     stdout_sha256=sha(out/name/'stdout.log'),stderr_sha256=sha(out/name/'stderr.log')))
 print('SHADOW_ORCHESTRATOR',success,str(out))
 if not success:raise SystemExit(2)
if __name__=='__main__':main()
