"""Real nodelet consumption/fault tests. Deliberately never launches a physics process."""
from pathlib import Path
import sys,json,os,time,subprocess,signal,socket,xmlrpc.client,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import write,sha
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',default='timeout,depth_loss,odom_delay,nan,planner_stop,controller_fault');args=p.parse_args()
 out=Path(args.output).resolve()
 if (R/'research_outputs').resolve() not in out.parents:raise ValueError('OUTPUT_ROOT')
 out.mkdir();port=11342
 sources=list((R/'ros/g4b_control_supervisor_v1').glob('*'))+list((R/'query_minco/integration/yopo_control_supervisor_v1').glob('*.py'))+[Path(__file__),R.parent/'YOPO/Controller/devel/lib/libso3_control_nodelet.so']
 write(out/'SOURCE_BINDING.json',{str(p):sha(p) for p in sources if p.is_file()})
 with socket.socket() as s:s.bind(('127.0.0.1',port))
 env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_IP='127.0.0.1',ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',
  ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'),CMAKE_PREFIX_PATH=str(R.parent/'YOPO/Controller/devel')+':/opt/ros/noetic',
  ROS_PACKAGE_PATH=str(R.parent/'YOPO/Controller/src')+':/opt/ros/noetic/share',LD_LIBRARY_PATH=str(R.parent/'YOPO/Controller/devel/lib')+':/opt/ros/noetic/lib',
  PYTHONPATH=str(R)+':/opt/ros/noetic/lib/python3/dist-packages:'+str(R.parent/'YOPO/Controller/devel/lib/python3/dist-packages'),
  PYTHONDONTWRITEBYTECODE='1',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 py='/home/lizheng/anaconda3/envs/yopo/bin/python';owned=[];results=[]
 def launch(folder,argv):
  folder.mkdir(parents=True)
  write(folder/'PROCESS_START.json',dict(argv=argv,cwd=str(R),environment={k:env[k] for k in ['ROS_MASTER_URI','CMAKE_PREFIX_PATH','ROS_PACKAGE_PATH','ROS_HOME','ROS_LOG_DIR','PYTHONPATH','LD_LIBRARY_PATH']},started_unix=time.time()))
  stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb');proc=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  rec=(folder,proc,stdout,stderr,time.monotonic());owned.append(rec);return rec
 def cleanup(items):
  for folder,proc,stdout,stderr,start in reversed(items):
   if proc.poll() is None:
    try:os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:pass
   try:code=proc.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   stdout.close();stderr.close();write(folder/'PROCESS_RESULT.json',dict(pid=proc.pid,actual_exit_code=code,elapsed_s=time.monotonic()-start,stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log')))
 master=launch(out/'roscore',['/opt/ros/noetic/bin/roscore','-p',str(port)])
 try:
  rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(100):
   try:
    if rpc.getPid('/g4b_audit')[0]==1:break
   except Exception:pass
   time.sleep(.1)
  else:raise RuntimeError('MASTER_UNAVAILABLE')
  time.sleep(1)
  for index,case in enumerate(args.cases.split(',')):
   if case not in ['timeout','depth_loss','odom_delay','nan','planner_stop','controller_fault']:raise ValueError('CASE')
   folder=out/case;folder.mkdir();prefix='/g4b/case'+str(index);baseline=len(owned)
   ctrl=launch(folder/'controller',['/opt/ros/noetic/bin/roslaunch',str(R/'ros/g4b_control_supervisor_v1/controller_only.launch'),'prefix:='+prefix])
   supervisor=launch(folder/'supervisor',[py,'-B','ros/g4b_control_supervisor_v1/supervisor_node.py','--prefix',prefix,'--output',str(folder/'supervisor_records'),'--message-test-profile'])
   fault_at=time.monotonic()+3.
   request=launch(folder/'planner',[py,'-B','ros/g4b_control_supervisor_v1/request_fixture.py','--prefix',prefix,'--fault',case,'--fault-at',str(fault_at)])
   probe=launch(folder/'probe',[py,'-B','ros/g4b_control_supervisor_v1/consumption_probe.py','--prefix',prefix,'--fault',case,'--fault-at',str(fault_at),'--output',str(folder/'probe_records'),'--controller-pgid',str(ctrl[1].pid),'--planner-pgid',str(request[1].pid)])
   code=probe[1].wait(timeout=20);rp=folder/'probe_records/RESULT.json'
   results.append(dict(case=case,exit_code=code,result=json.loads(rp.read_text()) if rp.exists() else None))
   write(folder/'LIVE_ROS_GRAPH.json',dict(system_state=rpc.getSystemState('/g4b_audit'),topic_types=rpc.getTopicTypes('/g4b_audit')))
   cleanup(owned[baseline:]);del owned[baseline:]
   print('CASE_COMPLETE',case,code,flush=True)
 finally:cleanup(owned)
 status='PASS' if all(x['exit_code']==0 and x['result']['status']=='PASS' for x in results) and len(results)==len(args.cases.split(',')) else 'FAIL'
 write(out/'RESULT.json',dict(status=status,cases=results,live_controller_consumption=status,no_quadrotor_physics_started=True,physical_braking='NOT_RUN'))
 print('CONSUMPTION_RESULT',status)
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
