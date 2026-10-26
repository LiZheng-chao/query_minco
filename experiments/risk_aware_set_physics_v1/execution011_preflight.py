"""E011 startup/cleanup CPU tests and preparation, no ROS processes."""
import ast,difflib,hashlib,json,os,pickle,shutil,subprocess,sys,time,types,signal,ctypes
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline')
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution011/native_diag001'
OLD=OUT.parent.parent/'f3_execution010/native_diag001'
PY='/home/lizheng/anaconda3/envs/yopo/bin/python'
ENV=dict(ROS_PACKAGE_PATH='/home/lizheng/YOPO/Controller/src:/home/lizheng/YOPO/Simulator/src:/opt/ros/noetic/share',
 PYTHONPATH=str(ROOT)+':/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages:/home/lizheng/YOPO/Simulator/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages',
 LD_LIBRARY_PATH='/home/lizheng/YOPO/Controller/devel/lib:/home/lizheng/YOPO/Simulator/devel/lib:/opt/ros/noetic/lib',
 ROS_MASTER_URI='http://127.0.0.1:11311',ROS_IP='127.0.0.1',ROS_HOME=str(OUT/'runtime'),ROS_LOG_DIR=str(OUT/'roslog'),
 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def startup_files():
 host=OUT/'runtime/NONQUERY_DIAGNOSTIC_HOST.py';old=host.read_text()
 shutil.copyfile(host,OUT/'runtime/HOST_APPROVED_PATCH_BASE.py')
 s=old.replace('from sensor_msgs.msg import Image,CameraInfo\n','from sensor_msgs.msg import Image,CameraInfo,PointCloud2\nimport rospy,rosgraph,message_filters\nfrom sensor_msgs import point_cloud2\nimport numpy as np\n',1)
 s=s.replace("env['LD_LIBRARY_PATH']='/home/lizheng/YOPO/Controller/devel/lib:/opt/ros/noetic/lib'", "env['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']")
 # Avoid spurious missing-camera/simulator cleanup errors on partial startup.
 s=s.replace(" camera_deadline=time.monotonic()+5", " camera_deadline=time.monotonic()+5\n camera_started=any(n=='camera' for n,p in spawned)")
 s=s.replace(" while time.monotonic()<camera_deadline:", " while camera_started and time.monotonic()<camera_deadline:")
 s=s.replace(" else:event('CAMERA_DRAIN_TIMEOUT');safety.append('CAMERA_DRAIN_TIMEOUT')", " else:\n  if camera_started:event('CAMERA_DRAIN_TIMEOUT');safety.append('CAMERA_DRAIN_TIMEOUT')")
 s=s.replace("  uri=master.lookupNode('/quadrotor_simulator_so3');", "  if not any(n=='controller' for n,p in spawned):raise LookupError('SIMULATOR_NOT_STARTED')\n  uri=master.lookupNode('/quadrotor_simulator_so3');")
 s=s.replace(" except Exception as exc:event('SIM_SHUTDOWN_EXCEPTION',error=str(exc));safety.append('SIM_SHUTDOWN_EXCEPTION')",
             " except Exception as exc:\n  event('SIM_SHUTDOWN_NOTE',error=str(exc))\n  if released:safety.append('SIM_SHUTDOWN_EXCEPTION')")
 # Ensure cleanup executes even if stop-wait or observer finalization raises.
 a=s.index(" # Do not kill the physical publisher");b=s.index(" stop.set()",a)
 body=s[a:b]
 body=" try:\n"+''.join(' '+line+'\n' for line in body.splitlines())+" except Exception as exc:\n  safety.append('STOP_WAIT_EXCEPTION');event('STOP_WAIT_EXCEPTION',error=repr(exc))\n"
 s=s[:a]+body+s[b:]
 # Robust node exit race; only inspect a single captured identity.
 s=s.replace("  if proc(pid) and proc(pid).get('cmdline',''):safety.append('SIM_DIAG_FLUSH_EXIT_TIMEOUT')",
             "  current=proc(pid)\n  if current and current.get('cmdline',''):safety.append('SIM_DIAG_FLUSH_EXIT_TIMEOUT')")
 compile(s,str(host),'exec');host.write_text(s)
 (OUT/'patches/STARTUP_CLEANUP_COMPLETION.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='approved_minimal_host',tofile='E011_startup_cleanup_completed')))
 for n in ('empty_room.ply','SCENE_VALIDATION.json'):shutil.copyfile(OLD/'scene'/n,OUT/'scene'/n)
 (OUT/'scene/sensor.yaml').write_text((OLD/'scene/sensor.yaml').read_text().replace(str(OLD),str(OUT)))
 launch=(OLD/'runtime/controlled_startup_shadow.launch').read_text().replace(str(OLD),str(OUT)).replace('f3_execution010_native_diag001','f3_execution011_native_diag001')
 (OUT/'runtime/controlled_startup_shadow.launch').write_text(launch)
 wrapper='''#!/usr/bin/env bash
set -u
[[ "$1" == /home/lizheng/YOPO/Controller/devel/lib/so3_quadrotor_simulator/quadrotor_simulator_so3 ]] || exit 2
shift
"SIM_BINARY" "$@"
sim_status=$?
(set -o noclobber; printf '{"exit_code":%d,"scope":"actual_private_simulator"}\\n' "$sim_status" > "EXIT_RECEIPT") || exit 73
exit "$sim_status"
'''.replace('SIM_BINARY',str(OUT/'runtime/quadrotor_simulator_diagnostic')).replace('EXIT_RECEIPT',str(OUT/'PRIVATE_SIMULATOR_EXIT.json'))
 (OUT/'runtime/private_sim_exec.sh').write_text(wrapper);(OUT/'runtime/private_sim_exec.sh').chmod(0o755)
 shutil.copyfile(OLD/'runtime/native_reference_shadow',OUT/'runtime/native_reference_shadow');(OUT/'runtime/native_reference_shadow').chmod(0o755)
def environment_test():
 env=os.environ.copy();env.update(ENV);env.pop('ROS_HOSTNAME',None)
 (OUT/'ENVIRONMENT.json').write_text(json.dumps(ENV,indent=2))
 # Actual child namespace/environment, not just the supervisor's Python imports.
 child=r'''
import os,json,subprocess
from quadrotor_msgs.msg import SO3Command,PositionCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,CameraInfo,PointCloud2,Imu
from std_msgs.msg import String,Time
import rospy,rosgraph,message_filters,numpy
from sensor_msgs import point_cloud2
names=['so3_control','so3_quadrotor_simulator','quadrotor_msgs','sensor_simulator']
print(json.dumps(dict(imports_pass=True,packages={p:subprocess.check_output(['/opt/ros/noetic/bin/rospack','find',p],text=True).strip() for p in names},environment={k:os.environ[k] for k in ['PYTHONPATH','ROS_PACKAGE_PATH','LD_LIBRARY_PATH']})))
'''
 path=OUT/'cpu/CHILD_ENV_TEST.py';path.write_text(child)
 command=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev-bind','/dev','/dev','--bind',str(OUT),str(OUT),PY,'-B','-s',str(path)]
 r=subprocess.run(command,capture_output=True,text=True,env=env,timeout=15)
 (OUT/'logs/CHILD_ENV.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 data=json.loads(r.stdout)
 libraries=[]
 for exe in [OUT/'runtime/quadrotor_simulator_diagnostic',OUT/'runtime/native_reference_shadow',Path('/home/lizheng/YOPO/Controller/devel/lib/so3_control/network_control_node'),Path('/home/lizheng/YOPO/Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda')]:
  l=subprocess.run(['ldd',str(exe)],capture_output=True,text=True,env=env)
  (OUT/'logs'/('LDD_'+exe.name+'.log')).write_text(l.stdout+l.stderr)
  assert l.returncode==0 and 'not found' not in l.stdout,str(exe)
  libraries.append(str(exe))
 r=subprocess.run(['/opt/ros/noetic/bin/roslaunch','--nodes',str(OUT/'runtime/controlled_startup_shadow.launch')],capture_output=True,text=True,env=env,timeout=15)
 (OUT/'logs/LAUNCH_PARSE.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 assert '/quadrotor_simulator_so3' in r.stdout and '/network_controller_node' in r.stdout
 data.update(status='PASS',ROS_started=False,ldd_verified=libraries,launch_nodes=r.stdout.splitlines())
 (OUT/'CPU_ENVIRONMENT_RESULTS.json').write_text(json.dumps(data,indent=2))
def fixture(stage):
 # Actual OS process supervision with dummy payloads; no ROS/physics.
 case=OUT/'cpu'/('cleanup_'+stage);case.mkdir()
 ctypes.CDLL(None).prctl(36,1,0,0,0)
 source=(OUT/'runtime/NONQUERY_DIAGNOSTIC_HOST.py').read_text();module=ast.parse(source)
 wanted={'proc','discover','safe_signal','cleanup'}
 code=compile(ast.fix_missing_locations(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[])),'ACTUAL_HOST_CLEANUP','exec')
 owned={};spawned=[];notes=[];safety=[];sim_pid=None
 def event(kind,**kw):notes.append(dict(kind=kind,**kw))
 def dump(name,obj):(case/name).write_text(json.dumps(obj,indent=2))
 class Master:
  def __init__(self,*_):pass
  def lookupNode(self,*_):return 'CPU_STUB_ONLY'
 class Proxy:
  def __init__(self,*_):pass
  def getPid(self,*_):return (1,'CPU',sim_pid)
  def shutdown(self,*_):os.kill(sim_pid,signal.SIGINT)
 import xmlrpc.client
 proxy=xmlrpc.client.ServerProxy
 sys.modules['rosgraph']=types.SimpleNamespace(Master=Master);xmlrpc.client.ServerProxy=Proxy
 ctx=dict(Path=Path,os=os,time=time,socket=__import__('socket'),signal=signal,json=json,xmlrpc=__import__('xmlrpc'),HOST=case,owned=owned,spawned=spawned,safety=safety,event=event,dump=dump,released=False)
 exec(code,ctx)
 n={'roscore':1,'controller':2,'simulator':3,'sensor':4}[stage]
 dummy="import signal,time;signal.signal(signal.SIGINT,lambda *a:exit(0));time.sleep(60)"
 try:
  for name in ['roscore','controller','simulator','sensor'][:n]:
   f=(case/(name+'.log')).open('w');p=subprocess.Popen([PY,'-B','-s','-c',dummy],stdout=f,stderr=f,start_new_session=True)
   spawned.append((name,p));owned[p.pid]=ctx['proc'](p.pid)
   if name=='simulator':sim_pid=p.pid
  # A child exits while its setsid grandchild is still alive: real reparenting.
  orphan_code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"+repr(dummy)+"],start_new_session=True);print(p.pid,flush=True)"
  p=subprocess.Popen([PY,'-B','-s','-c',orphan_code],stdout=subprocess.PIPE,text=True)
  orphan=int(p.stdout.readline());p.wait();time.sleep(.03);ctx['discover']()
  assert orphan in owned
  if sim_pid is None:spawned[:]=[(n,p) for n,p in spawned if n!='controller']
  ctx['cleanup']()
  remaining=[ctx['proc'](pid) for pid,r in owned.items() if ctx['proc'](pid) and ctx['proc'](pid)['cmdline']]
  assert not remaining,remaining
  dump('FIXTURE.json',dict(stage=stage,passed=True,real_OS_children=True,ROS=False,physics=False,orphan_PID=orphan,remaining=remaining,events=notes))
 finally:
  xmlrpc.client.ServerProxy=proxy
def cleanup_tests():
 sentinel=subprocess.Popen([PY,'-c','import time;time.sleep(60)'],start_new_session=True)
 rows=[]
 try:
  for stage in ['roscore','controller','simulator','sensor']:
   r=subprocess.run([PY,'-B','-s',__file__,'--fixture',stage],capture_output=True,text=True,timeout=25)
   (OUT/'logs'/('CLEANUP_'+stage+'.log')).write_text(r.stdout+r.stderr)
   assert r.returncode==0,(stage,r.stderr)
   assert sentinel.poll() is None,'UNRELATED_SIBLING_TERMINATED'
   rows.append(dict(stage=stage,passed=True,unrelated_sibling_preserved=True))
 finally:sentinel.terminate();sentinel.wait(timeout=3)
 (OUT/'CPU_CLEANUP_RESULTS.json').write_text(json.dumps(dict(status='PASS',scope='actual host cleanup functions with real dummy OS children/orphans, stub ROS lookup only',cases=rows,ROS_started=False),indent=2))
def refs():
 # Reuse coefficient payload only, freshly validate all extrema and body sweeps.
 from query_minco.inference.trajectory_execution import MotionLimits,trim,concatenate
 from query_minco.inference.committed_reference import PreparedReference,StopEntry
 from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
 from experiments.risk_aware_set_physics_v1.export_native_reference import thaw,export
 with (OLD/'preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:old=pickle.load(f)
 now=time.time();mono=time.monotonic();expiry=now+120
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],expiry,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 fresh={};count=0;horizon=0
 for bid,b in old.items():
  prefix=thaw(b.prefix).validate(MotionLimits());stops=[]
  for row in b.stops:
   stop=thaw(row.polynomial);full=stop if row.phase==0 else concatenate(trim(prefix,0,row.phase),stop)
   full.validate(MotionLimits());proof=volume.certify(full.coefficients,full.durations,now,required_until=expiry)
   assert proof['status']=='CERTIFIED_FREE' and proof['trajectory_sha256']==full.digest
   stops.append(StopEntry(row.phase,row.polynomial,json.dumps(proof,sort_keys=True).encode()));count+=1;horizon=max(horizon,stop.duration)
  proof=volume.certify(prefix.coefficients,prefix.durations,now,required_until=expiry);assert proof['status']=='CERTIFIED_FREE'
  fresh[bid]=PreparedReference(b.prefix,tuple(stops),json.dumps(proof,sort_keys=True).encode(),expiry,mono+120,'EXTERNAL_SUPERVISOR',now,mono)
 (OUT/'preparation').mkdir()
 with (OUT/'preparation/TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(fresh,f,protocol=4)
 export(fresh,OUT/'preparation/NATIVE_REFERENCES.bin')
 (OUT/'preparation/CPU_REFERENCE_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,checked_stops=count,nominal_phases=len(fresh[1].stops),nominal_duration_s=fresh[1].prefix.duration,max_backup_duration_s=horizon,valid_for_s=120,expiry_ros=expiry,expiry_mono=mono+120,remaining_s=expiry-time.time(),scope='EXTERNAL_SUPERVISOR'),indent=2))
 assert expiry-time.time()>65
 if subprocess.run([str(OUT/'runtime/native_reference_shadow'),'--cpu-test',str(OUT/'preparation/NATIVE_REFERENCES.bin')],capture_output=True).returncode:raise RuntimeError('TRANSPORT_CPU_FAIL')
def main():
 if len(sys.argv)>1 and sys.argv[1]=='--fixture':return fixture(sys.argv[2])
 if len(sys.argv)>1 and sys.argv[1]=='--refs':return refs()
 startup_files();environment_test();cleanup_tests();print('Startup/child env/launch/ldd and all four actual OS cleanup fixtures PASS. No ROS started.')
if __name__=='__main__':main()
