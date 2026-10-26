import os,sys,time,json,subprocess,signal,threading,socket,math,hashlib,ctypes,xmlrpc.client
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline')
HOST=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution003/native_ref001'
HOST.mkdir()
for name in ['logs','runtime','roslog']:(HOST/name).mkdir()
OUT=HOST.parent
PYTHON='/home/lizheng/anaconda3/envs/yopo/bin/python'
RUNNER=ROOT/'experiments/risk_aware_set_physics_v1/low_speed_commands_v1.sh'
# This supervisor owns only newly created descendants. No source/acceptance changes.
ctypes.CDLL(None).prctl(36,1,0,0,0) # subreaper for ROS children escaping process groups
events=(HOST/'HOST_EVENTS.jsonl').open('x',buffering=1)
resources=(HOST/'RESOURCE_TRACE.jsonl').open('x',buffering=1)
trace=(HOST/'MOTION_TRACE.jsonl').open('x',buffering=1)
spawned=[]; owned={}; stop=threading.Event(); safety=[]; handles=[]; latest={}; counts={}; first={}
start=time.monotonic(); shadow=None; shadow_id=None; result={'kind':'NATIVE_NON_QUERY_REFERENCE_CONTROLLER_BENCH','status':'NOT_RUN','Query_control':False}
def event(kind,**kw):
 events.write(json.dumps(dict(type=kind,wall_time_ns=time.time_ns(),elapsed_s=time.monotonic()-start,**kw))+'\n')
def dump(name,obj):
 with (HOST/name).open('x') as f:json.dump(obj,f,indent=2)
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  return dict(pid=pid,ppid=int(stat[1]),pgid=int(stat[2]),starttime=stat[19],cmdline=cmd)
 except (OSError,ValueError,IndexError):return None
def discover():
 rows={int(p.name):proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
 todo={os.getpid()}
 for _ in range(30):
  new={pid for pid,r in rows.items() if r and r['ppid'] in todo}
  if not new-todo:break
  todo |= new
 for pid in todo-{os.getpid()}:
  if rows.get(pid):owned[pid]=rows[pid]
def safe_signal(pid,sig):
 r=proc(pid); old=owned.get(pid)
 if r and old and r['starttime']==old['starttime']:
  event('OWNED_SIGNAL',signal=int(sig),**r)
  try:os.kill(pid,sig)
  except ProcessLookupError:pass
def launch(name,cmd,cwd=HOST,readonly=True):
 env=os.environ.copy()
 if name in ('camera','arbiter'):env['CUDA_VISIBLE_DEVICES']=''
 args=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev-bind','/dev','/dev','--bind',str(HOST),str(HOST),'--chdir',str(cwd)]+cmd if readonly else cmd
 if name=='sensor':
  at=args.index('--chdir');args[at:at]=['--ro-bind',str(HOST.parent/'scene/sensor.yaml'),'/home/lizheng/YOPO/Simulator/src/config/config.yaml']
 args=['timeout','--signal=INT','--kill-after=5s','85s']+list(map(str,args))
 f=(HOST/'logs'/f'{name}.log').open('xb');handles.append(f)
 p=subprocess.Popen(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 row=proc(p.pid); owned[p.pid]=row
 spawned.append((name,p));event('SPAWN',name=name,argv=args,identity=row)
 return p
def memory():
 return {k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
def vmstat():
 return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout','oom_kill')}
def gpu():
 s=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=4)
 used,total=map(int,s.strip().split(','));return dict(used_mib=used,total_mib=total,free_mib=total-used)
def windows_monitor():
 low=0
 path='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
 while not stop.is_set():
  try:
   text=subprocess.check_output([path,'-NoProfile','-NonInteractive','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],timeout=8).decode().strip()
   k=int(text)
   event('WINDOWS_AVAILABLE',available_kb=k)
   low=low+1 if k<1048576 else 0
   if low>=1:
    safety.append('WINDOWS_AVAILABLE_LT_1GIB');stop.set()
  except Exception as e:event('WINDOWS_MONITOR_ERROR',error=str(e))
  stop.wait(1)
def callback(key):
 def cb(m):
  counts[key]=counts.get(key,0)+1; latest[key]=m
  first.setdefault(key,time.time_ns())
 return cb
def motion_row():
 row=dict(wall_time_ns=time.time_ns(),elapsed_s=time.monotonic()-start,counts=counts.copy())
 o=latest.get('odom');d=latest.get('depth');c=latest.get('cmd')
 if o:
  row['odom']=dict(stamp_ns=o.header.stamp.to_nsec(),p=[o.pose.pose.position.x,o.pose.pose.position.y,o.pose.pose.position.z],v=[o.twist.twist.linear.x,o.twist.twist.linear.y,o.twist.twist.linear.z])
 if d:row['depth']=dict(stamp_ns=d.header.stamp.to_nsec(),seq=d.header.seq,shape=[d.height,d.width],encoding=d.encoding)
 if c:row['cmd']=dict(stamp_ns=c.header.stamp.to_nsec(),flag=c.trajectory_flag,p=[c.position.x,c.position.y,c.position.z])
 trace.write(json.dumps(row)+'\n');return row
def cleanup():
 socket.setdefaulttimeout(2)
 event('CLEANUP_BEGIN');discover()
 # Stop the callback relay BEFORE shutting down any ROS publishers/master.
 (HOST/'CAMERA_STOP').touch(exist_ok=True)
 deadline=time.monotonic()+5
 while time.monotonic()<deadline:
  try:
   rows=[json.loads(l) for l in (HOST/'CAMERA_TRACE.jsonl').read_text().splitlines()]
   if rows and rows[-1].get('type')=='EXIT':break
  except (OSError,ValueError):pass
  time.sleep(.05)
 else:event('CAMERA_DRAIN_TIMEOUT')

 # Ask only identified nodes created by this supervisor to shut down.
 try:
  import rosgraph
  master=rosgraph.Master('/f2_owned_host_cleanup')
  pubs,subs,srvs=master.getSystemState()
  names=set(n for _,ns in pubs+subs+srvs for n in ns)
  for n in sorted(names-{'/f2_owned_host_observer'}):
   try:
    proxy=xmlrpc.client.ServerProxy(master.lookupNode(n))
    code,msg,pid=proxy.getPid('/f2_owned_host_cleanup')
    r=proc(pid);old=owned.get(pid)
    if r and old and r['starttime']==old['starttime']:
     event('OWNED_ROS_SHUTDOWN',node=n,identity=r)
     proxy.shutdown('/f2_owned_host_cleanup','bounded owned F2 experiment finished')
   except Exception as e:event('ROS_SHUTDOWN_NOTE',node=n,error=str(e))
 except Exception as e:event('MASTER_CLEANUP_NOTE',error=str(e))
 # Stop the existing runner first, then host supervisor groups and escaped descendants.
 if shadow and shadow.poll() is None:safe_signal(shadow.pid,signal.SIGINT)
 time.sleep(1)
 discover()
 for name,p in reversed(spawned):
  if p.poll() is None:safe_signal(p.pid,signal.SIGINT)
 time.sleep(2);discover()
 for pid in list(owned):safe_signal(pid,signal.SIGTERM)
 time.sleep(2)
 for pid in list(owned):
  if proc(pid):safe_signal(pid,signal.SIGKILL)
 for name,p in spawned:
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:pass
 try:
  while True:
   pid,status=os.waitpid(-1,os.WNOHANG)
   if not pid:break
 except ChildProcessError:pass
 remaining=[proc(pid) for pid,r in owned.items() if proc(pid) and proc(pid)['starttime']==r['starttime']]
 event('CLEANUP_END',remaining=remaining)
 dump('CLEANUP.json',dict(remaining=remaining,owned=list(owned.values())))
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
monitor=threading.Thread(target=windows_monitor,daemon=True)
try:
 s=socket.socket();assert s.connect_ex(('127.0.0.1',11311))!=0,'EXISTING_MASTER';s.close()
 assert memory()['MemAvailable']>=3145728 and gpu()['free_mib']>=512
 w=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],timeout=8).decode().strip()
 assert int(w)>=1048576,'WINDOWS_MEMORY'
 monitor.start()
 launch('roscore',['/opt/ros/noetic/bin/roscore','-p','11311'])
 import rosgraph
 master=rosgraph.Master('/f3_native_observer')
 for _ in range(100):
  try:master.getPid();break
  except Exception:time.sleep(.1)
 else:raise RuntimeError('MASTER_TIMEOUT')
 arbiter=launch('arbiter',[PYTHON,'-B','-s','-u','-m','query_minco.inference.ros_reference_arbiter_v3','--run',str(HOST),'--reference-only'],ROOT)
 deadline=time.monotonic()+12
 while not (HOST/'ARBITER_READY.json').exists():
  if arbiter.poll() is not None:raise RuntimeError('ARBITER_START_EXIT')
  if time.monotonic()>deadline:raise RuntimeError('ARBITER_READY_TIMEOUT')
  time.sleep(.05)
 launch('controller',['/opt/ros/noetic/bin/roslaunch','/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/launch/simulator_attitude_control.launch','init_x:=-1.6','init_y:=0','init_z:=2'])
 import rospy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image,CameraInfo,PointCloud2
 from sensor_msgs import point_cloud2
 import numpy as np
 rospy.init_node('f3_native_observer',anonymous=False,disable_signals=True)
 for topic,typ,key in [('/sim/odom',Odometry,'odom'),('/query_shadow/depth',Image,'depth'),('/query_shadow/camera_info',CameraInfo,'info'),('/mock_map',PointCloud2,'map')]:
  rospy.Subscriber(topic,typ,callback(key),queue_size=2)
 launch('camera',[PYTHON,'-B','-s','-u','-m','query_minco.inference.shadow_camera_low_speed','--host',str(HOST)],ROOT)
 launch('sensor',['/home/lizheng/YOPO/Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda'],Path('/home/lizheng/YOPO/Simulator'))
 lastres=0;swapactive=0;basevm=vmstat();scene_ready=False
 while time.monotonic()-start<70 and not stop.is_set():
  now=time.monotonic()
  if now-lastres>=1:
   mem=memory();vm=vmstat();g=gpu()
   resources.write(json.dumps(dict(wall_time_ns=time.time_ns(),MemAvailable_kb=mem['MemAvailable'],SwapUsed_kb=mem['SwapTotal']-mem['SwapFree'],vmstat=vm,gpu=g))+'\n')
   if mem['MemAvailable']<3145728:safety.append('WSL_AVAILABLE_LT_3GIB');stop.set()
   if g['free_mib']<256:safety.append('GPU_FREE_LT_256MIB');stop.set()
   if vm['oom_kill']>basevm['oom_kill']:safety.append('OOM_KILL');stop.set()
   swapactive=swapactive+1 if vm['pswpout']>basevm['pswpout'] and mem['SwapTotal']-mem['SwapFree']>65536 else 0
   if swapactive>=3:safety.append('SUSTAINED_SWAP');stop.set()
   discover();lastres=now
  if not scene_ready and all(k in latest for k in ['map','depth','info','odom']):
   m=latest['map'];cam=latest['info'];d=latest['depth']
   xyz=np.array(list(point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=False)),np.float32)
   quant=np.rint(xyz*10).astype(np.int32);quant=quant[np.lexsort((quant[:,2],quant[:,1],quant[:,0]))]
   expected=json.loads((HOST.parent/'scene/SCENE_VALIDATION.json').read_text())
   digest=hashlib.sha256(quant.astype('<i4').tobytes()).hexdigest()
   assert digest==expected['quantised_xyz_sha256'] and len(xyz)==expected['point_count'],'LOADED_MAP_NOT_EQUAL'
   assert np.max(np.abs(xyz-quant/10))<1e-5,'MAP_FLOAT_GEOMETRY'
   assert m.header.frame_id=='world'
   assert cam.width==160 and cam.height==128 and np.array_equal(np.reshape(cam.K,(3,3)),[[100,0,79.5],[0,100,63.5],[0,0,1]]),'CAMERA_CONTRACT'
   assert d.width==160 and d.height==128 and d.encoding=='32FC1'
   assert [rospy.get_param('/quadrotor_simulator_so3/simulator/init_state_'+a) for a in 'xyz']==[-1.6,0.,2.]
   ps=master.getSystemState()[0];assert dict(ps).get('/so3_control/pos_cmd')==['/f3_reference_arbiter']
   with (HOST/'LOADED_MAP.npz').open('xb') as f:np.savez_compressed(f,xyz=xyz,quantised=quant)
   dump('SCENE_READY.json',dict(stamp=rospy.Time.now().to_sec(),point_count=len(xyz),loaded_map_quantised_sha256=digest,
    scene_validation=expected,source='EXTERNAL_SUPERVISOR',Query_online_certification=False,publishers=ps,
    camera_K=list(cam.K),depth_encoding=d.encoding,initial_origin=[-1.6,0,2]))
   scene_ready=True
  if (HOST/'ARBITER_RESULT.json').exists():
   result['status']='ARBITER_FINISHED';result['arbiter']=json.loads((HOST/'ARBITER_RESULT.json').read_text());break
  if any(p.poll() is not None for n,p in spawned):raise RuntimeError('NATIVE_COMPONENT_EXIT')
  time.sleep(.05)
 if stop.is_set():result['resource_stop']=safety or ['SIGNAL'];(HOST/'HOST_STOP').touch(exist_ok=True)
except Exception:
 import traceback
 result['status']='FAIL_HOST';result['exception']=traceback.format_exc();event('EXCEPTION',traceback=result['exception'])
finally:
 # Allow only this owned reference arbiter to finish its bounded stop before teardown.
 if 'arbiter' in globals() and arbiter.poll() is None:
  (HOST/'HOST_STOP').touch(exist_ok=True)
  deadline=time.monotonic()+6
  while not (HOST/'ARBITER_RESULT.json').exists() and time.monotonic()<deadline:time.sleep(.05)
 stop.set()
 if monitor.is_alive():monitor.join(timeout=9)
 cleanup()
 result['safety_reasons']=safety;result['elapsed_s']=time.monotonic()-start
 dump('HOST_RESULT.json',result);event('END',result=result)
 for f in handles:f.close()
 trace.close();resources.close();events.close()
print(json.dumps(result,indent=2))
