import os,sys,time,json,subprocess,signal,threading,socket,math,hashlib,ctypes,xmlrpc.client
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline')
HOST=ROOT/'research_outputs/rsps_forest_shadow_v1/low_speed_host001'
OUT=HOST.parent
PYTHON='/home/lizheng/anaconda3/envs/yopo/bin/python'
RUNNER=ROOT/'experiments/risk_aware_set_physics_v1/low_speed_commands_v1.sh'
# This supervisor owns only newly created descendants. No source/acceptance changes.
ctypes.CDLL(None).prctl(36,1,0,0,0) # subreaper for ROS children escaping process groups
events=(HOST/'HOST_EVENTS.jsonl').open('x',buffering=1)
resources=(HOST/'RESOURCE_TRACE.jsonl').open('x',buffering=1)
trace=(HOST/'MOTION_TRACE.jsonl').open('x',buffering=1)
spawned=[]; owned={}; stop=threading.Event(); safety=[]; handles=[]; latest={}; counts={}; first={}
start=time.monotonic(); shadow=None; shadow_id=None; result={'H0':'NOT_RUN','H1':'NOT_RUN','H2':'NOT_RUN','control':'ISOLATED_REFERENCE_ONLY'}
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
 if name in ('camera','reference'):env['CUDA_VISIBLE_DEVICES']=''
 args=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev-bind','/dev','/dev','--bind',str(HOST),str(HOST),'--chdir',str(cwd)]+cmd if readonly else cmd
 if name=='sensor':
  at=args.index('--chdir');args[at:at]=['--ro-bind',str(OUT/'low_speed_preparation001/sensor.yaml'),'/home/lizheng/YOPO/Simulator/src/config/config.yaml']
 args=['timeout','--signal=INT','--kill-after=5s','150s']+list(map(str,args))
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
for sig in (signal.SIGTERM,signal.SIGINT):
 signal.signal(sig,lambda *_:stop.set())
monitor=threading.Thread(target=windows_monitor,daemon=True)
try:
 s=socket.socket();assert s.connect_ex(('127.0.0.1',11311))!=0,'EXISTING_MASTER';s.close()
 assert memory()['MemAvailable']>=3145728
 assert gpu()['free_mib']>=512
 result['H0']='PASS'
 monitor.start()
 launch('roscore',['/opt/ros/noetic/bin/roscore','-p','11311'])
 import rosgraph
 master=rosgraph.Master('/f2_owned_host_observer')
 for _ in range(100):
  try:master.getPid();break
  except Exception:time.sleep(.1)
 else:raise RuntimeError('MASTER_TIMEOUT')
 launch('controller',['/opt/ros/noetic/bin/roslaunch','/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/launch/simulator_attitude_control.launch'])
 import rospy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import Image
 from quadrotor_msgs.msg import PositionCommand
 rospy.init_node('f2_owned_host_observer',anonymous=False,disable_signals=True)
 subs=[rospy.Subscriber('/sim/odom',Odometry,callback('odom'),queue_size=2),rospy.Subscriber('/depth_image',Image,callback('depth'),queue_size=2),rospy.Subscriber('/so3_control/pos_cmd',PositionCommand,callback('cmd'),queue_size=2)]
 launch('camera',[PYTHON,'-B','-s','-u','-m','query_minco.inference.shadow_camera_low_speed','--host',str(HOST)],ROOT)
 launch('sensor',['/home/lizheng/YOPO/Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda'],Path('/home/lizheng/YOPO/Simulator'))
 lastres=0;lasttrace=0;low=0;swapactive=0;basevm=vmstat()
 def resources_tick():
  global lastres,low,swapactive
  now=time.monotonic()
  if now-lastres>=1:
   m=memory();v=vmstat();g=gpu()
   resources.write(json.dumps(dict(wall_time_ns=time.time_ns(),elapsed_s=now-start,MemAvailable_kb=m['MemAvailable'],SwapUsed_kb=m['SwapTotal']-m['SwapFree'],vmstat=v,gpu=g))+'\n')
   low=low+1 if m['MemAvailable']<3145728 else 0
   swapactive=swapactive+1 if v['pswpout']>basevm['pswpout'] and m['SwapTotal']-m['SwapFree']>65536 else 0
   if low>=1:safety.append('WSL_AVAILABLE_LT_3GIB');stop.set()
   if swapactive>=3:safety.append('SUSTAINED_SWAP');stop.set()
   if v['oom_kill']>basevm['oom_kill']:safety.append('OOM_KILL');stop.set()
   if g['free_mib']<256:safety.append('GPU_FREE_LT_256MIB');stop.set()
   discover();lastres=now
 def wait_for(condition,seconds,label):
  deadline=time.monotonic()+seconds
  while not condition():
   resources_tick()
   if stop.is_set():raise RuntimeError('RESOURCE_STOP_'+label)
   if time.monotonic()>deadline:raise RuntimeError('TIMEOUT_'+label)
   if any(p.poll() is not None for n,p in spawned):raise RuntimeError('COMPONENT_EXIT_'+label)
   motion_row();time.sleep(.05)
 wait_for(lambda:counts.get('depth',0)>2,25,'CALIBRATED_CAMERA')
 assert [rospy.get_param('/quadrotor_simulator_so3/simulator/init_state_'+a) for a in 'xyz']==[0.,0.,2.],'INITIAL_ORIGIN'
 shadow_id='low_speed_shadow001'
 shadow=launch('shadow',['/bin/bash',str(RUNNER),'start',shadow_id],ROOT,readonly=False)
 result['H2']='WARMING';result['shadow_id']=shadow_id
 event('SHADOW_PREWARM_LAUNCHED',shadow_id=shadow_id)
 wait_for(lambda:(OUT/shadow_id/'BRIDGE_READY.json').exists(),20,'FROZEN_CPU_BRIDGE')
 launch('reference',[PYTHON,'-B','-s','-u','-m','query_minco.inference.low_speed_reference','--host',str(HOST)],ROOT)
 dump('RUN_COMMANDS_AND_PIDS.json',dict(wall_time_ns=time.time_ns(),goal=[30,0,2],profile='simulator_attitude_control.launch',processes=[dict(name=n,**proc(p.pid)) for n,p in spawned]))
 decision=None;collection_stopped=False
 while time.monotonic()-start<130 and not stop.is_set():
  now=time.monotonic();resources_tick()
  if now-lasttrace>=.1:r=motion_row();lasttrace=now
  if any(p.poll() is not None for n,p in spawned if n not in ('shadow','reference')):raise RuntimeError('NATIVE_COMPONENT_EXIT')
  reference_start=HOST/'REFERENCE_START.json'
  if reference_start.exists():
   refstamp=json.loads(reference_start.read_text())['stamp'];reft=rospy.Time.now().to_sec()-refstamp
   if decision is None and reft>=14.2:
    from experiments.risk_aware_set_physics_v1.low_speed_support_v1 import replay
    import numpy as np
    unique=np.load(OUT/'low_speed_review001/TRAINING_STATES.npz')['unique_states']
    rows=[r for r in replay(OUT/shadow_id,unique) if 8<=r['stamp_ns']/1e9-refstamp<=14]
    ev=[json.loads(l) for l in (OUT/shadow_id/'events.jsonl').read_text().splitlines()]
    fr=[e for e in ev if e['type']=='FRAME' and 8<=e['stamp']-refstamp<=14]
    gaps=np.diff([e['stamp_ns'] for e in fr])/1e9
    near=sum(r.get('near_joint_reference',False) for r in rows)/max(len(rows),1)
    direction=sum(r['direction'] for r in rows)/max(len(rows),1)
    maxgap=float(max(gaps)) if len(gaps) else None
    passed=bool(len(fr)>=25 and near>=.8 and direction<=.05 and maxgap is not None and maxgap<=.3)
    decision=dict(passed=passed,pilot_attempts=len(rows),accepted_frames=len(fr),near_joint_reference_fraction=near,
       exact_joint_hits=sum(r.get('exact',False) for r in rows),goal_direction_reject_fraction=direction,max_gap_s=maxgap,
       full_start_stamp=rospy.Time.now().to_sec() if passed else None,reference_start_stamp=refstamp,
       classification='CONTROLLED_REFERENCE_SENSOR_INTEGRATION_NOT_YOPO_AUTONOMY')
    dump('PILOT_DECISION.json',decision);event('PILOT_DECISION',decision=decision)
    if not passed:
     (OUT/shadow_id/'STOP_COLLECTION').touch(exist_ok=False);(HOST/'REFERENCE_STOP').touch(exist_ok=False);collection_stopped=True
   if decision and decision['passed'] and not collection_stopped and rospy.Time.now().to_sec()-decision['full_start_stamp']>=11.2:
    (OUT/shadow_id/'STOP_COLLECTION').touch(exist_ok=False);(HOST/'REFERENCE_STOP').touch(exist_ok=False);collection_stopped=True
   if collection_stopped and (HOST/'REFERENCE_EXIT.json').exists() and shadow.poll() is not None:
    result['shadow_launcher_returncode']=shadow.returncode;result['H2']='LOW_SPEED_OBSERVATION_FINISHED';break
  if result['H1']!='PASS':
   o=latest.get('odom');d=latest.get('depth');c=latest.get('cmd')
   ready=(o is not None and d is not None and c is not None and c.trajectory_flag==0)
   if ready:
    ready=all(0<=rospy.Time.now().to_sec()-x.header.stamp.to_sec()<=.25 for x in (o,d,c))
    ready=ready and math.sqrt(sum(v*v for v in [o.twist.twist.linear.x,o.twist.twist.linear.y,o.twist.twist.linear.z]))>.01
   if ready:
    pubs,subscribers,services=master.getSystemState();ps=dict(pubs)
    assert ps.get('/so3_control/pos_cmd')==['/low_speed_reference'],'CONTROLLER_SOURCE'
    assert ps.get('/sim/odom')==['/quadrotor_simulator_so3'],'ODOM_SOURCE'
    assert ps.get('/forest30/native_depth')==['/sensor_simulator_node'],'NATIVE_DEPTH_SOURCE'
    assert ps.get('/depth_image')==['/forest30_camera_lattice'],'YOPO_CAMERA_SOURCE'
    assert ps.get('/query_shadow/depth')==['/forest30_camera_lattice'],'QUERY_CAMERA_SOURCE'
    relay={t for t,ns in pubs if '/forest30_camera_lattice' in ns}
    assert relay<={'/depth_image','/camera_info','/query_shadow/depth','/query_shadow/camera_info','/rosout'},'RELAY_CONTROL_PUBLICATION'
    assert memory()['MemAvailable']>=3145728
    result['H1']='PASS';result['H2']='RUNNING'
    dump('H1_QUALIFICATION.json',dict(wall_time_ns=time.time_ns(),first_messages=first,counts=counts.copy(),graph=dict(publishers=pubs,subscribers=subscribers,services=services),memory=memory(),gpu=gpu(),motion=motion_row()))
    event('H1_REFERENCE_MOTION_OBSERVED')
  if shadow.poll() is not None and not collection_stopped:
   result['shadow_launcher_returncode']=shadow.returncode;result['H2']='START_EXITED'
   event('SHADOW_EXIT',returncode=shadow.returncode);break
  time.sleep(.02)
 if stop.is_set():result['stop_reason']=safety or ['SIGNAL']
except Exception:
 import traceback
 result['exception']=traceback.format_exc();event('EXCEPTION',traceback=result['exception'])
finally:
 stop.set()
 if monitor.is_alive():monitor.join(timeout=9)
 if shadow and shadow.poll() is None:
  discover();safe_signal(shadow.pid,signal.SIGINT)
  try:shadow.wait(timeout=10)
  except subprocess.TimeoutExpired:pass
 cleanup()
 if result['H2'] in ('WARMING','RUNNING'):result['H2']='ABORTED'
 result['elapsed_s']=time.monotonic()-start;result['safety_reasons']=safety
 dump('HOST_RESULT.json',result);event('END',result=result)
 for f in handles:f.close()
 trace.close();resources.close();events.close()
print(json.dumps(result,indent=2))
