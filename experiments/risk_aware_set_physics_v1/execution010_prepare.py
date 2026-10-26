"""Prepare exact nonQuery reference/scene and reuse existing bounded host. No ROS."""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import concatenate,trim,MotionLimits
from query_minco.inference.committed_reference import PreparedReference,StopEntry
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from experiments.risk_aware_set_physics_v1.export_native_reference import export,thaw
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution010/native_diag001'
OLD=OUT.parent.parent/'f3_execution008'
def prepare():
 assert json.loads((OUT/'CPU_DIAGNOSTIC_RESULTS.json').read_text())['all_passed']
 assert not (OUT/'STEP_DIAGNOSTIC.csv').exists()
 for name in ('empty_room.ply','SCENE_VALIDATION.json'):shutil.copyfile(OLD/'scene'/name,OUT/'scene'/name)
 (OUT/'scene/sensor.yaml').write_text((OLD/'scene/sensor.yaml').read_text().replace(str(OLD),str(OUT)))
 launch=(OLD/'runtime/controlled_startup_shadow.launch').read_text().replace(str(OLD),str(OUT)).replace('f3_execution008_shadow_inputs001','f3_execution010_native_diag001')
 launch=launch.replace('<param name="startup_hold" value="true"/>','<param name="startup_hold" value="true"/>\n        <param name="step_diagnostic_enabled" value="true"/>')
 (OUT/'runtime/controlled_startup_shadow.launch').write_text(launch)
 script=(OLD/'runtime/private_sim_exec.sh').read_text().replace('/home/lizheng/query_minco_mainline/research_outputs/rsps_f3_low_speed_v1/f3_execution004/runtime/quadrotor_simulator_startup_gate',str(OUT/'runtime/quadrotor_simulator_diagnostic'))
 (OUT/'runtime/private_sim_exec.sh').write_text(script);(OUT/'runtime/private_sim_exec.sh').chmod(0o755)
 publisher=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution005/runtime/native_reference_shadow'
 shutil.copyfile(publisher,OUT/'runtime/native_reference_shadow');(OUT/'runtime/native_reference_shadow').chmod(0o755)
 # Reuse immutable coefficients; independently recompute every proof and extrema.
 with (OLD/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:old=pickle.load(f)
 now=time.time();mono=time.monotonic();expiry=now+120
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],expiry,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 fresh={};checked=0;horizon=0
 for bid,b in old.items():
  prefix=thaw(b.prefix).validate(MotionLimits());entries=[]
  for row in b.stops:
   stop=thaw(row.polynomial);full=stop if row.phase==0 else concatenate(trim(prefix,0,row.phase),stop)
   full.validate(MotionLimits());proof=volume.certify(full.coefficients,full.durations,now,required_until=expiry)
   assert proof['status']=='CERTIFIED_FREE' and proof['scope']=='EXTERNAL_SUPERVISOR'
   assert proof['trajectory_sha256']==full.digest
   entries.append(StopEntry(row.phase,row.polynomial,json.dumps(proof,sort_keys=True).encode()))
   horizon=max(horizon,row.polynomial.duration);checked+=1
  proof=volume.certify(prefix.coefficients,prefix.durations,now,required_until=expiry);assert proof['status']=='CERTIFIED_FREE'
  fresh[bid]=PreparedReference(b.prefix,tuple(entries),json.dumps(proof,sort_keys=True).encode(),expiry,mono+120,'EXTERNAL_SUPERVISOR',now,mono)
 assert len(fresh[1].stops)==881 and fresh[1].prefix.duration==17.6
 (OUT/'preparation').mkdir()
 export(fresh,OUT/'preparation/NATIVE_REFERENCES.bin')
 with (OUT/'preparation/TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(fresh,f,protocol=4)
 receipt=dict(passed=True,checked_stops=checked,nominal_phases=881,prefix_coefficients_unchanged=True,new_proofs_independently_recomputed=True,valid_for_s=120,expiry_ros=expiry,expiry_mono=mono+120,max_backup_duration_s=horizon,nominal_duration_s=17.6,lifetime_remaining_s=expiry-time.time(),scope='EXTERNAL_SUPERVISOR')
 (OUT/'preparation/CPU_REFERENCE_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2))
 assert expiry-time.time()>=65
 return receipt
def render_host():
 s=(OLD/'runtime/E_INPUT_HOST.py').read_text()
 s=s.replace("HOST=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008/shadow_inputs001'","HOST=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution010/native_diag001'")
 s=s.replace('HOST.mkdir()','assert HOST.is_dir()').replace('(HOST/name).mkdir()','(HOST/name).mkdir(exist_ok=True)')
 s=s.replace('OUT=HOST.parent','OUT=HOST').replace("os.environ['QM_DYNAMIC_CONFIG']=str(OUT/'DYNAMIC_CONFIG.json')","")
 s=s.replace("str(OUT/'runtime/native_reference_dynamic_e4')","str(OUT/'runtime/native_reference_shadow')").replace("str(OUT/'shadow_preparation/NATIVE_REFERENCES.bin')","str(OUT/'preparation/NATIVE_REFERENCES.bin')")
 s='\n'.join(line for line in s.splitlines() if not line.startswith(" c1=launch("))+'\n'
 s=s.replace("with (OUT/'WORKER_PID.txt').open('x') as pidfile:pidfile.write(str(nodepid))","")
 begin=s.index("  if scene_ready and not released")
 end=s.index("  if not stress_started",begin)
 s=s[:begin]+"""  if scene_ready and not released and (HOST/'HOLD_READY.json').exists():
   # All heavy components ready while integration is held. Resource thresholds unchanged.
   if not all_components_ready_since:all_components_ready_since=time.monotonic()
   if time.monotonic()-all_components_ready_since>=1.:
    assert memory()['MemAvailable']>=3145728
    assert gpu()['free_mib']>=256
    if not stop.is_set():
     rospy.set_param('/quadrotor_simulator_so3/startup_release_session','f3_execution010_native_diag001')
     event('CONTROLLED_STARTUP_RELEASE_REQUEST',hold_evidence=json.loads((HOST/'HOLD_READY.json').read_text()))
     released=True
"""+s[end:]
 begin=s.index("  if not stress_started and");end=s.index("  if (HOST/'ARBITER_RESULT.json').exists():",begin)
 s=s[:begin]+s[end:] # No C1, no synthetic CPU stress.
 s=s.replace("scene_ready=False;released=False","scene_ready=False;released=False;all_components_ready_since=None")
 s=s.replace("HOST.parent/'scene/sensor.yaml'","HOST/'scene/sensor.yaml'").replace("HOST.parent/'scene/SCENE_VALIDATION.json'","HOST/'scene/SCENE_VALIDATION.json'")
 s=s.replace("from nav_msgs.msg import Odometry","from nav_msgs.msg import Odometry\n from quadrotor_msgs.msg import SO3Command,PositionCommand")
 s=s.replace(" rospy.init_node('f3_native_observer',anonymous=False,disable_signals=True)",""" rospy.init_node('f3_native_observer',anonymous=False,disable_signals=True)
 for topic,typ,key in [('/sim/odom',Odometry,'ODOM'),('/so3_cmd',SO3Command,'SO3'),('/so3_control/pos_cmd',PositionCommand,'COMMAND')]:
  rospy.Subscriber(topic,typ,record_callback(key),queue_size=100,tcp_nodelay=True)
""")
 s=s.replace("start=time.monotonic(); shadow=None;","records={'ODOM':[],'SO3':[],'COMMAND':[]};record_lock=threading.Lock();record_overflow=[]\nstart=time.monotonic(); shadow=None;")
 anchor='def callback(key):'
 recorder="""def record_callback(key):
 def cb(msg):
  receive_wall=time.time();receive_mono=time.monotonic();cpu=time.thread_time()
  row=dict(stamp=msg.header.stamp.to_sec(),seq=int(msg.header.seq),recv_wall=receive_wall,recv_mono=receive_mono,entry_thread_cpu=cpu)
  def vec(prefix,v):
   for k in 'xyz':row[prefix+k]=float(getattr(v,k))
  if key=='ODOM':
   vec('p',msg.pose.pose.position);vec('v',msg.twist.twist.linear);q=msg.pose.pose.orientation
   for k in 'xyzw':row['q'+k]=float(getattr(q,k))
  elif key=='SO3':
   vec('force_',msg.force);q=msg.orientation
   for k in 'xyzw':row['q'+k]=float(getattr(q,k))
   for i in range(3):row['kR'+str(i)]=float(msg.kR[i]);row['kOm'+str(i)]=float(msg.kOm[i])
   row['motors_enabled']=bool(msg.aux.enable_motors)
  else:
   vec('p',msg.position);vec('v',msg.velocity);vec('a',msg.acceleration);row['flag']=int(msg.trajectory_flag)
  row['done_mono']=time.monotonic()
  with record_lock:
   if len(records[key])>=20000:record_overflow.append(key);stop.set()
   else:records[key].append(row)
 return cb
"""
 assert anchor in s;s=s.replace(anchor,recorder+anchor)
 # Keep the actual source/pose in independent observer records; no fabricated measurements.
 s=s.replace("   discover();lastres=now","   pubs=master.getSystemState()[0]\n   event('PUBLISHER_CHECK',publishers=dict(pubs).get('/so3_control/pos_cmd',[]))\n   if dict(pubs).get('/so3_control/pos_cmd')!=['/f3_reference_arbiter']:safety.append('PUBLISHER_NOT_UNIQUE');stop.set()\n   discover();lastres=now")
 # First resource tick may precede publisher registration; arbiter readiness exists before loop.
 s=s.replace("result['resource_stop']=safety or ['SIGNAL'];(HOST/'HOST_STOP').touch(exist_ok=True)","result['resource_stop']=safety or ['SIGNAL'];event('RESOURCE_OR_HOST_STOP',released=released)")
 a=s.index('def cleanup():');b=s.index('for sig in (signal.SIGTERM',a)
 cleanup="""def cleanup():
 socket.setdefaulttimeout(2);event('CLEANUP_BEGIN');discover()
 (HOST/'CAMERA_STOP').touch(exist_ok=True)
 camera_deadline=time.monotonic()+5
 while time.monotonic()<camera_deadline:
  try:
   rows=[json.loads(x) for x in (HOST/'CAMERA_TRACE.jsonl').read_text().splitlines()]
   if rows and rows[-1].get('type')=='EXIT':break
  except (OSError,ValueError):pass
  time.sleep(.05)
 else:event('CAMERA_DRAIN_TIMEOUT');safety.append('CAMERA_DRAIN_TIMEOUT')
 # Stop native sensor once callbacks are drained.
 for name,p in spawned:
  if name=='sensor' and p.poll() is None:safe_signal(p.pid,signal.SIGINT)
 # Simulator loop ends and flushes BEFORE original Controller/Master shutdown.
 try:
  import rosgraph
  master=rosgraph.Master('/f3_diag_cleanup')
  uri=master.lookupNode('/quadrotor_simulator_so3');proxy=xmlrpc.client.ServerProxy(uri)
  code,msg,pid=proxy.getPid('/f3_diag_cleanup');discover()
  row=proc(pid);prior=owned.get(pid)
  if code!=1 or not row or not prior or row['starttime']!=prior['starttime']:raise RuntimeError('SIM_PID_NOT_OWNED')
  event('SIMULATOR_DIAGNOSTIC_SHUTDOWN',identity=row)
  proxy.shutdown('/f3_diag_cleanup','bounded diagnostic complete; finish buffer flush')
  deadline=time.monotonic()+8
  while time.monotonic()<deadline:
   current=proc(pid)
   if current is None or not current.get('cmdline',''):break
   time.sleep(.05)
  if proc(pid) and proc(pid).get('cmdline',''):safety.append('SIM_DIAG_FLUSH_EXIT_TIMEOUT')
 except Exception as exc:event('SIM_SHUTDOWN_EXCEPTION',error=str(exc));safety.append('SIM_SHUTDOWN_EXCEPTION')
 for name,p in reversed(spawned):
  if p.poll() is None:safe_signal(p.pid,signal.SIGINT)
 deadline=time.monotonic()+3
 while time.monotonic()<deadline and any(p.poll() is None for _,p in spawned):time.sleep(.05)
 discover()
 for pid in list(owned):
  row=proc(pid)
  if row and row['cmdline']:safe_signal(pid,signal.SIGTERM)
 time.sleep(.3);discover()
 for pid in list(owned):
  row=proc(pid)
  if row and row['cmdline']:safety.append('OWNED_FORCED_KILL');safe_signal(pid,signal.SIGKILL)
 for name,p in spawned:
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:safety.append('OWNED_WAIT_TIMEOUT')
 exits={name:p.poll() for name,p in spawned}
 try:
  while True:
   pid,status=os.waitpid(-1,os.WNOHANG)
   if not pid:break
 except ChildProcessError:pass
 remaining=[proc(pid) for pid,r in owned.items() if proc(pid) and proc(pid)['starttime']==r['starttime'] and proc(pid)['cmdline']]
 event('CLEANUP_END',remaining=remaining,process_exit=exits)
 dump('CLEANUP.json',dict(remaining=remaining,owned=list(owned.values()),process_exit=exits))
"""
 s=s[:a]+cleanup+s[b:]
 a=s.index(' # Allow only this owned reference arbiter');b=s.index(' stop.set()',a)
 s=s[:a]+""" # Do not kill the physical publisher while it has an existing complete stop.
 # Before release, physics is held. After release, existing core handles tracking/
 # clock/authority failures; resource load is shed before awaiting committed stop.
 if 'arbiter' in globals() and arbiter.poll() is None:
  if stop.is_set():
   (HOST/'CAMERA_STOP').touch(exist_ok=True);discover()
   for name,p in spawned:
    if name=='sensor' and p.poll() is None:safe_signal(p.pid,signal.SIGINT)
  deadline=min(start+70,time.monotonic()+26 if released else time.monotonic()+.5)
  while released and not (HOST/'ARBITER_RESULT.json').exists() and time.monotonic()<deadline:time.sleep(.05)
  if released and not (HOST/'ARBITER_RESULT.json').exists():safety.append('NO_COMPLETE_STOP_RECEIPT')
"""+s[b:]
 # Save independent raw records only after physics and publishers finish.
 s=s.replace(" result['safety_reasons']=safety;", """ import csv
 for key,rows in records.items():
  with (HOST/('OBSERVER_'+key+'.csv')).open('x',newline='') as f:
   if rows:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 dump('OBSERVER_RECEIPT.json',dict(counts={k:len(v) for k,v in records.items()},overflow=record_overflow))
 result['released']=globals().get('released',False);result['C1_inference']=False
 result['safety_reasons']=safety;""")
 s=s.replace("except Exception as e:event('WINDOWS_MONITOR_ERROR',error=str(e))", "except Exception as e:event('WINDOWS_MONITOR_ERROR',error=str(e));safety.append('WINDOWS_MONITOR_LOST');stop.set()")
 s=s.replace(" import csv", " if 'rospy' in globals():rospy.signal_shutdown('bounded diagnostic observer complete')\n import csv")
 s=s.replace("env['ROS_LOG_DIR']=str(HOST/'roslog')","env['ROS_LOG_DIR']=str(HOST/'roslog')\n env['LD_LIBRARY_PATH']='/home/lizheng/YOPO/Controller/devel/lib:/opt/ros/noetic/lib'")
 assert "c1=launch(" not in s and "if not stress_started" not in s and 'c1_proposal_shadow_e4' not in s
 compile(s,'E010_NONQUERY_HOST','exec')
 (OUT/'runtime/NONQUERY_DIAGNOSTIC_HOST.py').write_text(s)
def main():
 render_host();receipt=prepare()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 files=[p for p in (OUT/'runtime').iterdir() if p.is_file() and p.suffix not in ('.o',)]
 files+=list((OUT/'scene').iterdir())+list((OUT/'preparation').iterdir())
 (OUT/'NATIVE_BINDING.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
