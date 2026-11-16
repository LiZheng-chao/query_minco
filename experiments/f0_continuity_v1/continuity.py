"""NONPHYSICAL: actual Supervisor request callback, stub ROS publication, kinematic plant."""
import sys,json,time,ast,types,contextlib,hashlib
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.f0_continuity_v1.authority import BirthAuthority
from query_minco.integration.f0_continuity_v1.worker import Preparation
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.f0_continuity_v1.admission import ContinuousAdmission as Admission
O=Path('research_outputs/f0_continuity_20261116_v1')
# Compile the actual authoritative request callback, not a reimplementation.
tree=ast.parse(Path('ros/g4cp7b_fix2_v1/supervisor.py').read_text())
fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='request')
code=compile(ast.Module(body=[fn],type_ignores=[]),'actual_supervisor_request','exec')
prior=json.loads(Path('research_outputs/t2_acceptance_20261116_v1/replay/RESULT.json').read_text())['rows'][0]
detail=next(x for x in json.loads(Path('research_outputs/t2_usability_20261116_v1/OBSERVABILITY_RECORDS.json').read_text()) if x.get('frame')==0 and x.get('mode')=='single')
pose=detail['camera_pose'];center=np.asarray(pose['p']);target=center+[.28,0,0]
with np.load(prior['path']) as z:measured=z['depth'].copy()
# Warm the sole original solver before timing.
build(center,np.zeros(3),np.zeros(3),pose['q'],target,time.time())
results=[];faults=[]
for label,delay,source in [('recorded_rays',.05,'frozen_measured'),('moving_rays',.10,'analytical_static_plane')]:
 clock=time.time;authority=BirthAuthority(lambda now:BirthGrant(tuple(center),.6,start+10,'SYNTHETIC_FINITE_GRANT_NOT_A_PHYSICAL_CERTIFICATE'),clock)
 pubs=[];logs=[];aborts=[];samples={};last={}
 journal=types.SimpleNamespace(callback=lambda:contextlib.nullcontext(True))
 r=types.SimpleNamespace(mode='ARMED',samples=samples)
 env=dict(a=types.SimpleNamespace(forest=False),journal=journal,mode='RUNNING',json=json,np=np,last=last,forest_bridge=None,gate=Admission(),
  rospy=types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda:types.SimpleNamespace(to_sec=clock),from_sec=lambda x:x)),
  birth_authority=authority,run=types.SimpleNamespace(__truediv__=lambda s,x:None),r=r,
  log=lambda *x:logs.append(x),abort=lambda x:aborts.append(x),
  accept=lambda *x:True,PositionCommand=object,position_command=lambda cls,pva,stamp,yaw,seq:types.SimpleNamespace(pva=pva,stamp=stamp),
  command=types.SimpleNamespace(publish=lambda msg:pubs.append(dict(stamp=msg.stamp,pva=msg.pva.tolist()))))
 # Real certificate file-existence branch; file itself is not a synthetic production certificate.
 class Run:
  def __truediv__(self,x):return types.SimpleNamespace(exists=lambda:True)
 env['run']=Run();exec(code,env)
 def builder(*args):time.sleep(delay);return build(*args)
 worker=Preparation(builder);start=time.time();next_frame=0.;active=None;plans=[];timings=[];ticks=[];pending_ticks=0;first_ready=None
 try:
  for tick in range(440):
   until=start+tick*.01
   if until>clock():time.sleep(until-clock())
   now=clock();elapsed=now-start
   statepva=env['gate'].plan.at(now) if env['gate'].plan is not None else np.stack([center,np.zeros(3),np.zeros(3)])
   last['odom']=dict(p=statepva[0].tolist(),v=statepva[1].tolist(),q=pose['q']);last['imu']=dict(a_world=statepva[2].tolist())
   if elapsed>=next_frame:
    next_frame+=1/33.
    if source=='frozen_measured':depth=measured;camera_pose=pose
    else:
     theta=.003*elapsed
     camera_pose=dict(p=statepva[0].tolist(),q=[0,0,np.sin(theta/2),np.cos(theta/2)])
     den=np.cos(theta)+np.sin(theta)*(np.arange(160)-80)/80
     depth=np.tile(((center[0]+5-statepva[0,0])/den).astype(np.float32),(90,1))
    authority.observe(depth,camera_pose,now)
   if (not plans or (len(plans)==1 and now-first_ready>=1.8)) and worker.future is None:
    worker.submit(statepva[0].copy(),statepva[1].copy(),statepva[2].copy(),pose['q'],target.copy(),now)
   candidate=worker.take()
   if candidate is not None:
    active=candidate;plans.append(candidate)
    if first_ready is None:first_ready=now
   pending_ticks+=int(worker.future is not None)
   pva=active.at(now) if active else np.stack([center,np.zeros(3),np.zeros(3)])
   finished=active is not None and now-active.epoch>2.5
   if finished:pva=np.stack([target,np.zeros(3),np.zeros(3)])
   envelope=dict(frame='world',stamp=now,sequence=tick,pva=pva.tolist(),kind='trajectory' if active and not finished else 'hold')
   if envelope['kind']=='trajectory':envelope['plan']=active.record
   before=len(pubs);t=time.perf_counter()
   env['request'](types.SimpleNamespace(data=json.dumps(envelope)))
   timings.append(time.perf_counter()-t)
   proof=next((x[2] for x in reversed(logs) if x[0]=='birth_admission'),None)
   ticks.append(dict(tick=tick,stamp=now,published=len(pubs)-before,plan_id=env['gate'].plan.id if env['gate'].plan else None,
     goal=active.record['goal'] if active else target.tolist(),proof=proof,pending=worker.future is not None))
   if aborts:break
 finally:
  worker.close();env['gate'].close()
 passed=len(ticks)==440 and len(pubs)==440 and not aborts and len(plans)>=2 and all(t['proof'] and t['stamp']<=t['proof']['certificate_expiry'] for t in ticks)
 results.append(dict(label=label,scope='NONPHYSICAL_KINEMATIC_ACTUAL_CALLBACK_STUB_ROS',recorded_depth_path=prior['path'] if source=='frozen_measured' else None,depth_sha256=hashlib.sha256(measured.tobytes()).hexdigest() if source=='frozen_measured' else None,input_retimestamped=True,delay_ms=delay*1000,passed=passed,ticks=len(ticks),commands=len(pubs),aborts=list(aborts),plans=len(plans),pending_ticks=pending_ticks,control_p50_ms=float(np.percentile(timings,50))*1000,control_p95_ms=float(np.percentile(timings,95))*1000,control_max_ms=max(timings)*1000,over10ms=sum(t>.01 for t in timings),maxima=[p.maxima for p in plans],timeline=ticks))
 # Independent negatives: no command may publish through actual callback.
 now=clock();base_active=active
 for case in ['new_obstacle','depth_stale','birth_expired','outside_birth','nan','log_overflow']:
  bad=BirthAuthority(lambda t:BirthGrant(tuple(center),.6,t-1 if case=='birth_expired' else t+1,'NEGATIVE_FIXTURE'),clock)
  dep=np.full((90,160),.15 if case=='new_obstacle' else 5.,np.float32)
  bad.observe(dep,dict(p=center.tolist(),q=pose['q']),now-.201 if False else now)
  if case=='depth_stale':bad.frame.stamp=now-.201
  env['birth_authority']=bad;before=len(pubs);before_abort=len(aborts)
  if case=='outside_birth':
   other=build(center,np.zeros(3),np.zeros(3),pose['q'],center+[.5,0,0],now)
  else:other=build(center,np.zeros(3),np.zeros(3),pose['q'],target,now)
  env['gate']=Admission();env['gate'].plan=other;v=other.at(now)
  original_log=env['log']
  if case=='log_overflow':
   def broken_log(*args):raise OSError('LOG_BUFFER_OVERFLOW')
   env['log']=broken_log
  if case=='nan':v[0,0]=np.nan
  last['odom']=dict(p=center.tolist(),v=[0,0,0],q=pose['q']);last['imu']=dict(a_world=[0,0,0])
  envelope=dict(frame='world',stamp=now,pva=v.tolist(),kind='trajectory',plan=other.record)
  for retry in range(20):
   env['request'](types.SimpleNamespace(data=json.dumps(envelope)))
   if len(aborts)>before_abort:break
   time.sleep(.005)
  env['gate'].close();env['log']=original_log
  expected=dict(new_obstacle='F0_NEW_OCCUPIED_EVIDENCE',depth_stale='F0_DEPTH_STALE',birth_expired='BIRTH_EXPIRED',outside_birth='OUTSIDE_BIRTH_UNKNOWN',nan='REQUEST_PVA',log_overflow='LOG_BUFFER_OVERFLOW')[case]
  faults.append(dict(source=label,case=case,passed=len(pubs)==before and len(aborts)>before_abort and expected in aborts[-1],reason=aborts[-1] if aborts else None))
out=dict(HEALTHY_CONTROL_CONTINUITY_PASS=all(x['passed'] for x in results),runs=results)
(O/'F0_CONTROL_CONTINUITY_RESULTS.json').write_text(json.dumps(out,indent=2))
(O/'F0_FAULT_REGRESSION.json').write_text(json.dumps(dict(FAULT_FAIL_CLOSED_PASS=all(x['passed'] for x in faults),cases=faults),indent=2))
print(json.dumps([{k:v for k,v in x.items() if k!='timeline'} for x in results],indent=2));print(faults)
if not out['HEALTHY_CONTROL_CONTINUITY_PASS'] or not all(x['passed'] for x in faults):raise SystemExit(1)
