"""Verify actual C/D evidence and replay control/trajectory metrics,no new execution."""
from pathlib import Path
import sys,json,bisect,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.evidence import predecessor
from query_minco.integration.yopo_execution_v4.trajectory import Plan
from query_minco.integration.yopo_execution_v4.metrics import summarize
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1'
C=R/'research_outputs/g4cp2_manual_C_20261111_180435_1005/run/RESULT.json';D=R/'research_outputs/g4cp2_manual_D_20261111_182327_10722/run/RESULT.json'
d=json.loads(D.read_text());binding=d['source_binding'];mismatch=[p for p,h in binding.items() if sha(p)!=h];assert not mismatch
predecessor(C,'C_CAMPAIGN',binding);predecessor(D,'D',binding)
folder=D.parent/'none';p=folder/'supervisor_data/EVENTS.jsonl';assert sha(p)==d['events_sha256'];rows=[json.loads(x) for x in p.read_text().splitlines()]
refrows=[json.loads(x) for x in (folder/'reference_data/EVENTS.jsonl').read_text().splitlines()];plans=[x for x in refrows if x['kind']=='plan'];assert len(plans)==1
plan=Plan(plans[0]['plan']);stop=d['terminal']['stop_confirmed_monotonic']
events=[e for e in rows if e.get('monotonic',float('inf'))<=stop]
types={k:[e for e in events if e.get('kind')==k and e.get('accepted')] for k in ['odom','imu','request','controller_command','raw','final_so3','plant']}
final_by_stamp={e['stamp']:e for e in types['final_so3']};acks=[];forces=[]
for e in types['plant']:
 x=e['payload'];s=x['final_command_stamp']
 if s in final_by_stamp:
  sent=final_by_stamp[s];err=float(np.max(np.abs(np.asarray(sent['payload']['force_world'])-x['force_world'])))
  assert err<2e-5;acks.append(e['monotonic']-sent['monotonic']);forces.append(err)
freq={}
for k,rs in types.items():
 stamps=np.array([e['stamp'] for e in rs]);dt=np.diff(stamps)
 freq[k]=dict(samples=len(rs),source_hz=float((len(stamps)-1)/(stamps[-1]-stamps[0])) if len(stamps)>1 else None,max_gap_s=float(dt.max()) if len(dt) else None)
aligned=[]
for e in types['odom']:
 if e['stamp']>=plan.epoch:
  target=plan.at(e['stamp']);actual=e['payload'];aligned.append(dict(stamp=e['stamp'],p_error=float(np.linalg.norm(np.asarray(actual['p'])-target[0])),v_error=float(np.linalg.norm(np.asarray(actual['v'])-target[1]))))
nearest=lambda t,rs:min(rs,key=lambda x:abs(x['stamp']-t))
head_odom=nearest(plan.epoch,types['odom']);head_imu=nearest(plan.epoch,types['imu'])
head_actual=np.stack([head_odom['payload']['p'],head_odom['payload']['v'],head_imu['payload']['a_world']]);head_error=np.linalg.norm(plan.sample(0)-head_actual,axis=1)
rate_cmd=types['controller_command'];request_delay=[e['ros_now']-e['stamp'] for e in rate_cmd]
controller_times=[e['monotonic'] for e in rate_cmd];raw_delays=[]
for e in types['raw']:
 i=bisect.bisect_right(controller_times,e['monotonic'])-1
 if i>=0:raw_delays.append(e['monotonic']-controller_times[i])
stats=lambda v:dict(n=len(v),median_s=float(np.median(v)),max_s=float(np.max(v))) if v else dict(n=0)
faultstrings=['PLAN_AGE','INITIAL_PVA','PLAN_COMMAND_RACE','TRACKING_ENVELOPE']
faults=[e for e in rows if e.get('kind')=='fault']
assert not faults and d['runtime_status']==d['cleanup_status']=='PASS'
ranges={}
for name,key in [('force_world','force_world'),('rpm_actual','rpm_actual'),('rpm_input','rpm_input'),('thrust_actual_N','thrust_actual_N')]:
 a=np.asarray([e['payload'][key] for e in types['plant']]);ranges[name]=dict(min=np.min(a,axis=0).tolist(),max=np.max(a,axis=0).tolist())
review=dict(status='PHYSICAL_PASS',original_D=str(D),original_D_sha256=sha(D),original_C_sha256=sha(C),source_identity='STATIC_PASS',source_mismatches=mismatch,
 motion=d['motion'],full_available=summarize(events,anchor=[0,0,2]),rates=freq,planned_continuous_peaks=plan.maxima,head_epoch_error_s=abs(head_odom['stamp']-plan.epoch),head_PVA_errors=head_error.tolist(),
 source_time_aligned_max_p_error=max(x['p_error'] for x in aligned),source_time_aligned_max_v_error=max(x['v_error'] for x in aligned),
 controller_consumption='PHYSICAL_PASS at chain/actual response level;per-request internal controller ACK not recorded',plant_exact_ACK_count=len(acks),
 final_to_plant_ACK_latency=stats(acks),request_stamp_to_publication_age=stats(request_delay),controller_to_raw_nearest_prior_correlation=stats(raw_delays),
 latency_caveat='nearest-prior controller/raw correlation is not a causally identified per-command latency',
 forces_and_rpm=ranges,planning_s=plans[0]['planning_s'],solver_errors=plans[0]['solver_errors'],faults=faults,listed_admission_errors={s:False for s in faultstrings},
 cleanup_status=d['cleanup_status'],children=d['children'],warning=(folder/'plant/stderr.log').read_text(),new_physics_executions=0,new_network_updates=0)
write(O/'D_PHYSICAL_REVIEW.json',review);write(O/'D_SOURCE_TIME_ALIGNED_ERRORS.json',aligned)
print('D_PHYSICAL_PASS',d['motion'],'ACKS',len(acks),'SOURCE_BINDING_MATCH')
