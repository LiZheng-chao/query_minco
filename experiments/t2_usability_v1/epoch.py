"""Actual Engine, changing synthetic ray measurements and legal slow poses."""
import sys,time,json,threading
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
def main():
 from query_minco.integration.yopo_execution_v4.trajectory import build
 build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.1,0,0],time.time())
 real=json.loads(Path('research_outputs/t2_acceptance_20261116_v1/replay/RESULT.json').read_text())
 src=Path(real['rows'][0]['path']).parents[1]/'supervisor_data/EVENTS.jsonl'
 native=[json.loads(x) for x in src.read_text().splitlines()]
 stamps=sorted(set(x['stamp'] for x in native if x.get('kind')=='depth' and x.get('accepted')))
 intervals=np.diff(stamps);intervals=intervals[(intervals>0)&(intervals<=.2)].tolist()
 rows=[]
 for delay in [.05,.1,.15]:
  for conflict in [False,True]:
   e=Engine(lambda now:BirthGrant((0,0,0),.60,now+.15,'SYNTHETIC_FINITE_BIRTH_ONLY'))
   from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
   now=time.time();zero=dict(p=[0,0,0],v=[0,0,0],a=[0,0,0],q=[0,0,0,1])
   e.observe(np.full((90,160),5.,np.float32),zero,now,now);e.set_state(zero,now);e.set_goal([.28,0,0],now)
   warm=e.run_once()
   if warm:
    initial=e.command(time.time())
    if initial:e.published(initial)
   original=DepthPlanner.propose
   def slower(self,*args,**kwargs):time.sleep(delay);return original(self,*args,**kwargs)
   DepthPlanner.propose=slower
   e.start();start=time.perf_counter();commands=0;max_control=0.;injected=None;bad_after=0
   next_frame=0.;frame_index=0;inputs=[];latencies=[];late_ticks=[];expired_commands=0;stale_commands=0;grant_end=time.time()+2.8 if delay==.05 and not conflict else float('inf')
   e.birth_loader=lambda now:BirthGrant((0,0,0),.60,min(now+.15,grant_end),'SYNTHETIC_FINITE_BIRTH_ONLY')
   try:
    for tick in range(360):
     until=start+tick*.01
     if until>time.perf_counter():time.sleep(until-time.perf_counter())
     now=time.time();elapsed=time.perf_counter()-start;late_ticks.append(max(0.,time.perf_counter()-until))
     active=e.last_executed_plan
     pva=active.at(now) if active else np.zeros((3,3))
     state=dict(p=pva[0].tolist(),v=pva[1].tolist(),a=pva[2].tolist(),q=[0,0,np.sin(elapsed*.001),np.cos(elapsed*.001)])
     e.set_state(state,now);e.set_goal([.28,0,0],now)
     sensor_paused=not conflict and delay==.1 and elapsed>2.8
     if elapsed>=next_frame and not sensor_paused:
      next_frame+=intervals[frame_index%len(intervals)];frame_index+=1
      obstacle=conflict and elapsed>.9
      if obstacle and injected is None:injected=time.monotonic()
      world_plane=state['p'][0]+.15 if obstacle else 5.
      denominator=np.cos(elapsed*.002)+np.sin(elapsed*.002)*(np.arange(160)-80)/80
      depth=np.tile(((world_plane-state['p'][0])/denominator).astype(np.float32),(90,1))
      inputs.append(dict(stamp=now,elapsed=elapsed,pose=state,world_plane_x=world_plane,depth_sha256=__import__('hashlib').sha256(depth.tobytes()).hexdigest()))
      try:e.observe(depth,state,now,now)
      except ValueError:break
     t=time.perf_counter()
     c=e.command(now);latencies.append(time.perf_counter()-t);max_control=max(max_control,latencies[-1])
     if c:
      commands+=1;e.published(c)
      if injected is not None:bad_after+=1
      if now>grant_end:expired_commands+=1
      if sensor_paused and elapsed>3.01:stale_commands+=1
     # Simulated no publication: no physical/closed-loop tracking claim.
     e.take_events() if False else None
   finally:e.close();DepthPlanner.propose=original
   events=list(e.events)
   rows.append(dict(scope='NONPHYSICAL_KINEMATIC_REPLAY_STATIC_PLANE_FINITE_BIRTH',delay_ms=delay*1000,
     conflict=conflict,native_input_rate_hz=1/float(np.mean(intervals)),input_frames=inputs,control_calls=len(latencies),control_p50_ms=float(np.percentile(latencies,50))*1000,control_p95_ms=float(np.percentile(latencies,95))*1000,control_10ms_exceedances=sum(x>.01 for x in latencies),tick_10ms_late=sum(x>.01 for x in late_ticks),commands_after_grant_expired=expired_commands,commands_after_sensor_pause=stale_commands,dynamic_new_plan_commits=max(0,sum(x.get('replan',False) for x in events if x['event']=='CERTIFIED_COMMITTED')-int(warm)),warm_qualified=warm,fault_injected=injected is not None,replan_commits=sum(x.get('replan',False) for x in events if x['event']=='CERTIFIED_COMMITTED'),commits=e.counts['CERTIFIED_COMMITTED'],commands_available=commands,
     commands_after_conflict=bad_after,max_control_ms=max_control*1000,
     reasons={str(k):sum(x.get('reason')==k for x in events if x['event']=='REJECTED') for k in set(x.get('reason') for x in events if x['event']=='REJECTED')},
     queue_overflow=e.evidence_overflow,events=events))
 Path('research_outputs/t2_usability_20261116_v1/EPOCH_LIVENESS_TEST.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps([{k:v for k,v in r.items() if k not in ['events','input_frames']} for r in rows],indent=2))
if __name__=='__main__':main()
