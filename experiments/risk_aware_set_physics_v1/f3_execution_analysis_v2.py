"""672 sealed real states: raw vs old splice vs retimed execution; offline only."""
import collections,csv,json,time
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *
from query_minco.inference.constrained_prefix import constrained_prefix
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution002'
OLD=ROOT/'research_outputs/rsps_forest_shadow_v1'
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else str(a))
def violations(v):return [k for k,x,lim in zip(['speed','acceleration','jerk'],v,[1,1.5,10]) if x>lim+1e-8]
def main():
    dest=OUT/'analysis_causal_verified_final004';dest.mkdir();saved=dest/'prefixes';saved.mkdir()
    r=OLD/'low_speed_shadow001';h=OLD/'low_speed_host001'
    decision=json.loads((h/'PILOT_DECISION.json').read_text())
    trace=[x['actual'] for x in map(json.loads,(h/'REFERENCE_TRACE.jsonl').read_text().splitlines()) if x.get('actual')]
    # Real retrospective activation records: choose first observed Odom clock >=
    # source+.10 and <=source+.15, independent of candidate or model outcome.
    activation_rows=[]
    for i,a in enumerate(trace):
        if i==0:continue
        dt=a['stamp']-trace[i-1]['stamp']
        if .005<=dt<=.3:
            activation_rows.append((a['stamp'],np.stack([a['p'],a['v'],(np.array(a['v'])-trace[i-1]['v'])/dt])))
    from query_minco.inference.low_speed_reference import reference,braking
    refrows=list(map(json.loads,(h/'REFERENCE_TRACE.jsonl').read_text().splitlines()))
    refstart=json.loads((h/'REFERENCE_START.json').read_text())['stamp']
    firstbrake=next(x for x in refrows if x['phase']=='BRAKE')
    class ArchivedActiveReference:
        def pva(self,stamp):
            t=stamp-refstart
            if t<firstbrake['t']:x,v,a=reference(t)
            else:x,v,a=braking(t-firstbrake['t'],firstbrake['desired_p'][0],firstbrake['desired_v'][0])
            return np.array([[x,0,2],[v,0,0],[a,0,0]],float)
    active=ArchivedActiveReference()
    # Causally merge only actual past logged Odom samples from both existing
    # independent subscribers. Never interpolate a missing state or stamp.
    logged=[]
    for row in refrows:
        a=row.get('actual')
        if a is not None:logged.append((row['stamp'],a['stamp'],np.array(a['p']),np.array(a['v']),'reference_trace'))
    for row in map(json.loads,(h/'MOTION_TRACE.jsonl').read_text().splitlines()):
        a=row.get('odom')
        if a is not None:logged.append((row['wall_time_ns']/1e9,a['stamp_ns']/1e9,np.array(a['p']),np.array(a['v']),'host_motion_trace'))
    logged.sort(key=lambda x:x[0])
    rows=[];groups=collections.defaultdict(list)
    events=list(map(json.loads,(r/'events.jsonl').read_text().splitlines()))
    for e in events:
        if e['type']!='FRAME' or e['stamp']<decision['full_start_stamp']:continue
        with np.load(r/('frame_%06d.npz'%e['index'])) as z:
            m=Measurement(e['stamp'],np.stack([z['position_W'],z['R_WB0']@z['state'][:3],z['R_WB0']@z['state'][3:6]]))
            candidates=[a for a in activation_rows if .10<=a[0]-e['stamp']<=.15]
            act,actual=candidates[0] if candidates else (e['stamp']+.12,None)
            now=act-.06
            available={}
            for received,stamp,pos,vel,stream in logged:
                if received<=now:available[stamp]=(pos,vel,stream,received)
            keys=sorted(available)
            t1=keys[-1] if keys else None
            older=[t for t in keys if t1 is not None and .005<=t1-t<=.3]
            if older:
                t0=older[-1];p0,v0,_,_=available[t0];p1,v1,stream,received=available[t1]
                dt=t1-t0
                current=Measurement(t1,np.stack([p1,v1,(v1-v0)/dt]))
            else:
                current=m;dt=float(z['acceleration_dt'])


            # Translate into the new empty room for EXTERNAL_SUPERVISOR geometry
            # replay only. This proves NOTHING about the recorded forest map.
            translation=np.array([0,0,2.])-z['position_W']
            volume=FreeVolume((box_cell(np.array([-2.8,-2.8,.2])-translation,np.array([6.8,2.8,3.8])-translation,act+20,'TRANSLATED_EMPTY_ROOM_EXTERNAL_ONLY'),),Scope.EXTERNAL_SUPERVISOR)
            for label in ['C1','E3']:
                t_group=time.perf_counter()
                for qi in range(6):
                    w=Program(world_coefficients(z[label+'_coefficients'][qi],z['position_W'],z['R_WB0']),z[label+'_durations'][qi])
                    raw=w.extrema();phase=act-e['stamp']
                    original_prefix=trim(w,phase,min(w.duration,phase+.1))
                    rawprefix=original_prefix.extrema()
                    con=quintic(m.predict(act),w.pva(phase+.25),.25)
                    old=concatenate(con,trim(w,phase+.25,phase+.35))
                    row=dict(frame=e['index'],model=label,query=qi,source_stamp=e['stamp'],activation_stamp=act,
                        raw_extrema=raw,raw_violations=violations(raw),raw_prefix_extrema=rawprefix,raw_prefix_violations=violations(rawprefix),
                        old_splice_extrema=old.extrema(),old_splice_violations=violations(old.extrema()),
                        activation_actual_odom_available=actual is not None,
                        measured_activation_pva_residual=None if actual is None else np.max(np.abs(active.pva(act)-actual),axis=1).tolist(),
                        predicted_measurement_vs_previous_reference_gap=np.max(np.abs(m.predict(act)-active.pva(act)),axis=1).tolist())
                    try:
                        ans=constrained_prefix(w,current,act,volume,now,active=active,source_stamp=e['stamp'])
                        path=saved/('%06d_%s_q%d.npz'%(e['index'],label,qi))
                        with path.open('xb') as f:np.savez_compressed(f,coefficients=ans['full'].coefficients,durations=ans['full'].durations,
                            prefix_coefficients=ans['prefix'].coefficients,prefix_durations=ans['prefix'].durations,
                            brake_coefficients=ans['brake'].coefficients,brake_durations=ans['brake'].durations,
                            measured_source_pva=m.pva,measured_current_pva=current.pva,current_odom_stamp=np.array(current.stamp),predicted_activation_pva=current.predict(act),continuous_previous_reference_pva=active.pva(act),
                            measured_activation_pva=np.zeros((0,3)) if actual is None else actual,translation_W=translation)
                        row.update(result='KINEMATIC_AND_EXTERNAL_ROOM_PASS',artifact=str(path.relative_to(OUT)),
                            **{k:v for k,v in ans.items() if k not in ['prefix','brake','full']})
                        row['current_odom_age_s']=now-current.stamp
                        row['current_acceleration_dt_s']=dt
                        row['estimated_source_to_ready_s']=float(max(now-e['stamp'],e['input_age_s']+e['model_generation_s'][label])+ans['compute_s'])
                        row['estimated_ready_before_activation_guard']=row['estimated_source_to_ready_s']<=act-e['stamp']-.02
                        row['measured_reference_splice_gap']=np.max(np.abs(ans['prefix'].pva(0)-active.pva(act)),axis=1).tolist()
                    except ValueError as exc:row.update(result=str(exc))
                    rows.append(row)
                groups[(e['index'],label)].append(time.perf_counter()-t_group)
    results={}
    for label in ['C1','E3','ALL']:
        rr=[x for x in rows if label=='ALL' or x['model']==label];ok=[x for x in rr if x['result']=='KINEMATIC_AND_EXTERNAL_ROOM_PASS']
        results[label]=dict(candidates=len(rr),raw_whole_feasible=sum(not x['raw_violations'] for x in rr),
            raw_short_prefix_feasible=sum(not x['raw_prefix_violations'] for x in rr),
            old_splice_feasible=sum(not x['old_splice_violations'] for x in rr),retimed_with_full_brake_external_feasible=len(ok),
            results=dict(collections.Counter(x['result'] for x in rr)),
            activation_measured_gate_pass=sum(x['measured_activation_pva_residual'] is not None and (np.array(x['measured_activation_pva_residual'])<=[.02,.05,.1]).all() for x in ok),
            retained_progress_min_p50_max=np.quantile([x['retained_progress_m'] for x in ok],[0,.5,1]).tolist() if ok else None,
            bundle_progress_min_p50_max=np.quantile([x['prefix_progress_m'] for x in ok],[0,.5,1]).tolist() if ok else None,
            full_duration_min_p50_max=np.quantile([np.load(OUT/x['artifact'])['durations'].sum() for x in ok],[0,.5,1]).tolist() if ok else None,
            compute_p50_p95_max_ms=(1000*np.quantile([x['compute_s'] for x in ok],[.5,.95,1])).tolist() if ok else None)
    group_ms={l:1000*np.array([v[0] for (f,label),v in groups.items() if label==l]) for l in ['C1','E3']}
    for l,ms in group_ms.items():results[l]['six_query_adaptation_including_evidence_IO_p50_p95_max_ms']=np.quantile(ms,[.5,.95,1]).tolist()
    dump(dest/'CANDIDATE_RESULTS.json',rows)
    dump(dest/'SUMMARY.json',dict(results=results,native_online_certified=0,
        geometry_scope='EXTERNAL_SUPERVISOR_TRANSLATED_EMPTY_ROOM_NOT_RECORDED_FOREST',
        activation_scope='head from known previous reference at real recorded activation clock, real source PVA tracking gate; activation Odom retrospective only',
        planning_note='per-six timings include evidence IO; compare plus live inference with60ms budget; not a ROS execution claim'))
    print(json.dumps(results,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else str(a)))
if __name__=='__main__':main()
