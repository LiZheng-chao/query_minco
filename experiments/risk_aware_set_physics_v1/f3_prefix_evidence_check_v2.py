"""Independent coefficient/evidence check of final causal replay; no ROS."""
import collections,json
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution002';D=OUT/'analysis_causal_verified_final004'
def main():
    rows=json.loads((D/'CANDIDATE_RESULTS.json').read_text());old=ROOT/'research_outputs/rsps_forest_shadow_v1/low_speed_shadow001'
    checks=[];old_vs_raw=[];ready=collections.Counter()
    for r in rows:
        if r['result']!='KINEMATIC_AND_EXTERNAL_ROOM_PASS':continue
        with np.load(OUT/r['artifact']) as z:
            p=Program(z['coefficients'],z['durations']);prefix=Program(z['prefix_coefficients'],z['prefix_durations']);b=Program(z['brake_coefficients'],z['brake_durations'])
            p.validate(MotionLimits())
            np.testing.assert_allclose(p.pva(0),z['continuous_previous_reference_pva'],atol=1e-10)
            np.testing.assert_allclose(prefix.pva(prefix.duration),b.pva(0),atol=1e-9)
            assert np.linalg.norm(p.pva(p.duration)[1:])<1e-8
            assert .005<=r['current_acceleration_dt_s']<=.3 and 0<=r['current_odom_age_s']<=.05
            translate=z['translation_W']
            v=FreeVolume((box_cell(np.array([-2.8,-2.8,.2])-translate,np.array([6.8,2.8,3.8])-translate,r['activation_stamp']+20,'independent-verified-empty-box'),),Scope.EXTERNAL_SUPERVISOR)
            proof=v.certify(p.coefficients,p.durations,r['activation_stamp']-.06,required_until=r['activation_stamp']+p.duration)
            assert proof['status']=='CERTIFIED_FREE' and proof['trajectory_sha256']==r['certificate']['trajectory_sha256']
            checks.append(dict(model=r['model'],extrema=p.extrema(),duration=p.duration,
                prefix_progress=r['prefix_progress_m'],retained_progress=r['retained_progress_m'],
                source_interval=r['source_interval'],retained_s=r['retained_physical_s']))
            budget=bool(r['estimated_ready_before_activation_guard'])
            state=r['measured_activation_pva_residual'] is not None and bool((np.array(r['measured_activation_pva_residual'])<=[.02,.05,.1]).all())
            ready[r['model']+'_ready_estimate']+=budget;ready[r['model']+'_state_and_ready']+=budget and state
    # Isolate the added splice cost using the SAME archived reference head and
    # activation time for every candidate, independently of the new algorithm.
    from query_minco.inference.low_speed_reference import reference,braking
    h=old.parent/'low_speed_host001';refstart=json.loads((h/'REFERENCE_START.json').read_text())['stamp']
    refrows=list(map(json.loads,(h/'REFERENCE_TRACE.jsonl').read_text().splitlines()));br=next(x for x in refrows if x['phase']=='BRAKE')
    comparison=[]
    for r in rows:
        with np.load(old/('frame_%06d.npz'%r['frame'])) as z:
            label=r['model'];qi=r['query'];w=Program(world_coefficients(z[label+'_coefficients'][qi],z['position_W'],z['R_WB0']),z[label+'_durations'][qi])
            t=r['activation_stamp']-refstart
            x,v,a=reference(t) if t<br['t'] else braking(t-br['t'],br['desired_p'][0],br['desired_v'][0])
            head=np.array([[x,0,2],[v,0,0],[a,0,0]])
            phase=r['activation_stamp']-r['source_stamp']
            direct=concatenate(quintic(head,w.pva(phase+.25),.25),trim(w,phase+.25,phase+.35))
            orig=w.extrema();splice=direct.extrema()
            comparison.append(dict(model=label,raw=orig,direct_splice=splice,added_jerk=splice[2]-orig[2],
                raw_pass=not r['raw_violations'],raw_prefix_pass=not r['raw_prefix_violations'],
                splice_pass=all(v<=lim+1e-8 for v,lim in zip(splice,[1,1.5,10]))))
    results={}
    for label in ['C1','E3','ALL']:
        rr=[r for r in rows if label=='ALL' or r['model']==label];cc=[c for c in checks if label=='ALL' or c['model']==label];co=[c for c in comparison if label=='ALL' or c['model']==label]
        results[label]=dict(total=len(rr),verified_prefix_and_brake=len(cc),
            old_reference_aligned_direct_splice_pass=sum(c['splice_pass'] for c in co),
            raw_whole_pass=sum(c['raw_pass'] for c in co),raw_short_pass=sum(c['raw_prefix_pass'] for c in co),
            new_failures=dict(collections.Counter(r['result'] for r in rr if r['result']!='KINEMATIC_AND_EXTERNAL_ROOM_PASS')),
            actual_execution_extrema_max=np.max([c['extrema'] for c in cc],axis=0).tolist(),
            full_duration_range=[min(c['duration'] for c in cc),max(c['duration'] for c in cc)],
            direct_splice_jerk_minus_raw_min_p50_max=np.quantile([c['added_jerk'] for c in co],[0,.5,1]).tolist(),
            direct_splice_jerk_greater_than_raw=sum(c['added_jerk']>0 for c in co))
    report=dict(results=results,readiness_estimates=dict(ready),artifacts_verified=len(checks),native_depth_certified=0,
        scope='offline kinematic proof and verified-empty-room translation only; ROS Controller failure blocks ALL actual Query execution',
        operational_publication_qualified=0)
    with (OUT/'PREFIX_EVIDENCE_ACCEPTANCE.json').open('x') as f:json.dump(report,f,indent=2)
    with (OUT/'RAW_VS_SPLICE_COMPARISON.json').open('x') as f:json.dump(comparison,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
