"""Reuse historical exact-prefix IdealPlant; isolated analytic interface diagnostics."""
import hashlib,importlib,json
from pathlib import Path
import numpy as np
import query_minco
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_dev001'
OLD=Path('/home/lizheng/reach_planner/query_minco')
def dump(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)
def head(p=0,v=.8,a=.2):return np.array([[p,0,2],[v,0,0],[a,0,0]],float)
def main():
    query_minco.__path__.append(str(OLD)) # read-only legacy dependencies; mainline modules retain precedence.
    from query_minco.experiments.closed_loop_ideal_plant_v4.plant import IdealPlant
    protected={}
    for name in ['smoke_A_v4','smoke_B_v4','smoke_C_v4']:
        p=OLD/'outputs/closed_loop_ideal_plant_v4/runs'/name/'RESULT.json'
        protected[str(p)]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=json.loads(p.read_text()))
    def advance(plant,program):
        # Historical adapter requires exactly three pieces. Subdivision is exact.
        if len(program.durations)==1:
            parts=[trim(program,program.duration*i/3,program.duration*(i+1)/3) for i in range(3)]
            c=np.concatenate([p.coefficients for p in parts]);d=np.concatenate([p.durations for p in parts])
        elif len(program.durations)==2:
            t=program.durations[0]/2
            a=trim(program,0,t);b=trim(program,t,program.durations[0]);c0=trim(program,program.durations[0])
            c=np.concatenate([p.coefficients for p in [a,b,c0]]);d=np.concatenate([p.durations for p in [a,b,c0]])
        else:raise ValueError('DIAGNOSTIC_PIECES')
        c=np.einsum('ij,sjk->sik',plant.rotation.T,c);c[:,:,0]-=plant.rotation.T@plant.position
        actual=plant.prefix(c,d,float(d.sum()))
        expected=program.pva(program.duration)
        error=np.max(np.abs(np.stack([plant.position,plant.velocity,plant.acceleration])-expected),axis=1)
        assert (error<[.0002,.002,.01]).all(),error
        return dict(duration=program.duration,head_gap=actual['head_gap_pva'],final_error=error.tolist(),final_pva=expected.tolist())
    h=head();plant=IdealPlant(h[0].copy(),h[1].copy(),h[2].copy(),np.eye(3))
    stop,metrics=brake(h);stop_result=advance(plant,stop)
    final=terminal(np.stack([plant.position,plant.velocity,plant.acceleration]),[3,0,2])
    terminal_result=advance(plant,final)
    assert np.linalg.norm(plant.velocity)<1e-9 and np.linalg.norm(plant.position-[3,0,2])<1e-9
    # Splice at activation against an existing nonzero-v/a scheduled reference.
    src=Program(np.array([[[0,.8,.1,0,0,0],[0,0,0,0,0,0],[2,0,0,0,0,0]]],float),[2.])
    active=Scheduled(src,0,0,2,{},stop)
    m=Measurement(.1,src.pva(.1))
    local=Program(src.coefficients.copy(),src.durations)
    empty=FreeVolume((box_cell([-2,-2,0],[8,2,4],100,'CPU_EMPTY_ROOM_ONLY'),),Scope.EXTERNAL_SUPERVISOR)
    scheduled=prepare(local,0,[0,0,0],np.eye(3),m,.1,.16,empty,active=active)
    s=scheduled.program.pva(0);plant=IdealPlant(s[0].copy(),s[1].copy(),s[2].copy(),np.eye(3),time_s=.16)
    splice_result=advance(plant,scheduled.program);reserve_result=advance(plant,scheduled.brake_reserve)
    sources={}
    for name,mod in list(__import__('sys').modules.items()):
        p=getattr(mod,'__file__',None)
        if p and str(OLD) in p and Path(p).is_file():sources[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    dump('IDEAL_PLANT_INTERFACE_RESULT.json',dict(kind='CPU_EXACT_POLYNOMIAL_INTERFACE_FIXTURE_NOT_QUERY_CLOSED_LOOP',
        brake=stop_result,brake_bounds=metrics,terminal=terminal_result,splice=splice_result,reserved_brake=reserve_result,
        old_rejected_results_unchanged=protected,legacy_sources=sources,ROS_run=False,query_takeover=False,
        tracking_error_bound_validated=False))
    # Native-ray counterexample: an obstacle lies strictly between sampled pixel rays.
    from query_minco.evaluation.geometry import raycast
    backdrop=dict(name='backdrop',min=[10,-20,-20],max=[10.01,20,20])
    thin=dict(name='subpixel_rod',min=[1.499,-.016,-.001],max=[1.501,-.014,.001])
    _,_,d0,_=raycast(dict(obstacles=[backdrop]));_,_,d1,_=raycast(dict(obstacles=[backdrop,thin]))
    assert np.array_equal(d0,d1),'FIXTURE_MUST_MISS_SUBPIXEL_ROD'
    dump('SUBPIXEL_COUNTEREXAMPLE.json',dict(kind='ANALYTIC_SENSOR_FIXTURE_NOT_ONLINE_GT',
        depth_arrays_identical=True,thin_box=thin,body_center_at_obstacle=[1.5,-.015,0],
        conclusion='identical discrete depth admits a body collision; interpolation cannot certify the whole pixel footprint'))
    # Actual frozen candidate failures under fixed low-speed execution bounds.
    r=ROOT/'research_outputs/rsps_forest_shadow_v1/low_speed_shadow001'
    dec=json.loads((r.parent/'low_speed_host001/PILOT_DECISION.json').read_text())
    events=[json.loads(l) for l in (r/'events.jsonl').read_text().splitlines()]
    rows=[]
    for e in events:
        if e['type']!='FRAME' or e['stamp']<dec['full_start_stamp']:continue
        with np.load(r/('frame_%06d.npz'%e['index'])) as z:
            m=Measurement(0,np.stack([z['position_W'],z['R_WB0']@z['state'][:3],z['R_WB0']@z['state'][3:6]]))
            for label in ['C1','E3']:
                for qi in range(6):
                    p=Program(world_coefficients(z[label+'_coefficients'][qi],z['position_W'],z['R_WB0']),z[label+'_durations'][qi])
                    # Predeclared diagnostic durations; no limit changes.
                    for T in [.25,.35,.5,.75]:
                        con=quintic(m.predict(.08),p.pva(.08+T),T)
                        program=concatenate(con,trim(p,.08+T,.18+T))
                        maxima=program.extrema()
                        rows.append(dict(frame=e['index'],label=label,query=qi,splice_s=T,maxima=maxima,
                            violations=[key for key,val,limit in zip(['speed','acceleration','jerk'],maxima,[1.,1.5,10.]) if val>limit]))
    summary={}
    for T in [.25,.35,.5,.75]:
        subset=[x for x in rows if x['splice_s']==T]
        summary[str(T)]=dict(cases=len(subset),all_limits_passed=sum(not x['violations'] for x in subset),
            violations={k:sum(k in x['violations'] for x in subset) for k in ['speed','acceleration','jerk']},
            extrema_min_median_p95_max=np.quantile([x['maxima'] for x in subset],[0,.5,.95,1],axis=0).tolist())
    dump('EXECUTION_LIMIT_DIAGNOSTICS.json',dict(limits=dict(speed=1.,acceleration=1.5,jerk=10.),summary=summary,rows=rows,
        rule='diagnostic comparison only, no rate/limit relaxation, no actual publication'))
    print(json.dumps(dict(legacy_plant_fixture='PASS_NUMERICAL_INTERFACE_ONLY',old_results={k:v['result']['outcome'] for k,v in protected.items()},limits=summary),indent=2))
if __name__=='__main__':main()
