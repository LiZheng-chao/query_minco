"""Multiview geometry, Engine isolation, and K1 contract comparison. No actuator code."""
from pathlib import Path
import sys,json,time,hashlib,collections,copy,types,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=ROOT/'research_outputs/forest_f1_blockers_20261118_v3'
import numpy as np,torch
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared,sample_batch
from query_minco.integration.observed_space_v1.map import Config,Frame
from query_minco.integration.depth_minco_t2_v1.space import Space,BirthGrant,authorized_snapshot
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation,sample_world
from query_minco.evaluation.validation import norm_max

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,d): (OUT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def summarize(values):
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=max(values)) if values else None

def geometry(records):
    ledger=ObservedMap(Config(shape=(32,24,16)));maps=[];probes=[];last=None
    for item in records:
        dep=np.load(item['path'])['depth'];stamp=item['stamp'];t=time.perf_counter();entry=dict(index=item['index'],stamp=stamp)
        try:
            ledger.add(dep,item['pose'],stamp,stamp,stamp);sensor=ledger.snapshot(stamp)
            grant=BirthGrant((.25,-.25,2.),.6,stamp+.2,'VIRTUAL_ENGINEERING_BIRTH_SCOPE_ONLY')
            snap=authorized_snapshot(sensor,grant,stamp,sensor.epoch,lambda:sensor.epoch);last=(snap,item)
            entry.update(status='MAP_BUILT',cells={name:int((snap.states==v).sum()) for name,v in [('UNKNOWN',0),('KNOWN_FREE',1),('OCCUPIED',2)]},coverage={name:float((snap.states==v).mean()) for name,v in [('UNKNOWN',0),('KNOWN_FREE',1),('OCCUPIED',2)]})
            balls=[]
            for d in [.0,.2,.4,.5,.6,.7,.8,1.0]:
                b=snap.ball(np.array([.25,-.25,2.])+[d,0,0],.33,stamp);balls.append(dict(distance=d,**b))
            entry['forward_connectivity']=balls
            # Background A* uses exact original budgets; failures are classified separately.
            tick=time.perf_counter();clock=lambda:stamp+time.perf_counter()-tick
            try:
                plan,proof=DepthPlanner(clock,space=Space(snap,clock)).propose(dict(item['pose'],a=[0,0,0]),np.array([3.25,-.25,2.]),stamp)
                probes.append(dict(kind='ASTAR_MINCO',index=item['index'],status='CERTIFIED',plan=plan.record,proof=proof,profile=plan._planning_profile,outside=bool((np.linalg.norm(sample_batch(plan,np.linspace(0,2,129))[:,0]-np.array([.25,-.25,2.]),axis=1)>.6).any())))
            except Exception as ex:probes.append(dict(kind='ASTAR_MINCO',index=item['index'],status='REJECTED',reason=str(ex),details=getattr(ex,'record',None),elapsed=time.perf_counter()-tick))
            # Fixed, predeclared geometry-independent straight probes. Not GT-selected paths.
            if item['index'] in [2,11,23]:
                for angle in range(0,360,45):
                    for distance in [.28,.65,.8,1.0]:
                        goal=np.array([.25,-.25,2.])+distance*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
                        pt=time.perf_counter();rec=dict(kind='PREDECLARED_STRAIGHT_MINCO',index=item['index'],angle=angle,distance=distance)
                        try:
                            plan=build(np.array([.25,-.25,2.]),np.zeros(3),np.zeros(3),item['pose']['q'],goal,stamp);rec['minco_s']=time.perf_counter()-pt
                            ct=time.perf_counter();proof=snap.certify(Prepared(plan),stamp);rec.update(certification_s=time.perf_counter()-ct,status='CERTIFIED',outside=distance>.6,plan=plan.record,proof=proof)
                        except Exception as ex:rec.update(status='REJECTED',reason=str(ex),details=getattr(ex,'record',None))
                        rec['total_s']=time.perf_counter()-pt;probes.append(rec)
        except Exception as ex:entry.update(status='REJECTED',reason=str(ex))
        entry['total_s']=time.perf_counter()-t;maps.append(entry)
    successes=[p for p in probes if p.get('outside') and p['status']=='CERTIFIED']
    write('MULTIVIEW_REPLAY_RESULT.json',dict(scope='VIRTUAL_VIEW_GEOMETRY_ONLY_NOT_FLIGHT_QUALIFICATION',frames=len(records),capture_span_s=records[-1]['stamp']-records[0]['stamp'],
        status='OUTSIDE_BIRTH_CERTIFIED_VIRTUAL_ONLY' if successes else 'INSUFFICIENT_OBSERVED_SPACE',outside_certified=len(successes),maps=maps,probes=probes,rejections=dict(collections.Counter(x.get('reason') for x in probes if x['status']=='REJECTED')),GT_used=False))
    return last

def engine_tests(records):
    # A. Truly isolated static valid geometry, with the existing finite synthetic birth scope.
    clock=[100.];e=Engine(lambda now:BirthGrant((0,0,0),.6,now+.2,'SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION'),lambda:clock[0])
    dep=np.full((90,160),5.,np.float32);pose=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);e.observe(dep,pose,100.,100.);e.set_state(dict(pose,a=[0,0,0]),100.);e.set_goal([.28,0,0],100.)
    passed=e.run_once();cmd=e.command(100.);static=dict(scope='SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION',committed=passed,command_valid=cmd is not None,events=e.take_events());e.close()
    # B. Every captured frame at its true relative receipt time; virtual camera input is explicitly nonphysical.
    start=records[0]['receipt_ros'];wall=time.monotonic();clock=[records[0]['stamp']]
    engine=Engine(lambda now:BirthGrant((.25,-.25,2.),.6,now+.2,'VIRTUAL_SCOPE_NOT_FLIGHT'),lambda:clock[0]);engine.start();audit=[];samples=[]
    for item in records:
        target=item['receipt_ros']
        while start+time.monotonic()-wall<target:
            clock[0]=start+time.monotonic()-wall;c=engine.command(clock[0]);samples.append(dict(t=clock[0],valid=c is not None,queue=len(engine.pending)))
            time.sleep(.001)
        clock[0]=start+time.monotonic()-wall
        try:
            engine.set_state(dict(item['pose'],a=[0,0,0]),item['stamp']);engine.set_goal([.53,-.25,2.],item['stamp'])
            before=len(engine.pending);engine.observe(np.load(item['path'])['depth'],item['pose'],item['stamp'],item['stamp']);audit.append(dict(event='FRAME',queue_before=before,queue_after=len(engine.pending),revision=engine.revision,source_stamp=item['stamp']))
        except Exception as ex:audit.append(dict(event='INPUT_REJECTED',reason=str(ex),queue=len(engine.pending)))
        audit.extend(engine.take_events())
    engine.close();audit.extend(engine.take_events())
    # Concrete reproducible ingestion failure: successful ledger.add followed by snapshot failure.
    # No production mutation: inject failure only in a dedicated synthetic test instance.
    c=[200.];q=Engine(lambda now:None,lambda:c[0]);q.set_state(dict(pose,a=[0,0,0]),200.);q.set_goal([.28,0,0],200.)
    def fail_snapshot(now):raise ValueError('SYNTHETIC_SNAPSHOT_FAILURE')
    q.ledger.snapshot=fail_snapshot;timeline=[]
    for i in range(10):
        c[0]=200.+i*.03
        try:q.observe(dep,pose,c[0],c[0]);q.set_state(dict(pose,a=[0,0,0]),c[0]);q.set_goal([.28,0,0],c[0]);q.run_once()
        except Exception as ex:timeline.append(dict(i=i,reason=str(ex)))
        timeline.append(dict(i=i,pending=len(q.pending),ledger_last_stamp=q.ledger.last_stamp,evidence_overflow=q.evidence_overflow))
    q.close()
    write('ENGINE_REALTIME_TESTS.json',dict(static=static,realtime=dict(scope='VIRTUAL_VIEW_RECORDED_FRAME_RATE_NOT_FLIGHT',frames=len(records),events=audit,samples=samples,valid_fraction=sum(x['valid'] for x in samples)/max(1,len(samples)),max_queue=max([x.get('queue_after',x.get('queue',0)) for x in audit]+[0]),counts=dict(collections.Counter(x['event'] for x in audit))),reproduction=dict(scope='SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION',timeline=timeline)))

def k1_compare(records,last):
    from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner
    from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
    from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
    from query_minco.trajectory.minco_adapter import MincoAdapter
    from query_minco.training.k1_physics_trainer_v1.trainer import state_sha
    models={};identities={}
    pilot=ROOT/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104/step128.pt'
    formal=ROOT/'research_outputs/g4a_forest_shadow_20261111_v1/formal/A_reasonable_seed20261104/step128.pt'
    accepted=json.loads((ROOT/'research_outputs/g4b_formal128_and_control_20261111_v1/FORMAL128_IDENTITY_ACCEPTANCE.json').read_text());point=next(x for x in accepted['points'] if x['step']==128)
    for name,path in [('pilot',pilot),('formal',formal)]:
        if not path.exists():identities[name]=dict(status='NOT_AVAILABLE',path=str(path));continue
        side=json.loads(Path(str(path)+'.json').read_text());assert sha(path)==side['file_sha256']
        if name=='formal':assert sha(path)==point['file_sha256']
        blob=torch.load(path,map_location='cpu',weights_only=True);model=make_planner().eval();model.load_state_dict(blob['model'],strict=True);assert state_sha(model)==side['model_sha256'];models[name]=model;identities[name]=dict(path=str(path),sha256=sha(path),model_sha=state_sha(model),cursor=blob['cursor'])
    decoder=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);solver=MincoAdapter(num_candidates=1);rows=[]
    for item in records:
        d=np.load(item['path'])['depth'];p=np.array(item['pose']['p']);q=item['pose']['q'];stamp=item['stamp']
        ledger=ObservedMap(Config(shape=(32,24,16)));snap=None
        try:
            ledger.add(d,item['pose'],stamp,stamp,stamp);sensor=ledger.snapshot(stamp);snap=authorized_snapshot(sensor,BirthGrant(tuple(p),.6,stamp+.2,'VIRTUAL_ONLY'),stamp,sensor.epoch,lambda:sensor.epoch)
        except Exception:pass
        for distance in [.28,3.]:
            goal=p+np.array([distance,0,0]);x,s,R=observation(d,q,[0,0,0],[0,0,0],p,goal,item['K'])
            for name,model in models.items():
                row=dict(view=item['index'],model=name,goal_distance=distance);tick=time.perf_counter()
                with torch.no_grad():raw=model(torch.from_numpy(x),torch.from_numpy(s));par=decoder(raw,torch.from_numpy(s));tr=solver(par.head_pva,par.tail_pva,par.inner_pts,par.durations)
                c=tr.coeffs[0,0].numpy();dur=tr.durations[0,0].numpy();row['inference_decode_minco_s']=time.perf_counter()-tick
                plan=types.SimpleNamespace(c=c,d=dur,R=R,origin=p,id=hashlib.sha256(c.tobytes()).hexdigest());plan.maxima=[max(norm_max(a,float(t),k) for a,t in zip(c,dur)) for k in [1,2,3]]
                row.update(maxima=plan.maxima,pass_training_dynamics=all(v<=l+1e-7 for v,l in zip(plan.maxima,[3,8,100])),pass_F1_dynamics=all(v<=l+1e-7 for v,l in zip(plan.maxima,[1,8,100])),endpoint=(sample_batch(plan,np.array([2.]))[0,0]).tolist(),raw21=raw.tolist(),coefficients=c.tolist(),durations=dur.tolist())
                row['goal_error_m']=float(np.linalg.norm(np.array(row['endpoint'])-goal));row['geometry_pass']=False
                if snap is not None:
                    try:row['certificate']=snap.certify(Prepared(plan),stamp);row['geometry_pass']=True
                    except Exception as ex:row['geometry_rejection']=str(ex)
                else:row['geometry_rejection']='UNKNOWN_MAP_BUILD_FAILED'
                rows.append(row)
    summary=[]
    for name in models:
        for goal in [.28,3.]:
            group=[x for x in rows if x['model']==name and x['goal_distance']==goal];summary.append(dict(model=name,goal_distance=goal,n=len(group),mean_goal_error_m=float(np.mean([x['goal_error_m'] for x in group])),training_dynamics_pass=sum(x['pass_training_dynamics'] for x in group),F1_dynamics_pass=sum(x['pass_F1_dynamics'] for x in group),geometry_pass=sum(x['geometry_pass'] for x in group),latency=summarize([x['inference_decode_minco_s'] for x in group])))
    write('K1_GOAL_AND_DYNAMICS_COMPARISON.json',dict(scope='SAME_VIRTUAL_NATIVE_DEPTH_INPUT_NO_CONTROL',identities=identities,summary=summary,records=rows,execution_limit_unchanged=[1,8,100]))

def original_realtime():
    import query_minco.integration.depth_minco_t2_v1.engine as module
    pairs=json.loads((ROOT/'research_outputs/forest_f1_20261118_v2/PAIRS.json').read_text())['pairs']
    start=pairs[0]['receipt_ros'];wall=time.monotonic();now=[start]
    e=Engine(lambda t:BirthGrant((.25,-.25,2.),.6,t+.2,'SYNTHETIC_BIRTH_TIMING_ISOLATION_NOT_QUALIFICATION'),lambda:now[0])
    durations=collections.defaultdict(list)
    def timed(obj,name):
        original=getattr(obj,name)
        def call(*a,**kw):
            t=time.perf_counter()
            try:return original(*a,**kw)
            finally:durations[name].append(time.perf_counter()-t)
        setattr(obj,name,call)
    timed(e.ledger,'add');timed(e.ledger,'snapshot')
    # No synthetic free geometry is injected; the timing-only birth scope is explicitly separate from flight authority.
    e.start();events=[];samples=[];inputs=[]
    for pair in pairs:
        while start+time.monotonic()-wall<pair['receipt_ros']:
            now[0]=start+time.monotonic()-wall;command=e.command(now[0]);samples.append(dict(time=now[0],valid=command is not None,queue=len(e.pending),revision=e.revision))
            time.sleep(.001)
        now[0]=start+time.monotonic()-wall
        try:
            e.set_state(dict(pair['pose'],a=pair['imu']['a_world']),pair['stamp']);e.set_goal([3.25,-.25,2.],pair['stamp'])
            e.observe(np.load(pair['path'])['depth'],pair['pose'],pair['stamp'],pair['stamp'])
            inputs.append(dict(stamp=pair['stamp'],accepted=True,queue=len(e.pending),ledger_stamp=e.ledger.last_stamp,lag_s=now[0]-pair['receipt_ros']))
        except Exception as ex:inputs.append(dict(stamp=pair['stamp'],accepted=False,reason=str(ex),queue=len(e.pending)))
        events.extend(e.take_events())
    e.close();events.extend(e.take_events());gap=0.;began=None
    for sample in samples:
        if not sample['valid'] and began is None:began=sample['time']
        if sample['valid'] and began is not None:gap=max(gap,sample['time']-began);began=None
    if began is not None:gap=max(gap,samples[-1]['time']-began)
    report=dict(scope='REAL_F0_DEPTH_FRAME_RATE_WITH_SYNTHETIC_BIRTH_TIMING_ISOLATION_NOT_FLIGHT_QUALIFICATION',frames=len(pairs),inputs=inputs,events=events,samples=samples,
        counts=dict(collections.Counter(x['event'] for x in events)),input_rejections=dict(collections.Counter(x.get('reason') for x in inputs if not x['accepted'])),
        valid_fraction=sum(x['valid'] for x in samples)/max(1,len(samples)),longest_no_command_s=gap,stage_timing={k:summarize(v) for k,v in durations.items()},
        planning_profile_records=[x for x in events if x.get('event')=='CERTIFIED_COMMITTED'],unreached_stages='No successful plan means MINCO/certification/revalidation may be NOT_REACHED, not zero cost')
    write('ORIGINAL_FRAME_RATE_TEST.json',report);print('ORIGINAL_FRAME_RATE',report['counts'],report['input_rejections'],report['valid_fraction'])

def main():
    torch.set_num_threads(1);records=json.loads((OUT/'INPUTS.json').read_text())['records'];last=geometry(records);engine_tests(records);k1_compare(records,last)
if __name__=='__main__':
    if '--original-realtime' in sys.argv:torch.set_num_threads(1);original_realtime()
    else:main()
