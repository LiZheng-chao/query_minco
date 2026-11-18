"""Recorded native geometry and publication causality diagnostics, no control IO."""
from pathlib import Path
import sys,time,json,hashlib,collections,types,heapq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=ROOT/'research_outputs/forest_f1_final_diagnosis_20261118_v1'
import numpy as np,torch
from scipy.ndimage import label
from query_minco.integration.observed_space_v2.core import ObservedMap,sample_batch
from query_minco.integration.observed_space_v1.map import Config
from query_minco.integration.depth_minco_t2_v1.space import Space,BirthGrant,authorized_snapshot
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.depth_minco_t2_v1.engine import Engine

def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def routes(records):
    ledger=ObservedMap(Config(shape=(32,24,16)));rows=[]
    for item in records:
        now=item['stamp'];row=dict(view=item['index'],stamp=now,pose=item['pose'],task_goal=[3.25,-.25,2.]);dep=np.load(item['path'])['depth'];clock_start=time.perf_counter();clock=lambda:now+time.perf_counter()-clock_start
        try:
            ledger.add(dep,item['pose'],now,now,now);sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,BirthGrant((.25,-.25,2.),.6,now+.2,'VIRTUAL_ONLY'),now,sensor.epoch,lambda:sensor.epoch)
            ids=np.indices(snap.states.shape).transpose(1,2,3,0);centers=snap.origin+(ids+.5)*.1;outside=np.linalg.norm(centers-[.25,-.25,2.],axis=-1)>.6
            components,n=label(snap.states==1);sizes=np.bincount(components.ravel());sizes[0]=0
            start=np.array(item['pose']['p']);ij=np.floor((start-snap.origin)/.1).astype(int);component=int(components[tuple(ij)])
            horizon=np.clip(row['task_goal'],snap.origin+.33,snap.origin+np.array(snap.cfg.shape)*.1-.33)
            row.update(map_origin=snap.origin.tolist(),map_shape=list(snap.cfg.shape),map_extent=(np.array(snap.cfg.shape)*.1).tolist(),horizon=horizon.tolist(),search_lattice_step=.2,search_bounds_nodes=[15,10],goal_tolerance=.25,
                raw_free_components=int(n),largest_raw_free_component=int(sizes.max()),outside_raw_known_free_cells=int(((snap.states==1)&outside).sum()),start_component=component,start_component_size=int(sizes[component]) if component else 0,
                start_component_outside_cells=int(((components==component)&outside).sum()) if component else 0)
            row['start_ball']=snap.ball(start,.33,now)
            from query_minco.integration.observed_space_v1.map import birth_covers_intersection
            probe=start+np.array([.3,0,0]);flat=centers.reshape(-1,3);lo=flat-.05;hi=flat+.05
            nearest=np.minimum(np.maximum(probe,lo),hi);intersect=np.linalg.norm(nearest-probe,axis=1)<=.33+1e-12
            covered=birth_covers_intersection(lo,hi,probe,.33,snap.birth_ball)
            blocked=intersect&((snap.states.ravel()==2)|((snap.states.ravel()==0)&~covered))
            selected=np.flatnonzero(blocked)
            row['forward_probe']=dict(point=probe.tolist(),radius=.33,proof=snap.ball(probe,.33,now),blocking_cells=[dict(center=flat[i].tolist(),state=int(snap.states.ravel()[i])) for i in selected])
            # Trace actual frozen A*; avoid duplicate certification inside its deadline.
            space=Space(snap,clock);original=space.ball_free;queries=[];pops=[0];expanded=[];orig_pop=heapq.heappop
            def ball(p,r):
                ok=original(p,r);queries.append(dict(point=np.asarray(p).tolist(),radius=r,passed=bool(ok)));return ok
            def pop(queue):
                pops[0]+=1;value=orig_pop(queue);expanded.append(list(value[1]));return value
            space.ball_free=ball;search=space.search_goal
            def traced_search(start,goal):
                row['astar_start']=np.asarray(start).tolist();row['astar_goal']=np.asarray(goal).tolist()
                result=search(start,goal);row['local_waypoint']=np.asarray(result).tolist();return result
            space.search_goal=traced_search;heapq.heappop=pop
            try:
                plan,proof=DepthPlanner(clock,space=space).propose(dict(item['pose'],a=[0,0,0]),np.array(row['task_goal']),now)
                row.update(status='CERTIFIED',plan=plan.record,proof=proof,profile=plan._planning_profile,outside_plan=bool((np.linalg.norm(sample_batch(plan,np.linspace(0,2,129))[:,0]-start,axis=1)>.6).any()))
            except Exception as ex:row.update(status='REJECTED',reason=str(ex),details=getattr(ex,'record',None))
            finally:heapq.heappop=orig_pop
            row['astar_heap_pops']=pops[0];row['expanded_lattice_nodes']=expanded;row['ball_queries']=len(queries);row['first_failed_edge_query']=None
            for query in queries:
                if not query['passed']:
                    diagnostic=snap.ball(query['point'],query['radius'],now);row['first_failed_edge_query']=dict(query,diagnostic=diagnostic);break
            row['query_trace']=queries[:512]
            row['origin_birth_conflict']=row['start_ball'].get('occupied',0)>0
        except Exception as ex:row.update(status='MAP_REJECTED',reason=str(ex))
        rows.append(row)
    write('ROUTE_FRAMES.json',rows);return rows

def commands(records):
    # Separate input and command timelines; polling occurs both before and after
    # each input. Record every gate, not only a boolean returned to an old harness.
    start=records[0]['receipt_ros'];wall=time.monotonic();clock=lambda:start+time.monotonic()-wall
    e=Engine(lambda n:BirthGrant((.25,-.25,2.),.6,n+.2,'VIRTUAL_ONLY'),clock);e.start();events=[];samples=[];inputs=[]
    def sample():
        n=clock()
        with e.lock:
            c=e.committed;state=e.state
            if c is None:why='NO_COMMITTED_PLAN'
            elif e.evidence_overflow or e.audit_overflow:why='OVERFLOW'
            elif e.revision!=c['map_epoch']:why='EPOCH_MISMATCH'
            elif not 0<=n-c['source_stamp']<=.2:why='SOURCE_AGE'
            elif n>c['proof']['valid_until']:why='PROOF_EXPIRED'
            elif not 0<=n-e.goal_stamp<=.2:why='GOAL_AGE'
            elif state is None or not 0<=n-state[1]<=.2:why='STATE_AGE'
            elif not 0<=n-c['plan'].epoch<=2.5:why='PLAN_LIFETIME'
            else:why='ELIGIBLE_BEFORE_COMMAND'
            row=dict(ros_time=n,monotonic=time.monotonic(),reason=why,revision=e.revision,queue=len(e.pending),closed=e.closed.is_set(),executed_plan=e.last_executed_plan.id if e.last_executed_plan else None)
            if c:row.update(plan_id=c['plan'].id,plan_epoch=c['plan'].epoch,source_stamp=c['source_stamp'],valid_until=c['proof']['valid_until'],map_epoch=c['map_epoch'])
        cmd=e.command(clock());row['valid']=cmd is not None
        if cmd is not None:e.published(cmd)
        samples.append(row);events.extend(e.take_events())
    for item in records:
        while clock()<item['receipt_ros']:sample();time.sleep(.001)
        try:
            e.set_state(dict(item['pose'],a=[0,0,0]),item['stamp']);e.set_goal([.53,-.25,2.],item['stamp']);e.observe(np.load(item['path'])['depth'],item['pose'],item['stamp'],item['stamp'])
            inputs.append(dict(stamp=item['stamp'],monotonic=time.monotonic(),receipt=clock(),revision=e.revision))
        except Exception as ex:inputs.append(dict(stamp=item['stamp'],monotonic=time.monotonic(),reason=str(ex)))
        sample()
    # No refreshed state/goal after the recording ends; the original freshness
    # bounds must expire. This tail is observed, not synthetic extra sensor data.
    end=clock()+.25
    while clock()<end:sample();time.sleep(.001)
    e.close();events.extend(e.take_events())
    report=dict(scope='OFFLINE_VIRTUAL_INPUT_WITH_PUBLICATION_STATE_NOT_ROS_OR_PLANT',samples=samples,inputs=inputs,events=events,counts=dict(collections.Counter(x['event'] for x in events)),gate_reasons=dict(collections.Counter(x['reason'] for x in samples)),valid_fraction=sum(x['valid'] for x in samples)/len(samples),physical_publisher=False)
    write('COMMAND_TIMELINE.json',report);return report

def main():
    if '--check-only' in sys.argv:
        binding=json.loads((OUT/'CURRENT_SOURCE_BINDING.json').read_text())
        for path,expected in binding.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('SOURCE_DRIFT '+path)
        print('SOURCE_BINDING_MATCH_NO_ROS_NO_ARM F1_PHYSICAL=NO_GO ROUTE_AND_NEW_QUALIFICATION_PENDING');return
    torch.set_num_threads(1);records=json.loads((ROOT/'research_outputs/forest_f1_blockers_20261118_v3/INPUTS.json').read_text())['records'];a=routes(records);b=commands(records);print('ROUTES',collections.Counter(x.get('reason',x['status']) for x in a));print('COMMANDS',b['counts'],b['gate_reasons'],b['valid_fraction'])
if __name__=='__main__':main()
