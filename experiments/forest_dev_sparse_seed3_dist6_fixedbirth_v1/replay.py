"""Fixed-scene native Depth replay. No ROS/control/GT input."""
from pathlib import Path
import sys,json,collections,time,os
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
BASE=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1';O=BASE/'replay_ordered'
import torch,numpy as np
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
OUT=O
def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2,allow_nan=False))
def commands(records):
    # Separate input and command timelines; polling occurs both before and after
    # each input. Record every gate, not only a boolean returned to an old harness.
    start=records[0]['receipt_ros'];wall=time.monotonic();clock=lambda:start+time.monotonic()-wall
    e=Engine(lambda n:BirthGrant((.25,-.25,2.),.6,n+.2,'NONPHYSICAL_FIXED_BIRTH_ONLY'),clock);e.start();events=[];samples=[];inputs=[]
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
            e.set_state(dict(item['pose'],a=[0,0,0]),item['stamp']);e.set_goal([3.25,-.25,2.],item['stamp']);e.observe(np.load(item['path'])['depth'],item['pose'],item['stamp'],item['stamp'])
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

from experiments.forest_f1_v1 import route_command_diagnosis as diagnostic

def main():
 O.mkdir();torch.set_num_threads(1);diagnostic.OUT=O
 records=json.loads((BASE/'capture_ordered/INPUTS.json').read_text())['records']
 # Reuse existing exact full-tube route diagnostics, never use exported GT.
 before=time.perf_counter();rows=diagnostic.routes(records)
 (O/'GEOMETRY_RUN.json').write_text(json.dumps(dict(exit_code=0,elapsed_s=time.perf_counter()-before,reasons=dict(collections.Counter(x.get('reason',x['status']) for x in rows))),indent=2))
 # Same engine command recorder, with the fixed experiment goal. The historical
 # recorder's 0.28 m goal belongs to the earlier F0 continuity diagnostic only.
 report=commands(records)
 print('ROUTES',collections.Counter(x.get('reason',x['status']) for x in rows));print('REALTIME',report['counts'],report['valid_fraction'])
if __name__=='__main__':main()
