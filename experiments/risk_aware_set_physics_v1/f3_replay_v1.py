"""Bounded CPU replay of sealed F1/F2 tensors; no rendering, GT selection or control."""
import argparse,collections,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from query_minco.inference.shadow import K,FrozenPair,FROZEN,sha
from query_minco.inference.observed_space import DepthFrame,DepthHistory,FreeVolume,box_cell,Scope
from query_minco.inference.selector_safety import select_with_safety,world_coefficients
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.trajectory_execution import *
from query_minco.trajectory import TrajectoryPostprocessor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_dev001'
F2=ROOT/'research_outputs/rsps_forest_shadow_v1'
def dump(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)
def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(OUT/'replay_final001'))
    args=parser.parse_args();dest=Path(args.run).resolve()
    assert dest.parent==OUT.resolve(),'EXCLUSIVE_REPLAY_OUTPUT_PARENT'
    dest.mkdir();OUT=dest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pair=FrozenPair(sha(FROZEN));post=TrajectoryPostprocessor();records=[];failures=collections.Counter();input_hashes={};execution=[]
    start=time.monotonic()
    def process(name,task,label,depth,state,out,origin,R,stamp,age,history):
        result=select_with_safety(depth,state,out,age,history,stamp+age,origin,R)
        assert result['selected_query'] is None
        for r in result['candidates']:
            qi=r['query'];pts=out['uniform_p'][qi]@R.T+origin
            diag=history.diagnostics(pts,stamp+age)
            diag['point_coverage_reason']=diag.pop('reason')
            records.append(dict(group=name,task=task,model=label,query=qi,status=r['status'],reason=r['safety_reason'],
                kinematic_reasons='|'.join(r.get('reasons',[])),**diag))
        # Distinct external-supervisor execution diagnostic on frozen low-speed outputs.
        # This is NEVER counted as an online accepted candidate.
        if name=='low_speed_shadow001':
            external=FreeVolume((box_cell([-2,-2,0],[40,2,4],stamp+10,'EXTERNAL_EMPTY_ROOM_TEST_FIXTURE'),),Scope.EXTERNAL_SUPERVISOR)
            for qi in range(6):
                p=Program(out['coefficients'][qi],out['durations'][qi])
                m=Measurement(stamp,np.stack([origin,R@state[:3],R@state[3:6]]))
                try:
                    scheduled=prepare(p,stamp,origin,R,m,stamp+.02,stamp+.08,external)
                    row=dict(task=task,model=label,query=qi,result='PASS_EXTERNAL_FIXTURE_ONLY',extrema=scheduled.program.extrema(),duration=scheduled.program.duration)
                except ValueError as e:row=dict(task=task,model=label,query=qi,result=str(e))
                execution.append(row)
    f1=ROOT/'research_outputs/rsps_forest15m_v1/forest_paired001'
    seal=json.loads((f1/'RUN_SEAL.json').read_text())['files']
    for p in sorted((f1/'inputs').glob('*.npz')):
        assert sha(p)==seal[str(p.relative_to(f1))];input_hashes[str(p)]=sha(p)
        with np.load(p) as z:depth=z['depth_input'];state=z['state']
        history=DepthHistory();history.add(DepthFrame(0.,depth,np.zeros(3),np.eye(3),'F1_OFFLINE_SNAPSHOT_'+p.stem),K)
        for label in ['C1','E3']:
            res=f1/'results'/(p.stem+'_'+label+'.npz');assert sha(res)==seal[str(res.relative_to(f1))];input_hashes[str(res)]=sha(res)
            with np.load(res) as z:
                raw=torch.from_numpy(z['raw'].copy());s=torch.from_numpy(state[None].copy())
                with torch.inference_mode():o=post(raw,s,num_samples=65);full=InferenceAdapter.export(o,raw,o.samples)
                for key in ['coefficients','durations','head_pva','tail_pva','inner_pts']:
                    assert np.array_equal(full[key],z[key]),'F1_REHYDRATION '+key
                out={k:v[0] for k,v in full.items()}
            process('F1',p.stem,label,depth,state,out,np.zeros(3),np.eye(3),0.,0.,history)
    for name in ['forest30m_shadow001','forest85m_shadow001','forest85s_shadow001','low_speed_shadow001']:
        run=F2/name;history=DepthHistory()
        events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
        for e in events:
            if e['type']!='FRAME':continue
            p=run/('frame_%06d.npz'%e['index']);input_hashes[str(p)]=sha(p)
            with np.load(p) as z:
                d=z['depth_input'];s=z['state'];origin=z['position_W'];R=z['R_WB0'];stamp=e['stamp']
                history.add(DepthFrame(stamp,d,origin,R,name+'_'+str(e['index'])),K)
                for label in ['C1','E3']:
                    out={k[len(label)+1:]:z[k] for k in z.files if k.startswith(label+'_')}
                    process(name,str(e['index']),label,d,s,out,origin,R,stamp,e['output_age_s'],history)
    with (OUT/'SELECTOR_REPLAY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    reports={}
    for group in ['F1','forest30m_shadow001','forest85m_shadow001','forest85s_shadow001','low_speed_shadow001']:
        rows=[r for r in records if r['group']==group]
        reports[group]=dict(candidates=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),
            gate_reasons=dict(collections.Counter(r['kinematic_reasons'] or r['reason'] for r in rows)),
            point_front_p50_p95=np.quantile([r['point_front_fraction'] for r in rows],[.5,.95]).tolist(),accepted=0)
    dump('SELECTOR_REPLAY_SUMMARY.json',dict(groups=reports,total=len(records),GT_used=False,
        full_body_certified=0,selected=0,F1_timestamp='offline snapshot fixture clock; not ROS age evidence',
        elapsed_s=time.monotonic()-start,scope='native depth only, external supervisor excluded'))
    dump('REPLAY_INPUT_SHA256.json',input_hashes)
    dump('EXECUTION_EXTERNAL_FIXTURE_REPLAY.json',dict(cases=len(execution),results=dict(collections.Counter(x['result'] for x in execution)),rows=execution,
        safety='fabricated empty-room fixture for CPU execution diagnostics only, NOT a real forest guarantee or online accepted candidates'))
    print(json.dumps(dict(selector=reports,execution=collections.Counter(x['result'] for x in execution),elapsed_s=time.monotonic()-start),indent=2))
if __name__=='__main__':main()
