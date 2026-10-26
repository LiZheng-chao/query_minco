"""Independent offline replay; GT is never an input."""
import collections,csv,json
from pathlib import Path
import numpy as np
from query_minco.inference.online_selector import select
from experiments.risk_aware_set_physics_v1.low_speed_support_v1 import replay,summary,joint,OUT
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def main():
    dest=OUT/'low_speed_review001';u=np.load(dest/'TRAINING_STATES.npz')['unique_states'];rows=[];reports={}
    for name in ['forest30m_shadow001','forest85m_shadow001','forest85s_shadow001','low_speed_shadow001']:
        run=OUT/name;ev=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
        for e in ev:
            if e['type']!='FRAME':continue
            with np.load(run/('frame_%06d.npz'%e['index'])) as z:
                for label in ['C1','E3']:
                    out={k[len(label)+1:]:z[k] for k in z.files if k.startswith(label+'_')}
                    r=select(z['depth_input'],z['state'],out,e['output_age_s'])
                    assert r['selected_query'] is None and r['certified_count']==0
                    for c in r['candidates']:rows.append(dict(run=name,frame=e['index'],model=label,
                        **c,diagnostic_rank=r['diagnostic_ranking'].index(c['query']) if c['query'] in r['diagnostic_ranking'] else None))
        selected=[r for r in rows if r['run']==name]
        reports[name]=dict(candidates=len(selected),certified_free=0,selected=0,
            observability=dict(collections.Counter(r['observability'] for r in selected)),
            execution_gate=dict(collections.Counter(r['execution_gate'] for r in selected)),
            blocked_reasons=dict(collections.Counter(q for r in selected for q in r.get('reasons',[]))),
            point_unknown_p50_p95=np.quantile([r.get('point_unknown_fraction',1) for r in selected],[.5,.95]).tolist())
    with (dest/'SELECTOR_REPLAY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    dump(dest/'SELECTOR_REPLAY_SUMMARY.json',reports)
    state_rows=replay(OUT/'low_speed_shadow001',u)
    dump(dest/'LOW_SPEED_ALL_RAW_STATES.json',state_rows)
    decision=json.loads((OUT/'low_speed_host001/PILOT_DECISION.json').read_text())
    full=[r for r in state_rows if r['stamp_ns']/1e9>=decision['full_start_stamp']]
    low=dict(all=summary(state_rows),full=summary(full),
        full_near_joint_fraction=sum(r.get('near_joint_reference',False) for r in full)/len(full),
        full_state_groups_quantiles={k:np.quantile([r[k] for r in full],[0,.5,.95,1]).tolist() for k in ['speed','acceleration','gravity_deviation','goal_y_slope','goal_z_slope']})
    dump(dest/'LOW_SPEED_SUPPORT_RESULT.json',low)
    print(json.dumps(dict(selector=reports,low_speed=low),indent=2))
if __name__=='__main__':main()
