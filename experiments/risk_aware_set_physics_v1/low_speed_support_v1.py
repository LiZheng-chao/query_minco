"""Read-only, hash-verified empirical joint-state support audit."""
import collections,hashlib,json,csv
from pathlib import Path
import numpy as np
from query_minco.inference.shadow import rotation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/rsps_forest_shadow_v1'
MAN=ROOT/'research_outputs/unified_trainer_v1/tests/reference_qualification002_process/qualification/C1_orthogonal_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_training():
    m=json.loads(MAN.read_text());rows=[]
    for r in m['rows']:
        a=r['artifact'];assert sha(a['path'])==a['sha256']
        with np.load(a['path'],allow_pickle=False) as z:
            s=z['state'].reshape(12).copy()
        rows.append((r['id'],s,a))
    assert len(rows)==384 and len({r[0] for r in rows})==384
    return rows
def joint(s,unique):
    d=np.abs(unique-s)
    # Pre-existing Shadow diagnostic tolerances; neighbourhood, NOT learned support.
    near=(d[:,:3].max(1)<=.05)&(np.linalg.norm(d[:,3:6],axis=1)<=.05)&(d[:,6:9].max(1)<=.01)&(d[:,9:12].max(1)<=.01)
    return dict(exact=bool((d.max(1)==0).any()),near_joint_reference=bool(near.any()),
        nearest_group_scaled_linf=float(np.maximum.reduce([d[:,:3].max(1)/.05,np.linalg.norm(d[:,3:6],axis=1)/.05,d[:,6:9].max(1)/.01,d[:,9:12].max(1)/.01]).min()))
def replay(run,unique):
    ev=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    rows=[];prev=None;globalgoal=json.loads((run/'RUN_SPEC.json').read_text())['global_goal_W']
    for e in ev:
        if e['type'] not in ('FRAME','REJECT'):continue
        art=e.get('input_artifact') or e.get('attempt',{}).get('input_artifact')
        if art:
            with np.load(run/art,allow_pickle=False) as z:
                ns=int(z['stamp_ns']);v=z['velocity_W'];p=z['position_W'];R=rotation(z['q_xyzw'])
        elif e['type']=='FRAME':
            art='frame_%06d.npz'%e['index']
            with np.load(run/art,allow_pickle=False) as z:
                ns=int(z['stamp_ns']);v=z['velocity_W'];p=z['position_W'];R=z['R_WB0']
        else:
            # 30m rejected depth was NOT retained. State telemetry is original event data.
            a=e['attempt'];ns=e['stamp_ns'];v=np.array(a['velocity_W']);p=np.array(a['position_W']);R=rotation(a['q_xyzw'])
            art='events.jsonl:state_only_no_rejected_depth'
        delta=np.array(globalgoal)-p;dist=np.linalg.norm(delta);g=R.T@delta*2.4/dist
        row=dict(run=run.name,stamp_ns=ns,artifact=art,primary='ACCEPTED' if e['type']=='FRAME' else e['reason'],
             dynamic=bool(np.linalg.norm(v)>=.05),terminal=bool(dist<2.4),direction=bool(g[0]<=0 or (np.abs(g[1:]/g[0])>.175+1e-6).any()),
             speed=float(np.linalg.norm(v)),goal_y_slope=float(g[1]/g[0]),goal_z_slope=float(g[2]/g[0]))
        if prev:
            dt=(ns-prev[0])/1e9
            a=(v-prev[1])/dt
            state=np.r_[R.T@v,R.T@a,g,R.T@[0.,0.,-1.]].astype(np.float32)
            row.update(joint(state,unique),acceleration=float(np.linalg.norm(a)),gravity_deviation=float(np.max(np.abs(state[9:]-[0,0,-1]))),state=state.tolist(),dt=dt)
        prev=(ns,v.copy());rows.append(row)
    return rows
def summary(rows):
    dynamic=[r for r in rows if r['dynamic'] and not r['terminal']]
    return dict(attempts=len(rows),primary=dict(collections.Counter(r['primary'] for r in rows)),
        dynamic_nonterminal=len(dynamic),direction_rejects_dynamic=sum(r['direction'] for r in dynamic),
        exact_joint_hits=sum(r.get('exact',False) for r in dynamic),
        near_joint_hits=sum(r.get('near_joint_reference',False) for r in dynamic),
        speed_quantiles=np.quantile([r['speed'] for r in dynamic],[0,.5,.95,1]).tolist() if dynamic else [])
def main():
    dest=OUT/'low_speed_review001';train=load_training();s=np.stack([r[1] for r in train]);u,n=np.unique(s,axis=0,return_counts=True)
    corr=np.zeros((12,12));vary=np.std(s,axis=0)>1e-7;corr[np.ix_(vary,vary)]=np.corrcoef(s[:,vary],rowvar=False)
    result=dict(samples=384,unique_states=len(u),manifest_sha=sha(MAN),min=s.min(0).tolist(),max=s.max(0).tolist(),
        unique=[dict(state=a.tolist(),count=int(b)) for a,b in zip(u,n)],correlation=corr.tolist(),
        velocity_goal_pairs=len(np.unique(s[:,:9],axis=0)),near_reference_is_not_distribution_membership=True,
        artifacts=[dict(id=r[0],**r[2]) for r in train],runs={})
    for name in ['forest30m_shadow001','forest85m_shadow001','forest85s_shadow001']:
        rows=replay(OUT/name,u);result['runs'][name]=summary(rows)
        with (dest/(name+'_states.json')).open('x') as f:json.dump(rows,f,indent=2)
    with (dest/'TRAINING_AND_ROS_SUPPORT.json').open('x') as f:json.dump(result,f,indent=2)
    np.savez(dest/'TRAINING_STATES.npz',states=s,unique_states=u)
    print(json.dumps({k:v for k,v in result.items() if k in ('samples','unique_states','min','max','runs')},indent=2))
if __name__=='__main__':main()
