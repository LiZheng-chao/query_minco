import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.observed_space_v2.core import ObservedMap
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot
def main():
 out=Path('research_outputs/t2_depth_minco_20261116_v1')
 row=json.loads((out/'frozen_replay_v2/RESULT.json').read_text())['rows'][0]
 path=Path(row['path']);run=path.parents[1];stamp=row['stamp']
 events=[json.loads(x) for x in (run/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
 odom=next(x for x in events if x.get('kind')=='odom' and x.get('stamp')==stamp and x.get('accepted'))
 dep=next(x for x in events if x.get('kind')=='depth' and x.get('stamp')==stamp and x.get('accepted'))
 truth=[json.loads(x) for x in (run/'truth_data/EVENTS.jsonl').read_text().splitlines()]
 truth=[x for x in truth if x.get('kind')=='truth' and x['status']=='PASS' and 0<=dep['monotonic']-x['monotonic']<=.2][-1]
 args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
 grant=BirthGrant.from_records((run/'BIRTH_CERTIFICATE.json').read_bytes(),truth,args,json.loads((out/'BIRTH_SELECTION.json').read_text()),stamp,dep['monotonic'])
 ledger=ObservedMap();ledger.add(np.load(path)['depth'],odom['payload'],stamp,stamp,stamp)
 sensor=ledger.snapshot(stamp);snap=authorized_snapshot(sensor,grant,stamp,1,lambda:1)
 start=np.array(odom['payload']['p']);known={(0,0)};todo=[(0,0)];edges=[];blocked=[]
 while todo and len(known)<2000:
  u=todo.pop(0);p=start+np.array([u[0]*.2,u[1]*.2,0])
  for dx in [-1,0,1]:
   for dy in [-1,0,1]:
    if not (dx or dy):continue
    v=(u[0]+dx,u[1]+dy)
    if abs(v[0])>15 or abs(v[1])>10 or v in known:continue
    q=start+np.array([v[0]*.2,v[1]*.2,0])
    checks=[snap.ball(x,.33,stamp) for x in np.linspace(p,q,max(2,int(np.linalg.norm(q-p)/.05)+1))]
    bad=next((x for x in checks if not x['passed']),None)
    if bad:blocked.append(dict(edge=[u,v],first_rejection=bad));continue
    known.add(v);todo.append(v);edges.append([u,v])
 result=dict(scope='OBSERVABILITY_GRAPH_DIAGNOSTIC_NOT_CONTINUOUS_PHYSICAL_INFEASIBILITY_PROOF',
   source=str(path),radius=.33,step_m=.2,edge_check_spacing_m=.05,geometry='sensor+finite_birth_only',
   reachable_centers=sorted(list(known)),max_forward_m=max(x[0]*.2 for x in known),
   reached_local_goal=any(np.linalg.norm(np.array([x*.2,y*.2])-[.6,0])<.25 for x,y in known),
   blocked_edges=blocked,birth_cells=snap.birth_cells,
   interpretation='No route in the frozen conservative certified graph is not proof that the physical task is impossible.')
 (out/'OBSERVABILITY_DIAGNOSIS.json').write_text(json.dumps(result,indent=2))
 print('OBSERVED GRAPH',len(known),'MAX_X',result['max_forward_m'],'GOAL',result['reached_local_goal'])
if __name__=='__main__':main()
