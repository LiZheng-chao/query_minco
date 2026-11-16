"""Read-only raw-log comparison;output must be a new exclusive directory."""
from pathlib import Path
import argparse,json,csv
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def read(path):
 return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
probe=read(root/'sensor_probe/EVENTS.jsonl');relay=read(root/'relay_data/EVENTS.jsonl');sup=read(root/'supervisor_data/EVENTS.jsonl')
def index(rows,kind,key='source_stamp'):return {x[key]:x for x in rows if x.get('kind')==kind and key in x}
stages=dict(renderer=index(probe,'renderer'),probe=index(probe,'depth'),relay=index(relay,'depth'),supervisor_arrival=index(sup,'depth_arrival'),decision=index(sup,'depth_chain'))
stamps=sorted(set().union(*[set(v) for v in stages.values()]));rows=[]
for s in stamps:
 pieces={k:v.get(s) for k,v in stages.items()}
 decision=pieces['decision'];rel=pieces['relay']
 status='ACCEPTED' if decision and decision.get('accepted') else decision.get('reason') if decision else 'RELAY_FILTERED' if rel and not rel['published'] else 'NO_SUPERVISOR_CALLBACK_RECORD' if rel and not pieces['supervisor_arrival'] else 'NO_RELAY_INPUT_RECORD' if pieces['probe'] and not rel else 'PENDING_OR_UNRECORDED'
 rows.append(dict(source_stamp=s,status=status,stages=pieces))
(out/'CHAIN_ROWS.json').write_text(json.dumps(rows,indent=2))
counts={}
for row in rows:counts[row['status']]=counts.get(row['status'],0)+1
(out/'SUMMARY.json').write_text(json.dumps(dict(frames=len(rows),classifications=counts,caveat='Missing records alone do not prove ROS packet loss;compare stop/teardown boundaries. Actual ROS internal queue occupancy is unavailable.'),indent=2))
print(json.dumps(counts,indent=2))
