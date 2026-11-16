from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.metrics import summarize
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1';rows=[]
for name in ['g4cp1_manual_held_20261111_164653_1005','g4cp1_manual_hover_retry_20261111_165004_1005']:
 folder=R/'research_outputs'/name;accept=next(folder.glob('ACCEPTANCE_*.json'));original=json.loads(accept.read_text())
 raw=[json.loads(x) for x in (folder/'preparation/EVENTS.jsonl').read_text().splitlines()]
 events=[]
 for e in raw:
  if e.get('kind') in ['odom','imu','depth','request','raw','plant']:
   events.append(dict(kind=e['kind'],stamp=e['stamp'],ros_now=e['ros_now'],monotonic=e['monotonic'],accepted=e['data']['accepted'],payload=e['data']['payload']))
  if e.get('kind')=='transition' and e['mode']=='ABORT':events.append(dict(kind='fault',monotonic=e['monotonic'],reason=e['data']['reason']))
 starts=[e['monotonic'] for e in raw if e.get('kind')=='plant' and e['data']['payload'].get('integration_steps',0)>0]
 begin=min(starts) if starts else -float('inf')
 summary=summarize(events,begin=begin,anchor=[0,0,2])
 children={p.parent.name:json.loads(p.read_text()) for p in folder.glob('*/PROCESS_RESULT.json')}
 cleanup_fail=[k for k,v in children.items() if k!='plant' and v['actual_exit_code']!=0]
 row=dict(run=name,artifact_sha256=sha(accept),events_sha256=sha(folder/'preparation/EVENTS.jsonl'),original_status=original['status'],
  original_recorded_max_steps=original['max_integration_steps'],full_event_window=summary,cleanup_status='FAIL' if cleanup_fail else 'PASS',
  unexpected_child_results=cleanup_fail,physical_window_status='PASS' if original['stage']=='B_HOVER' and summary['max_integration_steps']>=5000 and summary['maxima']['speed']<=1 and summary['maxima']['acceleration']<=8 and summary['maxima']['position_drift']<=.25 else ('PASS_HELD_ZERO_INTEGRATION' if not starts else 'FAIL'),
  historical_events_missing_stop_timestamp='P1 terminal record lacks absolute stop_monotonic;post-stop exact attribution cannot be retroactively certified')
 rows.append(row)
write(O/'P1_FULL_WINDOW_REPLAY.json',rows)
b=rows[1];assert b['physical_window_status']=='PASS'
write(O/'LEGACY_B_PHYSICAL_WINDOW.json',dict(status='PASS',scope='PHYSICAL_WINDOW_ONLY_NOT_CLEANUP',original_acceptance=str(R/'research_outputs'/b['run']/'ACCEPTANCE_B.json'),
 original_acceptance_sha256=b['artifact_sha256'],events_sha256=b['events_sha256'],cleanup_status=b['cleanup_status'],metrics=b['full_event_window']))
print('LEGACY_PHYSICAL_WINDOW_PASS_CLEANUP_FAIL',[(r['run'],r['cleanup_status'],r['full_event_window']['max_integration_steps']) for r in rows])
