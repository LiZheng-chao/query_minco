"""Read-only C7 evidence summary; never launches any process."""
import argparse,json,hashlib
from pathlib import Path
CASES=['request_timeout','depth_loss','odom_rollback','nan_control','reference_exit','controller_exit','truth_failure']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args()
 root=Path(a.campaign);final=json.loads((root/'RESULT.json').read_text())
 binding=json.loads((Path(__file__).resolve().parents[2]/'research_outputs/t1_stop_manual_20261116_v1/SOURCE_BINDING_NEW.json').read_text())
 errors=[];rows=[]
 if final.get('status')!='PASS' or final.get('source_binding')!=binding or [x['case'] for x in final.get('episodes',[])]!=CASES:errors.append('CAMPAIGN_NOT_COMPLETE_SAME_VERSION_PASS')
 for row in final.get('episodes',[]):
  path=Path(row['result_path']);d=json.loads(path.read_text());folder=path.parent
  plant=json.loads((folder/'plant/PROCESS_RESULT.json').read_text())
  owner=json.loads((folder/'OWNER_STOP_TIMELINE.json').read_text())
  fault=json.loads((folder/'FAULT.json').read_text())
  elapsed=d['terminal'].get('exit_observed_monotonic',float('inf'))-fault['monotonic']
  ok=(sha(path)==row['result_sha256'] and row['status']=='PASS' and d['status']=='PASS'
      and d['source_binding']==binding and d['physical_feedback']
      and d['physics_steps_at_qualification']>=500 and fault['armed'] and fault['steps']>0
      and elapsed<=.350 and d['cleanup_status']=='PASS' and d['runtime_status']=='PASS'
      and d['terminal']['terminal_state']=='OWNED_PROCESS_TERMINATED'
      and plant['actual_exit_code'] is not None
      and any(x['event']=='PLANT_EXIT_CONFIRMED' for x in owner))
  rows.append(dict(case=row['case'],checked=ok,fault_to_actual_exit_ms=elapsed*1000,
                   plant_exit_code=plant['actual_exit_code'],physics_steps=d['physics_steps_at_qualification'],
                   owner_timeline=str(folder/'OWNER_STOP_TIMELINE.json'),
                   stop_timeline=str(folder/'PLANT_STOP_EVENTS.json'),truth=str(folder/'truth_data/EVENTS.jsonl')))
  if not ok:errors.append(row['case'])
 print(json.dumps(dict(status='PASS' if not errors else 'FAIL',rows=rows,errors=errors,
     caveat='Read full raw timelines and verify expected fault cause; elapsed uses conservative fault publication timestamp. No new physical qualification synthesized.'),indent=2))
 raise SystemExit(1 if errors else 0)
if __name__=='__main__':main()
