"""Read-only startup evidence timeline and source-identity diagnosis."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1';P=R/'research_outputs/g4b_researcher_physics_20261111_v1/safety'
def main():
 old=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
 frozen=json.loads((old/'FILES_AND_SHA.json').read_text());binding=json.loads((P/'RESULT.json').read_text())['source_binding']
 mismatches=[f for f,h in binding.items() if not Path(f).is_file() or sha(f)!=h]
 source_snapshot={**frozen['unchanged_old_source_sha256'],**{str(R/k):v['sha256'] for k,v in frozen['source_new'].items()}}
 snapshot_mismatch=[f for f,h in source_snapshot.items() if not Path(f).is_file() or sha(f)!=h]
 e=P/'timeout';feedback=json.loads((e/'feedback/RESULT.json').read_text());lease=json.loads((e/'lease/RESULT.json').read_text())
 offset=sum(x['ros_stamp']-x['monotonic'] for x in lease['records'])/len(lease['records']);timeline=[]
 for topic,rows in feedback['records'].items():
  for row in rows:timeline.append(dict(monotonic=row['monotonic'],ros_estimated=row['monotonic']+offset,source_stamp=row['stamp'],source='feedback/'+topic,details=json.dumps(row)))
 for row in lease['records']:timeline.append(dict(monotonic=row['monotonic'],ros_estimated=row['ros_stamp'],source_stamp=row['ros_stamp'],source='lease',details=json.dumps(row)))
 for row in lease['transitions']:timeline.append(dict(monotonic=row['monotonic'],ros_estimated=row['monotonic']+offset,source_stamp=None,source='lease_transition',details=json.dumps(row)))
 processes=[]
 for f in e.glob('*/PROCESS_START.json'):
  st=json.loads(f.read_text());end=json.loads((f.parent/'PROCESS_RESULT.json').read_text());processes.append(dict(component=f.parent.name,start=st,result=end))
  timeline.append(dict(monotonic=st['started_unix']-offset,ros_estimated=st['started_unix'],source_stamp=None,source='process_start/'+f.parent.name,details=json.dumps(st['argv'])))
 timeline.sort(key=lambda x:x['monotonic']);t0=next(x['start']['started_unix']-offset for x in processes if x['component']=='physics')
 for x in timeline:x['since_physics_popen_s']=x['monotonic']-t0
 with (O/'STARTUP_UNIFIED_TIMELINE.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(timeline[0]));w.writeheader();w.writerows(timeline)
 counts={k:len(v) for k,v in feedback['records'].items()}
 missing={k:'NOT_RECORDED in independent observer;lease indicates receipt state only' for k in ['depth','request','so3_cmd'] if k not in counts}
 imu=feedback['records'].get('imu',[]);odom=feedback['records'].get('odom',[])
 outcome=dict(status='EVIDENCE_DIAGNOSED',current_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  g4b_snapshot_source_mismatches=snapshot_mismatch,physical_source_binding_mismatches=mismatches,
  source_identity='PASS' if not mismatches and not snapshot_mismatch else 'FAIL',
  ros_monotonic_offset_s=offset,clock_alignment='lease paired clocks;Popen/unix alignment approximate,not first integration timestamp',
  actual_processes=processes,observed_counts=counts,missing_recordings=missing,imu_all_stamp_zero=bool(imu) and all(x['stamp']==0 for x in imu),
  first_odom=odom[0] if odom else None,last_odom=odom[-1] if odom else None,
  first_imu=imu[0] if imu else None,first_abort=lease['transitions'][0],ever_active=any(x['mode']=='ACTIVE' for x in lease['records']),
  ever_final_gate_open=any(x['final_so3_gate_open'] for x in lease['records']),
  held_before_integration=False,physical_tracking='FAILED_STARTUP_NOT_TRACKING',new_physics_execution=False)
 write(O/'STARTUP_EVIDENCE_DIAGNOSIS.json',outcome)
 print(json.dumps({k:v for k,v in outcome.items() if k not in ['actual_processes']},indent=2));assert outcome['source_identity']=='PASS'
if __name__=='__main__':main()
