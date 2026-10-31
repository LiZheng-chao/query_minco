"""Run registered low-speed rig faults only after the .4m/s physical-stop pilot passes."""
import argparse,json
from .rig import BASE,run,FAULTS

def main():
 p=argparse.ArgumentParser();p.add_argument('--enable-isolated-dynamics-test',action='store_true');a=p.parse_args()
 if not a.enable_isolated_dynamics_test:raise ValueError('DEFAULT_DISABLED')
 pilot=json.loads((BASE/'SYSTEM/v400_publisher_exit_r6/GATEWAY_RESULT.json').read_text());assert pilot['status']=='DYNAMICS_STOP_PASS'
 names=['v400_'+x for x in FAULTS if x!='publisher_exit']+['v800_'+x for x in FAULTS]+['semantics_ready','semantics_feedback','domain_shadow'];records=[]
 for name in names:
  r=run(name);records.append(dict(case=name,status=r['status'],trigger=r.get('trigger'),distance=r.get('maximum_stop_displacement_m'),FALLBACK_VERIFIED=r.get('FALLBACK_VERIFIED')));print('RIG_CASE',name,r['status'],flush=True)
  if r['status']!='DYNAMICS_STOP_PASS':break
 (BASE/'RIG_SUITE_RESULT.json').write_text(json.dumps(dict(records=records,planned=names,completed=len(records),full_StageC='BLOCKED_NO_SENSOR_SPACE_OR_MOTOR_BOUND_QUALIFICATION'),indent=2))
if __name__=='__main__':main()
