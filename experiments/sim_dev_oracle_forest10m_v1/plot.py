"""Plot only actual logged native odom and recorded MINCO; never invent movement."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from query_minco.integration.yopo_execution_v4.trajectory import Plan

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 file=a.run/'supervisor_data/EVENTS.jsonl'
 if not file.exists():file=a.run/'none/supervisor_data/EVENTS.jsonl'
 rows=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
 od=[e for e in rows if e.get('kind')=='odom' and e.get('accepted') and 'p' in e.get('payload',{})]
 if not od:raise ValueError('NO_REAL_ODOM')
 t=np.array([e['stamp'] for e in od]);xyz=np.array([e['payload']['p'] for e in od]);vel=np.array([e['payload']['v'] for e in od]);plans={}
 for e in rows:
  q=e.get('payload',{})
  if q.get('event')=='CANDIDATE_GENERATED' and 'plan' in q:
   plan=Plan(q['plan'])
   if plan.id!=q['plan_id']:raise ValueError('PLAN_ID_MISMATCH')
   plans[plan.id]=plan
 a.output.mkdir(parents=True,exist_ok=False)
 for name,i,j in [('xy',0,1),('xz',0,2)]:
  fig,ax=plt.subplots();ax.plot(xyz[:,i],xyz[:,j],color='orange',label='actual native odom');ax.set_xlabel('xyz'[i]+' (m)');ax.set_ylabel('xyz'[j]+' (m)');ax.legend();fig.savefig(a.output/(name+'.png'));plt.close(fig)
 fig,axes=plt.subplots(2,1);axes[0].plot(t-t[0],xyz[:,0]-xyz[0,0]);axes[0].set_ylabel('forward progress (m)');axes[1].plot(t-t[0],np.linalg.norm(vel,axis=1));axes[1].set_ylabel('speed (m/s)');axes[1].set_xlabel('source ROS time since start (s)');fig.savefig(a.output/'progress_speed.png');plt.close(fig)
 fig,ax=plt.subplots()
 for plan in plans.values():
  q=np.array([plan.sample(s)[0] for s in np.linspace(0,2,81)]);ax.plot(q[:,0],q[:,1],color='blue',alpha=.3)
 ax.plot(xyz[:,0],xyz[:,1],color='orange',label='actual');ax.set_title('Recorded candidates (blue); actual odom (orange)');fig.savefig(a.output/'plan_actual.png');plt.close(fig)
 (a.output/'METRICS.json').write_text(json.dumps(dict(input=str(file),odom_samples=len(od),recorded_candidates=len(plans),forward_progress=float(xyz[-1,0]-xyz[0,0]),travel_distance=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),physical_success_claimed=False),indent=2))
if __name__=='__main__':main()
