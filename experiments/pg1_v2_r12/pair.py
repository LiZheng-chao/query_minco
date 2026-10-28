"""Compare read-only per-checkpoint exports; never trains."""
import argparse,json,csv
from pathlib import Path
from query_minco.training.io import read,write,artifact,require
from query_minco.training.pg1_v2_r12.spec import R0,RTAIL
p=argparse.ArgumentParser();p.add_argument('--r0',type=Path,required=True);p.add_argument('--rtail',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=read(a.r0/'SUMMARY.json');y=read(a.rtail/'SUMMARY.json')
require(x['method'] in (R0,'query_minco_pg1_task_complete_v1') and y['method']==RTAIL,'R12_PAIR_METHODS')
bx=read(artifact(x['bindings']));by=read(artifact(y['bindings']))
require({k:v for k,v in bx.items() if k!='code'}=={k:v for k,v in by.items() if k!='code'},'R12_PAIR_DATA')
rx=Path(x['run']);ry=Path(y['run'])
spx=read(rx/'RUN_SPEC.json');spy=read(ry/'RUN_SPEC.json')
require(spx['seed']==spy['seed']==42 and spx['device']==spy['device']=='cpu' and spx['batch_size']==spy['batch_size']==8 and spx['precision']==spy['precision']=='FP32','R12_PAIR_RUNTIME_CONTRACT')
require(spx['optimizer']==spy['optimizer'] and spx['schedule']==spy['schedule'],'R12_PAIR_OPTIMIZER')
cx=spx['code']['files'];cy=spy['code']['files']
require(all(cx[k]==cy[k] for k in set(cx)&set(cy)),'R12_PAIR_COMMON_SOURCE_SHA')
for index in range(1,33):
 ux=read(rx/'commits'/('step%04d'%index)/'UPDATE.json');uy=read(ry/'commits'/('step%04d'%index)/'UPDATE.json')
 require(ux['ids']==uy['ids'] and ux['lr']==uy['lr'] and ux['completed_optimizer_updates']==uy['completed_optimizer_updates']==index,'R12_PAIR_ACTUAL_ORDER_LR')
a.output.mkdir(parents=True,exist_ok=False);rows=[];summary=[]
for step in [0,8,16,32]:
 sx=read(a.r0/('step%04d.json'%step));sy=read(a.rtail/('step%04d.json'%step))
 if step==0:require(sx['weights_sha256']==sy['weights_sha256'],'R12_PAIR_INITIAL')
 xx=sx['evaluation']['r12_candidate_details'];yy=sy['evaluation']['r12_candidate_details']
 require([(r['sample_id'],r['query']) for r in xx]==[(r['sample_id'],r['query']) for r in yy],'R12_PAIR_TASK_QUERY')
 for b,c in zip(xx,yy):
  row=dict(step=step,id=b['sample_id'],query=b['query'])
  for name,r in [('R0',b),('Rtail',c)]:
   for k in ['hard_pass','clearance_m','minimum_safety_margin_m','vmax','amax','jmax','time_s','matched','heuristic_mode','acceleration_energy_m2_s3','jerk_energy_m2_s5','tail','obstacle']:
    row[name+'_'+k]=r[k]
   row[name+'_goal_pass']=r['goal']['passed'];row[name+'_progress_m']=r['goal']['projected_progress_m']
  rows.append(row)
 summary.append(dict(step=step,R0={k:sx['evaluation'][k] for k in ['physics_rate','goal_rate','oracle_rate','feasible_mode_count']},Rtail={k:sy['evaluation'][k] for k in ['physics_rate','goal_rate','oracle_rate','feasible_mode_count']}))
with (a.output/'PAIRED_CANDIDATES.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(a.output/'PAIRED_SUMMARY.json',dict(records=summary,evaluation='open-dev8 diagnostic only; not independent final test',optimizer_steps=0,training_benefit='determine from actual paired metrics; never assume'))
