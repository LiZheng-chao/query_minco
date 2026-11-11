"""Read frozen trajectory arrays for descriptive metrics; no planner/optimizer construction."""
from pathlib import Path
import json,csv,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import Counter
import numpy as np
import torch
from query_minco.evaluation.forest_adapter_v1 import ForestAdapter
from query_minco.evaluation.validation import evaluate
from query_minco.evaluation.mirror_holdout_v1 import write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
def main():
 torch.set_num_threads(1)
 data=json.loads((O/'FOREST_RESULTS.json').read_text());idx={r['snapshot_id']:r for r in json.loads((O/'dataset/FOREST_INDEX.json').read_text())}
 arr=np.load(O/'FOREST_TRAJECTORIES.npz');cache={};rows=[]
 for row in data['records']:
  sid=row['snapshot_id']
  if sid not in cache:cache[sid]=ForestAdapter.load_snapshot(idx[sid]['manifest'],idx[sid]['manifest_sha256'])
  b=cache[sid];g=b.geometry[0];key=sid+'__'+row['method']
  c=arr[key+'_coeff'][0,0];dt=arr[key+'_durations'][0,0]
  points=[];vel=[];acc=[];jerk=[];times=[];offset=0.
  for coef,T in zip(c,dt):
   t=np.linspace(0,float(T),201)
   points.append(evaluate(coef,t));vel.append(evaluate(coef,t,1));acc.append(evaluate(coef,t,2));jerk.append(evaluate(coef,t,3));times.append(t+offset);offset+=T
  pts=np.concatenate(points);v=np.concatenate(vel);a=np.concatenate(acc);j=np.concatenate(jerk);ts=np.concatenate(times)
  world=pts@g.rotation.T+g.origin;known=((world>=g.roi[0])&(world<=g.roi[1])).all(1)
  distances=g.sdf(torch.tensor(pts[known],dtype=torch.float64)).numpy() if known.any() else np.array([])
  sampled=float(distances.min()) if len(distances) else None
  continuous_lower=sampled-row['hard']['dynamic_polynomial_maxima'][0]*float(max(dt))/400-1e-6 if known.all() else None
  h=row['hard'];goal=h['goal'];surface_sampled=sampled-.12 if sampled is not None else None
  rec=dict(snapshot_id=sid,map_id=row['map_id'],seed=row['seed'],density=row['density'],task=row['task'],classification=row['preclassification'],method=row['method'],
   original_hard_passed=h['passed'],physical_hard_without_task=not any(r!='GOAL_TASK' for r in h['reasons']),hard_task=row['hard_task'],
   reasons='|'.join(h['reasons']),certificate_reasons='|'.join(x['reason'] for x in h['certificates']),
   speed_peak=h['dynamic_polynomial_maxima'][0],acceleration_peak=h['dynamic_polynomial_maxima'][1],jerk_peak=h['dynamic_polynomial_maxima'][2],
   projected_progress_m=goal['projected_progress_m'],goal_error_m=goal['endpoint_goal_error_m'],terminal_ray_offset_m=goal['terminal_ray_offset_m'],terminal_speed_mps=goal['terminal_speed_mps'],
   endpoint_y_m=float(pts[-1,1]),center_clearance_sampled_m=sampled,surface_clearance_sampled_m=surface_sampled,
   center_clearance_continuous_lower_bound_m=continuous_lower,surface_clearance_continuous_lower_bound_m=continuous_lower-.12 if continuous_lower is not None else None,
   exact_sdf_sampled_margin_violation=sampled<.15 if sampled is not None else None,
   continuous_clearance_additional_bound_passed=continuous_lower>=.15 if continuous_lower is not None else False,
   sdf_unknown_sample_fraction=float((~known).mean()),p0c_unknown_center_sample_fraction=row['unknown_center_sample_fraction'],
   hard_unknown=any(x['reason'] in ['UNKNOWN_OR_ROI_MARGIN','continuous_clearance_unproven'] for x in h['certificates']),
   path_length_quadrature_m=float(np.trapz(np.linalg.norm(v,axis=1),ts)),
   acceleration_squared_integral_quadrature=float(np.trapz((a*a).sum(1),ts)),
   jerk_squared_integral_quadrature=float(np.trapz((j*j).sum(1),ts)),loss=row['loss'],**row['timings'])
  for k,value in row['terms'].items():rec['L0_'+k]=float(np.array(value).item())
  rows.append(rec)
 with (O/'FOREST_RESULTS.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={}
 for method in sorted({r['method'] for r in rows}):
  xs=[r for r in rows if r['method']==method and r['classification']=='CERTIFIED_FEASIBLE']
  stats=dict(n=len(xs),hard_task=sum(r['hard_task'] for r in xs),physical_hard_without_task=sum(r['physical_hard_without_task'] for r in xs),
   failures=[dict(snapshot_id=r['snapshot_id'],reasons=r['reasons'],certificates=r['certificate_reasons'],clearance_sampled=r['center_clearance_sampled_m']) for r in xs if not r['hard_task']],
   by_density={d:dict(n=sum(r['density']==d for r in xs),hard_task=sum(r['hard_task'] for r in xs if r['density']==d)) for d in ['sparse','medium','dense']},
   by_task={d:dict(n=sum(r['task']==d for r in xs),hard_task=sum(r['hard_task'] for r in xs if r['task']==d)) for d in ['near_stop','far_progress']},
   known_unknown_count=sum(r['hard_unknown'] for r in xs),
   timing_ms={k:dict(median=1000*float(np.median([r[k] for r in xs])),max=1000*max(r[k] for r in xs)) for k in ['network_or_reference_s','minco_decode_s','independent_hard_and_reference_s']},
   peak_dynamics=[max(r[k] for r in xs) for k in ['speed_peak','acceleration_peak','jerk_peak']],
   feasible_quality_mean={k:float(np.mean([r[k] for r in xs if r['hard_task']])) for k in ['path_length_quadrature_m','jerk_squared_integral_quadrature','surface_clearance_sampled_m','surface_clearance_continuous_lower_bound_m','loss']})
  summary[method]=stats
 write(O/'FOREST_DESCRIPTIVE_METRICS.json',dict(summary=summary,records=rows,network_updates=0,
  clearance_note='All exact known-geometry SDF samples include floor and ROI; continuous lower bound uses 1-Lipschitz SDF and exact original speed maximum times max sample gap/2 plus1e-6 reserve. Negative lower bound alone does not prove collision. Existing broadphase certificate clearance field is not used as global minimum.',
  quality_note='Path and squared derivative integrals are numerical quadrature, descriptive only; no Hard or L0 changed.'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
