"""Read-only original hard evaluator plus detailed continuous numeric diagnostics."""
import copy
import numpy as np
import torch
from query_minco.training.pg1.execution import evaluate_development as original
from query_minco.training.pilot_evaluation import load_rows,_certified_empty
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
from query_minco.trajectory import TrajectoryPostprocessor
from .objective import tail_from_samples
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss
from experiments.pg1_v2_r1.cpu0_readonly import metrics,numpy_integral
def candidate_metrics(raw,sample,scene,q):
 if not _certified_empty(scene):return metrics(raw,sample,scene,q)
 from query_minco.evaluation.validation import validate_teacher,evaluate as polynomial_evaluate
 from query_minco.evaluation.goal import task_metrics
 out=TrajectoryPostprocessor()(raw,sample.state[None]);par=out.parameters
 c=out.trajectory.coeffs[0,q].numpy();d=par.durations[0,q].numpy()
 hard=validate_teacher(c,d,par.head_pva[0,q].numpy(),par.tail_pva[0,q].numpy(),par.inner_pts[0,q].numpy(),scene)
 goal=task_metrics(polynomial_evaluate(c[-1],d[-1]),polynomial_evaluate(c[-1],d[-1],1),sample.state.numpy()[6:9])
 quality=dynamic_quality_loss(out.trajectory)['terms']
 vals=dict(obstacle=raw[...,0]*0,acceleration=quality['acceleration'],jerk=quality['jerk'],time=quality['time'],lse=raw[...,0]*0,tail=raw[...,0]*0)
 return dict(clearance_m=None,closest_time_s=None,closest_box=None,hard_pass=hard['passed'],hard_reasons=hard['reasons'],
  vmax=hard['max_speed_mps'],amax=hard['max_acceleration_mps2'],jmax=hard['max_jerk_mps3'],time_s=hard['total_time_s'],goal=goal,
  crossing_position=None,heuristic_mode='direct',per_query_terms={k:float(v[0,q]) for k,v in vals.items()},scene_terms={k:float(v.mean()) for k,v in vals.items()})

def evaluate(model,binding):
 # Original path consumes exactly its original RNG; deepcopy consumes no random draws.
 result=original(model,binding)
 replica=copy.deepcopy(model).cpu().eval();details=[]
 with torch.no_grad():
  for sample,meta in load_rows(binding):
   raw=replica(depth_input=sample.depth_input[None],state=sample.state[None])
   out=TrajectoryPostprocessor()(raw,sample.state[None])
   cost,_=cost_matrix(out,sample,0);il,qq,tt=match(cost,'B_set')
   boxes=[raw.new_tensor([[b['min'],b['max']] for b in meta['scene']['obstacles']]).reshape(-1,2,3)]
   p,v,w=segment_samples(out.trajectory,64)
   tail=tail_from_samples(p,w,boxes,[_certified_empty(meta['scene'])])
   obs,_=physical_loss(p,v,w,boxes,geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
   c=out.trajectory.coeffs.numpy();d=out.parameters.durations.numpy()
   energies={k:numpy_integral(c,d,o)[0].tolist() for k,o in [('acceleration_energy_m2_s3',2),('jerk_energy_m2_s5',3)]}
   for q in range(6):
    m=candidate_metrics(raw,sample,meta['scene'],q)
    details.append(dict(sample_id=sample.sample_id,query=q,matched=q in qq.tolist(),
      teacher_index=next((t for a,t in zip(qq.tolist(),tt.tolist()) if a==q),None),
      **m,acceleration_energy_m2_s3=energies['acceleration_energy_m2_s3'][q],jerk_energy_m2_s5=energies['jerk_energy_m2_s5'][q],
      tail=float(tail[0,q]),obstacle=float(obs[0,q]),imitation_costs=cost[q].tolist()))
 result['r12_candidate_details']=details
 result['continuous_clearance']='numeric polynomial/AABB root minimum; original Bernstein remains hard authority'
 result['evaluation_set']='open-dev8 MECHANISM DIAGNOSTIC, not independent final test'
 return result
