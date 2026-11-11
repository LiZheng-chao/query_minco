"""Frozen final-only paired inference on exposed G3D/G3A sets;zero updates."""
from pathlib import Path
import sys,json,time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from query_minco.evaluation.forest_adapter_v1 import ForestAdapter
from query_minco.evaluation.mirror_holdout_v1 import load_scene,sha,write
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective,make_planner
from query_minco.training.k1_physics_trainer_v1.trainer import result_numeric,state_sha,metric_terms
from query_minco.audit.k1_signal_diagnostics_v1 import model_at
from query_minco.trajectory.state_goal_reference_v1 import reference_raw
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.evaluation.validation import norm_max,evaluate
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
def main():
 torch.set_num_threads(1);old,identity=model_at(R,128)
 path=R/'research_outputs/g4a_forest_shadow_20261111_v1/formal/A_reasonable_seed20261104/step128.pt'
 side=json.loads(Path(str(path)+'.json').read_text());assert sha(path)==side['file_sha256']
 new=make_planner().eval();new.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True);assert state_sha(new)==side['model_sha256']
 models={'old_K1_step128':old,'forest_K1_final128':new};before={k:state_sha(v) for k,v in models.items()}
 forest=json.loads((R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/FOREST_INDEX.json').read_text())
 mirror=json.loads((R/'research_outputs/g3_a_mirror_holdout_20261110_v1/scenes/QUALIFICATION_RESULT.json').read_text())['scenes']
 rows=[dict(group='G3D_exposed_test',id=x['snapshot_id'],source=x) for x in forest]+[dict(group='G3A_exposed_mechanism',id=x['scene_id'],source=x) for x in mirror]
 records=[];arrays={}
 with torch.no_grad(),patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
  for row in rows:
   x=row['source'];q=None
   if row['group']=='G3D_exposed_test':
    batch=ForestAdapter.load_snapshot(x['manifest'],x['manifest_sha256']);classification=x['classification']
   else:
    depth,state,g,m,q=load_scene(x['manifest'],x['manifest_sha256'])
    batch=BoundBatch(depth,state,(g,),m,g.validate);batch._depth=depth.numpy().tobytes();batch._state=state.numpy().tobytes();batch.validate()
    classification='CERTIFIED_FEASIBLE'
   for method in ['fixed_forward','state_goal_quintic','old_K1_step128','forest_K1_final128']:
    start=time.perf_counter();raw=models[method](batch.depth,batch.state) if method in models else reference_raw(batch.state,method);forward=time.perf_counter()-start
    start=time.perf_counter();out=make_objective()(raw,batch);objective_s=time.perf_counter()-start
    if not torch.isfinite(out.loss) or raw.shape!=(1,1,21):raise ValueError('NONFINITE_OR_SHAPE')
    start=time.perf_counter();numeric,hard=result_numeric(out,batch);hard_s=time.perf_counter()-start
    c=out.trajectory.coeffs[0,0].numpy();dt=out.trajectory.durations[0,0].numpy();cross=None
    if q is not None:cross=crossing_metric(c,dt,q['planes_B0'],q['gap_B0'],q['side'])
    key=row['id']+'__'+method;arrays[key+'_raw21']=raw.numpy();arrays[key+'_coeff']=c;arrays[key+'_durations']=dt
    records.append(dict(group=row['group'],sample_id=row['id'],classification=classification,method=method,loss=float(out.loss),terms=metric_terms(out),
     hard=hard,crossing=cross,hard_task_safe_crossing=bool(hard['passed'] and cross['passed']) if cross is not None else None,
     forward_s=forward,objective_with_minco_s=objective_s,hard_reference_s=hard_s,independent_minco=numeric,
     raw21=raw.tolist(),endpoint=evaluate(c[-1],dt[-1]).tolist(),quality_if_hard={k:float(out.metrics[k].mean()) for k in ['S','P','C_shape']} if hard['passed'] else None))
   print('PAIRED_EVALUATED',row['group'],row['id'],flush=True)
 assert all(state_sha(v)==before[k] and all(p.grad is None for p in v.parameters()) for k,v in models.items())
 summary={}
 for group in ['G3D_exposed_test','G3A_exposed_mechanism']:
  summary[group]={}
  for method in ['fixed_forward','state_goal_quintic','old_K1_step128','forest_K1_final128']:
   summary[group][method]={}
   for cls in ['CERTIFIED_FEASIBLE','FEASIBILITY_UNDETERMINED']:
    rr=[v for v in records if v['group']==group and v['method']==method and v['classification']==cls]
    summary[group][method][cls]=dict(n=len(rr),hard_task=sum(x['hard']['passed'] for x in rr),safe_crossing=sum(bool(x['hard_task_safe_crossing']) for x in rr),
     mean_loss=float(np.mean([x['loss'] for x in rr])) if rr else None)
 differences=[]
 for row in rows:
  prefix=row['id']+'__';a=arrays[prefix+'old_K1_step128_coeff'];b=arrays[prefix+'forest_K1_final128_coeff'];dt=arrays[prefix+'old_K1_step128_durations']
  differences.append(dict(sample_id=row['id'],continuous_position_difference_max_m=max(norm_max(x-y,t,0) for x,y,t in zip(a,b,dt))))
 write(O/'PAIRED_OFFLINE_RESULTS.json',dict(records=records,summary=summary,differences=differences,model_identity=dict(old=identity,new=dict(path=str(path),file_sha256=sha(path),model_sha256=state_sha(new))),
  model_before=before,model_after={k:state_sha(v) for k,v in models.items()},new_network_updates=0,leaf_updates=0,
  exposed_test_history=True,checkpoint_selection='preregistered final128 only,no test-based selection'))
 np.savez_compressed(O/'PAIRED_TRAJECTORIES.npz',**arrays)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
