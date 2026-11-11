"""Thin forest preparation/evaluation entry. Never imports YOPO policy code."""
from pathlib import Path
import sys,json,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,numpy as np
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
from query_minco.evaluation.forest_adapter_v1 import prepare,ForestAdapter,verify_files
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from query_minco.audit.k1_signal_diagnostics_v1 import model_at
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.trajectory.state_goal_reference_v1 import reference_raw
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha,metric_terms
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
def evaluation():
    index=json.loads((O/'dataset/FOREST_INDEX.json').read_text())
    models={};before={};identities=[]
    for step in [0,128]:
        model,identity=model_at(R,step);models[step]=model;before[step]=state_sha(model);identities.append(identity)
    records=[];arrays={};dec=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);adapter=MincoAdapter(num_candidates=1)
    with torch.no_grad():
        for row in index:
            batch=ForestAdapter.load_snapshot(row['manifest'],row['manifest_sha256'])
            for method in ['fixed_forward','state_goal_quintic','K1_step0','K1_step128']:
                begin=time.perf_counter()
                raw=reference_raw(batch.state,method) if method in ['fixed_forward','state_goal_quintic'] else models[0 if method=='K1_step0' else 128](batch.depth,batch.state)
                forward=time.perf_counter()-begin
                if raw.shape!=(1,1,21) or not torch.isfinite(raw).all():raise ValueError('K1_RAW')
                begin=time.perf_counter();p=dec(raw,batch.state);tr=adapter(p.head_pva,p.tail_pva,p.inner_pts,p.durations);decode=time.perf_counter()-begin
                begin=time.perf_counter();result=make_objective()(raw,batch);cost_time=time.perf_counter()-begin
                if not torch.equal(tr.coeffs,result.trajectory.coeffs) or not tr.solve_valid.all() or not torch.isfinite(result.loss):raise ValueError('TRAJECTORY_OR_COST')
                if not torch.equal(p.head_pva[0,0,1:],batch.state[0,:6].reshape(2,3)):raise ValueError('HEAD_PVA')
                begin=time.perf_counter();hard=validate_result_candidates(result,batch)[0];hard_time=time.perf_counter()-begin
                if 'INDEPENDENT_MINCO' in hard['reasons']:raise ValueError('MINCO_REFERENCE')
                q=result.training_geometry_records[0];quality=metric_terms(result) if hard['passed'] else None
                certificates=hard.get('certificates',[])
                unknown=sum(x['reason'] in ['UNKNOWN_OR_ROI_MARGIN','continuous_clearance_unproven'] for x in certificates)
                row_result=dict(snapshot_id=row['snapshot_id'],map_id=row['map_id'],seed=row['seed'],density=row['density'],task=row['task'],preclassification=row['classification'],
                    method=method,raw21=raw.tolist(),loss=float(result.loss),terms=metric_terms(result),hard=hard,
                    hard_task=hard['passed'] and hard['goal']['passed'],quality_if_feasible=quality,
                    unknown_center_sample_fraction=float(q['outside_mask'].float().mean()),hard_unknown_piece_count=unknown,
                    timings=dict(network_or_reference_s=forward,minco_decode_s=decode,objective_with_repeated_minco_s=cost_time,independent_hard_and_reference_s=hard_time,
                                 measured_forward_minco_hard_sum_s=forward+decode+hard_time))
                records.append(row_result);key=row['snapshot_id']+'__'+method
                arrays[key+'_raw21']=raw.numpy();arrays[key+'_coeff']=tr.coeffs.numpy();arrays[key+'_durations']=tr.durations.numpy()
            print('FOREST_EVALUATED',row['snapshot_id'],row['classification'],flush=True)
    if any(state_sha(models[s])!=before[s] or any(p.grad is not None for p in models[s].parameters()) for s in models):raise ValueError('MODEL_MUTATION')
    summary={}
    for method in ['fixed_forward','state_goal_quintic','K1_step0','K1_step128']:
        summary[method]={}
        for group in ['CERTIFIED_FEASIBLE','FEASIBILITY_UNDETERMINED','CERTIFIED_INFEASIBLE']:
            xs=[x for x in records if x['method']==method and x['preclassification']==group]
            summary[method][group]=dict(n=len(xs),hard_passed=sum(x['hard']['passed'] for x in xs),hard_task=sum(x['hard_task'] for x in xs),
                nominal_unknown_count=sum(x['hard_unknown_piece_count']>0 for x in xs),mean_loss=sum(x['loss'] for x in xs)/len(xs) if xs else None)
    write(O/'FOREST_RESULTS.json',dict(records=records,summary=summary,checkpoint_identity=identities,network_updates=0,leaf_updates=0,known_feasible_denominator_fixed_before_eval=True,
        yopo_baseline='NOT_RUN_NOT_COMPARABLE',time_note='Isolated MINCO and objective both decode; objective timing is not additive with isolated decode for end-to-end planning.'))
    np.savez_compressed(O/'FOREST_TRAJECTORIES.npz',**arrays)
    print(json.dumps(summary,indent=2))
def prior_k2_evidence():
    b=R/'research_outputs/g3_b_signal_and_data_20261110_v1'
    source=b/'gradient/LATERAL_COST_HARD_SCAN.json';manifest=json.loads((b/'FILES_AND_SHA.json').read_text())
    if sha(source)!=manifest['artifacts']['gradient/LATERAL_COST_HARD_SCAN.json']['sha256']:raise ValueError('HISTORICAL_REFERENCE_SHA')
    rows=json.loads(source.read_text())['records'];records=[]
    for scene in sorted({x['scene_id'] for x in rows}):
        pair=[x for x in rows if x['scene_id']==scene and x['start']=='fixed_forward' and x['physical_inner_y_shift_m'] in [-.5,.5]]
        records.append(dict(scene_id=scene,physical_offsets_m=[x['physical_inner_y_shift_m'] for x in pair],hard=[x['hard']['passed'] for x in pair],oracle_at2=any(x['hard']['passed'] for x in pair),
                            both_hard_in_same_world=all(x['hard']['passed'] for x in pair)))
    write(O/'K2_PRIOR_REFERENCE_COVERAGE.json',dict(status='PRIOR_FROZEN_EVIDENCE_SHA_VERIFIED_NOT_NEW_K2_IMPLEMENTATION',source=str(source),sha256=sha(source),records=records,network_updates=0))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate','k2-prior'],required=True);args=p.parse_args()
    torch.set_num_threads(1)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_UPDATE')):
        if args.stage=='prepare':
            with patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('QUALIFY_BEFORE_MODEL')):
                result=prepare(R,O/'PREREGISTRATION.json',O/'dataset')
            print(json.dumps(result,indent=2))
        elif args.stage=='evaluate':evaluation()
        else:prior_k2_evidence()
if __name__=='__main__':main()
