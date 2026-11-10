"""G3 thin experiment stages; all geometry, metrics and Hard math live in package components."""
from pathlib import Path
import sys,argparse,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
from query_minco.evaluation.mirror_holdout_v1 import qualification,load_scene,write,sha
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.evaluation.validation import evaluate,norm_max
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
def model_at(step):
    run=R/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104'
    p=run/('step%03d.pt'%step);s=json.loads(Path(str(p)+'.json').read_text())
    if sha(p)!=s['file_sha256']:raise ValueError('CHECKPOINT_SHA')
    b=torch.load(p,map_location='cpu',weights_only=True)
    if b['cursor']!=step:raise ValueError('CURSOR')
    m=make_planner().eval();m.load_state_dict(b['model'],strict=True)
    if state_sha(m)!=s['model_sha256']:raise ValueError('MODEL_SHA')
    return m,dict(path=str(p),file_sha256=sha(p),model_sha256=state_sha(m))
def baseline(state,kind):
    goal=state[0,6:9].double();end=torch.tensor([2.4,0.,0.],dtype=torch.float64) if kind=='fixed_forward' else goal/goal.norm()*min(2.4,float(goal.norm()))
    u=torch.tensor([1/3,2/3,1.],dtype=torch.float64);pts=(10*u**3-15*u**4+6*u**5)[:,None]*end
    low=torch.tensor([0.,-4.,-2.],dtype=torch.float64);high=torch.tensor([10.,4.,2.],dtype=torch.float64)
    raw=torch.zeros(1,1,21);raw[0,0,:9]=torch.atanh((2*(pts-low)/(high-low)-1).clamp(-1+1e-7,1-1e-7)).reshape(-1).float()
    return raw
def evaluate_stage():
    q=json.loads((O/'scenes/QUALIFICATION_RESULT.json').read_text())
    if q['status']!='QUALIFIED' or sum(x['qualified'] for x in q['pairs'])<2:
        write(O/'FROZEN_RESULTS_NOT_RUN.json',dict(status='NOT_RUN',reason='TWO_PAIR_QUALIFICATION_REQUIRED',network_updates=0));return
    m0,id0=model_at(0);m128,id128=model_at(128);before=[state_sha(m0),state_sha(m128)]
    dec=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);adapter=MincoAdapter(num_candidates=1)
    rows=[];arrays={}
    with torch.no_grad():
        for scene in q['scenes']:
            depth,state,g,manifest,cert=load_scene(scene['manifest'],scene['manifest_sha256'])
            fixed=baseline(state,'fixed_forward');fp=dec(fixed,state);ft=adapter(fp.head_pva,fp.tail_pva,fp.inner_pts,fp.durations)
            fc=ft.coeffs[0,0].double().numpy()
            for method in ['A_reasonable_step0','A_reasonable_step128','fixed_forward','state_goal']:
                start=time.perf_counter()
                if method=='A_reasonable_step0':raw=m0(depth,state)
                elif method=='A_reasonable_step128':raw=m128(depth,state)
                else:raw=baseline(state,method)
                infer=time.perf_counter()-start
                p=dec(raw,state);tr=adapter(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
                if not tr.solve_valid.all():raise ValueError('MINCO_SOLVE')
                if not torch.equal(p.head_pva[0,0,1:],state[0,:6].reshape(2,3)):raise ValueError('INITIAL_PVA')
                result=SimpleNamespace(parameters=p,trajectory=tr);batch=SimpleNamespace(geometry=(g,),state=state)
                hard=validate_result_candidates(result,batch)[0]
                c=tr.coeffs[0,0].double().numpy();dt=tr.durations[0,0].numpy()
                crossing=crossing_metric(c,dt,cert['planes_B0'],cert['gap_B0'],scene['side'])
                distance=max(norm_max(a-b,float(t),0) for a,b,t in zip(c,fc,dt))
                key=scene['scene_id']+'_'+method;arrays[key+'_raw21']=raw.numpy();arrays[key+'_coeff']=c
                sample=np.concatenate([evaluate(a,np.linspace(0,float(t),65)) for a,t in zip(c,dt)]);arrays[key+'_xyz']=sample
                unknown=any(z['reason'] in ['continuous_clearance_unproven','UNKNOWN_OR_ROI_MARGIN'] for z in hard.get('certificates',[]))
                if unknown:classification='CERTIFICATION_UNKNOWN'
                elif not crossing['crossed_all_planes'] or not crossing['terminal_beyond_exit']:classification='NOT_CROSSED'
                elif hard['passed'] and crossing['passed']:classification='CORRECT_RESPONSE'
                else:classification='WRONG_RESPONSE_OR_COLLISION'
                rows.append(dict(scene_id=scene['scene_id'],pair_id=manifest['pair']['pair_id'],side=scene['side'],method=method,
                    hard=hard,hard_and_task=hard['passed'] and hard['goal']['passed'],crossing=crossing,
                    hard_task_and_required_crossing=hard['passed'] and crossing['passed'],classification=classification,
                    single_candidate_forward_s=infer,continuous_difference_from_fixed_m=distance,
                    cpu_threads=torch.get_num_threads(),timing_note='single cold/unwarmed forward; excludes disk loading, decoder and Hard'))
    assert before==[state_sha(m0),state_sha(m128)]
    response={}
    for method in ['A_reasonable_step0','A_reasonable_step128','fixed_forward','state_goal']:
        xs=[x for x in rows if x['method']==method]
        response[method]=dict(hard_and_task=sum(x['hard_and_task'] for x in xs),crossed=sum(x['crossing']['crossed_all_planes'] for x in xs),
            hard_task_and_crossing=sum(x['hard_task_and_required_crossing'] for x in xs),n=len(xs),correct_mirror_pairs=sum(all(x['hard_task_and_required_crossing'] for x in xs if x['pair_id']==pair['pair_id']) for pair in q['pairs']))
    np.savez(O/'FROZEN_MIRROR_ARRAYS.npz',**arrays)
    write(O/'FROZEN_MIRROR_RESULTS.json',dict(status='EXECUTED',checkpoint_identity=[id0,id128],records=rows,summary=response,
        weights_unchanged=True,optimizer_constructed=False,new_network_updates=0,no_model_based_scene_tuning=True))
    print(json.dumps(response,indent=2))
def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['qualify','evaluate'],required=True);args=a.parse_args()
    torch.set_num_threads(1)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER_ALLOWED')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_NETWORK_UPDATES')):
        if args.stage=='qualify':
            from query_minco.models import QueryMincoPlanner
            binary=R/'research_outputs/physics_only_v1_2_e1_r2a_supervisor_v2/runs/build_001/artifacts/e1_renderer_bridge'
            if sha(binary)!='7cb7ead45a858da5f56fc0d9ebc3b9c35a6274ede690f54c58564fcae5b0069b':raise ValueError('NATIVE_BINARY_SHA')
            registry=R/'research_outputs/physics_only_v1_2_e1_dataset_v2/runs/qualification_001/ALL_SCENE_IDENTITIES.json'
            with patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('NO_MODEL_BEFORE_QUALIFICATION')):
                q=qualification(O/'PREREGISTRATION.json',O/'scenes',binary,registry)
            print(q['status']);print([(x['scene_id'],x['status'],x.get('reason')) for x in q['scenes']])
        else:evaluate_stage()
if __name__=='__main__':main()
