import json,hashlib,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import defaultdict,Counter
import numpy as np
import torch
from unittest.mock import patch
from query_minco.training.k1_physics_trainer_v1 import trainer as T
from query_minco.training.k1_physics_trainer_v1.optimizer import DeadRowGuard,digest
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner,make_objective
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial,validate_result
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import evaluate as peval
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/g2b_k1_visual_audit_20261110_v1'
G2=ROOT/'research_outputs/g2a_k1_trainer_qualification_20261110_v1'
RUN=ROOT/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104'
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def check(v,msg):
 if not v:raise AssertionError(msg)
def modelblob(blob):
 m=make_planner().eval();m.load_state_dict(blob['model'],strict=True);return m
def curve(c,d):
 return np.concatenate([peval(a,np.linspace(0,float(t),65)) for a,t in zip(c,d)])
def describe(raw,state,g):
 params=DEC(raw,state);tr=AD(params.head_pva,params.tail_pva,params.inner_pts,params.durations)
 check(bool(tr.solve_valid.all()),'MINCO_SOLVE')
 c=tr.coeffs[0,0].double().numpy();d=tr.durations[0,0].numpy()
 check(np.isfinite(c).all() and np.isfinite(d).all() and (d>0).all(),'INVALID_TRAJECTORY')
 return dict(raw15=raw[0,0,:15].tolist(),coefficients=c.tolist(),durations=d.tolist(),hard=validate_polynomial(c,d,g,state[0,6:9].numpy()),xyz=curve(c,d).tolist(),knots=params.inner_pts[0,0].tolist(),terminal_pva=params.tail_pva[0,0].tolist())
DEC=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1)
AD=MincoAdapter(num_candidates=1)
def baseline_raw(state,mode):
 goal=state[0,6:9].double()
 end=torch.tensor([2.4,0.,0.],dtype=torch.float64) if mode=='fixed_forward' else goal/goal.norm().clamp_min(1e-8)*min(float(goal.norm()),2.4)
 u=torch.tensor([1/3,2/3,1.],dtype=torch.float64)
 points=(10*u**3-15*u**4+6*u**5)[:,None]*end
 low=torch.tensor([0.,-4.,-2.],dtype=torch.float64);high=torch.tensor([10.,4.,2.],dtype=torch.float64)
 physical_clipped=bool(((points<=low)|(points>=high)).any())
 unit=(2*(points-low)/(high-low)-1).clamp(-1+1e-7,1-1e-7)
 raw=torch.zeros(1,1,21);raw[0,0,:9]=torch.atanh(unit).reshape(-1).float()
 return raw,physical_clipped

def provenance(spec,rows):
 logs=[json.loads(x) for x in (RUN/'STEPS.jsonl').read_text().splitlines()]
 check(len(logs)==128 and [x['step'] for x in logs]==list(range(1,129)),'STEPS_SEQUENCE')
 check(read(RUN/'FORMAL_STEP_LEDGER.json')==dict(attempted=128,completed=128),'LEDGER')
 for i,x in enumerate(logs):
  check(x['classification']=='FORMAL_RESEARCH_TRAINING' and x['split']=='train','TRAIN_ONLY')
  check(x['sample_id']==spec['train_order'][i%13],'ORDER')
  check(x['raw15_21_nonzero']==0,'DEAD_RAW_GRADIENT')
  check(all(v['parameter_update_norm']>0 for v in x['module_updates'].values()),'MODULE_UPDATES')
  if i:check(x['model_sha_before']==logs[i-1]['model_sha_after'],'CHAIN')
 auth=read(RUN/'AUTHORIZATION.json')
 check(auth['runspec_sha256']==T.sha(G2/'K1_TRAINING_RUNSPEC_DRAFT.json'),'SPEC_AUTH')
 matches=[p for p in G2.glob('*.json') if T.sha(p)==auth['approval_sha256']]
 check(len(matches)==1,'APPROVAL_ORIGINAL_NOT_FOUND')
 approval=read(matches[0]);T.authorize(G2/'K1_TRAINING_RUNSPEC_DRAFT.json',matches[0],20261104,'A_reasonable',RUN)
 checkpoints={};dead=[];replays=[];arrays={}
 for step in [0,8,32,64,128]:
  path=RUN/('step%03d.pt'%step);side=read(str(path)+'.json')
  check(T.sha(path)==side['file_sha256'],'CHECKPOINT_BYTES')
  blob=torch.load(path,map_location='cpu',weights_only=True);m=modelblob(blob)
  check(T.state_sha(m)==side['model_sha256'],'MODEL_SHA')
  check(blob['cursor']==step and blob['metadata']==auth,'CURSOR_METADATA')
  if step:check(side['model_sha256']==logs[step-1]['model_sha_after'],'STEP_CHECKPOINT')
  else:check(side['model_sha256']==logs[0]['model_sha_before'],'STEP_ZERO')
  guard=DeadRowGuard(m);dead.append(guard.sha())
  check(all(torch.equal(v,p[15:]) for v,p in zip(blob['dead_reference'],guard.parameters)),'DEAD_REFERENCE')
  if step:
   steps=[float(s['step']) for s in blob['optimizer']['state'].values()]
   check(steps and all(v==step for v in steps),'ADAM_ACTUAL_STEPS')
   keys=blob['optimizer']['param_groups'][0]['params']
   named=list(m.parameters())
   for p in guard.parameters:
    i=next(i for i,q in enumerate(named) if q is p);st=blob['optimizer']['state'][keys[i]]
    check(all(not v[15:].count_nonzero() for k,v in st.items() if k in ['exp_avg','exp_avg_sq','max_exp_avg_sq']),'DEAD_MOMENT')
  stored=read(RUN/('evaluation%03d.json'%step));check(stored['dev_backward_calls']==0,'DEV_LOG')
  check(stored['model_sha_before']==stored['model_sha_after']==side['model_sha256'],'EVAL_MODEL_SHA')
  replay=T.evaluate(m,make_objective(),rows)
  check(replay==stored,'EVALUATION_REPLAY_%s'%step)
  checkpoints[step]=blob
  side_info=dict(step=step,path=str(path),file_sha256=side['file_sha256'],model_sha256=side['model_sha256'],dead_sha256=guard.sha(),optimizer_step=step,train=stored['train'],dev=stored['dev'])
  # additional aggregate term, lateral bias and failure information
  for split in ['train','dev']:
   xs=[x for x in stored['records'] if x['split']==split]
   side_info[split]['mean_terms']={k:float(np.mean([x['terms'][k] for x in xs])) for k in T.TERM_NAMES}
  displacements=[]
  for row in rows:
   b=T.load(row,row['split']);raw=m(b.depth,b.state);res=describe(raw,b.state,b.geometry[0]);a=np.asarray(res['xyz'])
   displacements.append(dict(sample_id=row['sample_id'],y_min=float(a[:,1].min()),y_max=float(a[:,1].max()),endpoint=res['terminal_pva'][0],maxima=res['hard']['dynamic_polynomial_maxima'],passed=res['hard']['passed'],reasons=res['hard']['reasons']))
   arrays['step%s_%s_raw'%(step,row['sample_id'])]=raw.numpy()
   arrays['step%s_%s_coeff'%(step,row['sample_id'])]=np.asarray(res['coefficients'])
  side_info['lateral_and_dynamics']=displacements;replays.append(side_info)
 check(len(set(dead))==1,'DEAD_SHA_DRIFT')
 source=[dict(path=k,expected=v,actual=T.sha(k),matches=T.sha(k)==v) for k,v in spec['source_sha256'].items()]
 check(all(x['matches'] for x in source),'CURRENT_SOURCE_DRIFT')
 process_files=[str(x) for x in RUN.rglob('*') if x.is_file() and any(z in x.name.lower() for z in ['stdout','stderr','process','argv','command'])]
 identity=dict(artifact_consistency='PASS',independent_evaluation_replay='PASS',checkpoints=replays,source_binding=source,
  approval_original=dict(path=str(matches[0]),sha256=T.sha(matches[0]),content=approval),
  authorization=auth,step_ledger=read(RUN/'FORMAL_STEP_LEDGER.json'),step_count=128,dead_sha256=dead[0],
  full_process_logs=process_files,actual_launch_argv='NOT_RECORDED_BY_TRAINER',
  historical_process_execution_provenance='UNVERIFIED' if not process_files else 'REQUIRES_MANUAL_PROCESS_LOG_AUDIT',
  limits='Recomputed artifacts and source do not independently prove historical runtime argv, process exit, or absence of unlogged activity.',
  new_network_updates=0)
 write('FORMAL_128STEP_EVIDENCE.json',identity);np.savez(OUT/'REPLAY_ARRAYS.npz',**arrays)
 return checkpoints

def visual(spec,rows,blobs):
 batches=[T.load(x,x['split']) for x in rows]
 groups=defaultdict(list)
 for i,b in enumerate(batches):groups[b.state.numpy().tobytes()].append(i)
 pairs=[]
 for indices in groups.values():
  for a in indices:
   for b in indices:
    if a>=b:continue
    if batches[a].geometry[0].identity['source_scene_id']==batches[b].geometry[0].identity['source_scene_id']:continue
    pairs.append((a,b))
 check(pairs,'NO_VALID_SAME_STATE_PAIRS')
 outputs={};results=[];arrays={}
 for step in [0,32,64,128]:
  m=modelblob(blobs[step]);before=T.state_sha(m)
  cache=[]
  for b in batches:
   raw=m(b.depth,b.state)
   cache.append((raw,describe(raw,b.state,b.geometry[0])))
  for a,b in pairs:
   ra,da=cache[a];rb,db=cache[b]
   ca=np.asarray(da['coefficients']);cb=np.asarray(db['coefficients'])
   xa=np.asarray(da['xyz']);xb=np.asarray(db['xyz'])
   key='step%s_pair%s_%s'%(step,a,b)
   arrays[key+'_raw_a']=ra.numpy();arrays[key+'_raw_b']=rb.numpy()
   arrays[key+'_coeff_a']=ca;arrays[key+'_coeff_b']=cb;arrays[key+'_xyz_a']=xa;arrays[key+'_xyz_b']=xb
   ga=batches[a].geometry[0];gb=batches[b].geometry[0]
   ha=da['hard'];hb=db['hard']
   hab=validate_polynomial(cb,np.asarray(db['durations']),ga,batches[a].state[0,6:9].numpy())
   hba=validate_polynomial(ca,np.asarray(da['durations']),gb,batches[b].state[0,6:9].numpy())
   check(torch.equal(batches[a].state,batches[b].state),'STATE_NOT_IDENTICAL')
   results.append(dict(step=step,a=rows[a]['sample_id'],b=rows[b]['sample_id'],
    state_sha256=hashlib.sha256(batches[a].state.numpy().tobytes()).hexdigest(),
    depth_mask_max_abs=float((batches[a].depth-batches[b].depth).abs().max()),
    raw15_max_abs=float((ra[...,:15]-rb[...,:15]).abs().max()),
    coefficient_max_abs=float(np.abs(ca-cb).max()),curve_max_separation_m=float(np.linalg.norm(xa-xb,axis=-1).max()),
    same_world_original_hard=[ha['passed'],hb['passed']],swapped_sensor_hard=[hab['passed'],hba['passed']],
    swapped_sensor_reasons=[hab['reasons'],hba['reasons']]))
  check(T.state_sha(m)==before,'VISUAL_WEIGHTS_CHANGED')
  xs=[x for x in results if x['step']==step]
  outputs[step]=dict(pairs=len(xs),raw15_max_abs=max(x['raw15_max_abs'] for x in xs),
   curve_max_separation_m=max(x['curve_max_separation_m'] for x in xs),
   hard_outcomes_changed=sum(x['same_world_original_hard']!=x['swapped_sensor_hard'] for x in xs))
 write('VISUAL_PAIRED_RESULTS.json',dict(status='EXECUTED',summary=outputs,records=results,
  interpretation='Different valid scene observations at exactly same state test causal input sensitivity. Swaps evaluated in target worlds are interventions, not physically consistent paired sensors. Independent mirror navigation causality remains unqualified.',new_network_updates=0))
 np.savez(OUT/'VISUAL_PAIR_ARRAYS.npz',**arrays)
 return batches

def baselines(rows,batches,blobs):
 records=[]
 for name in ['fixed_forward','state_goal','A_random','A_reasonable_step0','trained_step128']:
  m=T.model_for(SPEC,'A_random',20261104).eval() if name=='A_random' else modelblob(blobs[0 if name=='A_reasonable_step0' else 128]) if name.startswith(('A_reasonable','trained')) else None
  for row,b in zip(rows,batches):
   if m is None:raw,clipped=baseline_raw(b.state,name)
   else:raw=m(b.depth,b.state);clipped=False
   res=describe(raw,b.state,b.geometry[0]);out=make_objective()(raw,b)
   records.append(dict(method=name,sample_id=row['sample_id'],split=row['split'],loss=float(out.loss),terms=T.metric_terms(out),position_bound_clipped=clipped,**res))
 summary={}
 for name in sorted(set(x['method'] for x in records)):
  summary[name]={}
  for split in ['train','dev']:
   xs=[x for x in records if x['method']==name and x['split']==split]
   summary[name][split]=dict(n=len(xs),hard_and_task=sum(x['hard']['passed'] and x['hard']['goal']['passed'] for x in xs),mean_loss=float(np.mean([x['loss'] for x in xs])))
 write('FIXED_REFERENCE_RESULTS.json',dict(status='EVALUATED',rules=read(OUT/'PREREGISTRATION.json')['baseline_rules'],summary=summary,records=records,new_network_updates=0))

def ood(rows,batches,blobs):
 # Qualified free-ROI scene held fixed; these are state interventions, not new independently-qualified task manifests.
 i=next(i for i,x in enumerate(rows) if x['sample_id']=='train_free_roi_1101_p0_near')
 b=batches[i];m=modelblob(blobs[128]);conditions=read(OUT/'PREREGISTRATION.json')['ood'];records=[]
 for case in conditions:
  state=b.state.clone();state[0,:6]=torch.tensor([case['v'],0,0,case['a'],0,0]);state[0,6:9]=torch.tensor(case['goal'])
  for name in ['fixed_forward','state_goal','trained_step128']:
   if name=='trained_step128':raw=m(b.depth,state);clipped=False
   else:raw,clipped=baseline_raw(state,name)
   res=describe(raw,state,b.geometry[0])
   records.append(dict(case=case,method=name,position_bound_clipped=clipped,**res))
 write('OOD_RESULTS.json',dict(status='EXECUTED_STATE_INTERVENTIONS',source_manifest=rows[i],records=records,
  limits='State-only interventions in original free-ROI geometry. Not independent-map qualification. Raw A decoder forward-only x bound prevents reverse endpoints; no architecture change.',new_network_updates=0))

def hard_api(rows,batches,blobs):
 b=batches[0];m=modelblob(blobs[128]);obj=make_objective();out=obj(m(b.depth,b.state),b)
 try:
  validate_result(out,b)
 except IndexError as exc:
  failure=dict(status='EXPECTED_K1_API_FAILURE_REPRODUCED',exception=repr(exc))
 else:raise AssertionError('K1_LEGACY_API_UNEXPECTED_PASS')
 from query_minco.models import QueryMincoPlanner
 from query_minco.models.config import QueryMincoConfig
 from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
 torch.manual_seed(20261104);k6=QueryMincoPlanner(QueryMincoConfig()).eval()
 k6obj=RoiSafeTrainingObjective(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=6)
 result=k6obj(k6(b.depth,b.state),b);legacy=validate_result(result,b)
 for i,x in enumerate(legacy):
  direct=validate_polynomial(result.trajectory.coeffs[0,i].double().numpy(),result.trajectory.durations[0,i].numpy(),b.geometry[0],b.state[0,6:9].numpy())
  check(all(x[k]==v for k,v in direct.items()),'K6_AUX_DRIFT')
 invalid=[]
 for d in [np.zeros(3),-np.ones(3),np.array([np.nan,1,1]),np.array([np.inf,1,1]),np.ones((1,3)),np.ones(2)]:
  x=validate_polynomial(out.trajectory.coeffs[0,0].double().numpy(),d,b.geometry[0],b.state[0,6:9].numpy())
  check(not x['passed'],'INVALID_DURATION_ACCEPTED');invalid.append(dict(reasons=x['reasons']))
 write('HARD_API_RESULTS.json',dict(k1_legacy_validate_result=failure,k1_production_validate_polynomial='PASS',k6_aux_direct_equivalence='PASS_6_CANDIDATES',invalid_durations=invalid,core_modified=False,overall_k1_hard_api_compatibility='FAIL'))

if __name__=='__main__':
 torch.set_num_threads(1)
 SPEC=T.verify_spec(G2/'K1_TRAINING_RUNSPEC_DRAFT.json')
 with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('AUDIT_NO_OPTIMIZER')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('AUDIT_NO_STEP')),torch.no_grad():
  rows=SPEC['train']+SPEC['dev']
  blobs=provenance(SPEC,rows);print('PROVENANCE_REPLAY_PASS 5 checkpoints 105 task evaluations',flush=True)
  batches=visual(SPEC,rows,blobs);print('VISUAL_PAIR_PASS',flush=True)
  baselines(rows,batches,blobs);print('BASELINES_EVALUATED',flush=True)
  ood(rows,batches,blobs);print('OOD_EVALUATED',flush=True)
  hard_api(rows,batches,blobs);print('K1_HARD_AUX_FAILURE_REPRODUCED_K6_PASS',flush=True)
 write('ZERO_UPDATE_EXIT.json',dict(status='PASS',new_network_updates=0,optimizer_instantiations=0,backward_calls=0))
 print('AUDIT_COMPLETE_NEW_NETWORK_UPDATES=0',flush=True)
