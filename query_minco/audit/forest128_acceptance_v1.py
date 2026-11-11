"""Read-only formal128 identity acceptance and exact zero-update evaluation replay."""
from pathlib import Path
import json,hashlib,csv
import numpy as np,torch
from query_minco.training.k1_forest_v1.trainer import verify,model_for,evaluate
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha,metric_terms
from query_minco.training.k1_physics_trainer_v1.optimizer import digest
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from unittest.mock import patch
COMPONENT_ID='formal128_artifact_process_and_math_replay_acceptance_v1'
def audit(root,output):
 r=Path(root);o=Path(output);owner=r/'research_outputs/g4a_forest_shadow_20261111_v1';run=owner/'formal/A_reasonable_seed20261104';process=owner/'formal_process/A_reasonable_seed20261104'
 specpath=owner/'K1_FOREST_TRAINING_RUNSPEC.json';spec=verify(specpath);specsha=sha(specpath)
 result=json.loads((run/'RESULT.json').read_text());assert result['status']=='COMPLETED_EXECUTION_NOT_SCIENTIFIC_SUCCESS' and result['formal_updates_this_process']==128
 p=json.loads((process/'PROCESS_RESULT.json').read_text());approval=json.loads((process/'APPROVAL.json').read_text())
 assert p['exit_code']==0 and not p['timed_out'] and p['stdout_sha256']==sha(process/'stdout.log') and p['stderr_sha256']==sha(process/'stderr.log')
 assert 'COMPLETED' in (process/'stdout.log').read_text()
 assert sha(process/'RUNSPEC.json')==specsha==p['inputs']['RUNSPEC']['sha256']
 approsha=sha(process/'APPROVAL.json')
 assert approsha==p['inputs']['APPROVAL']['sha256']==sha(owner/'approvals/A_reasonable_seed20261104.json')
 assert approval['runspec_sha256']==specsha and approval['decision']=='APPROVE_EXACT_RUNSPEC' and approval['run_id']=='A_reasonable_seed20261104'
 assert p['argv'][p['argv'].index('--spec')+1]==str(specpath) and p['argv'][p['argv'].index('--output-run')+1]==str(run)
 ledger=json.loads((run/'ATOMIC_LEDGER.json').read_text());assert ledger['attempted']==ledger['completed']==ledger['budget']==128 and ledger['pending']==[] and len(set(ledger['done']))==128
 steps=[json.loads(x) for x in (run/'STEPS.jsonl').read_text().splitlines()]
 assert len(steps)==128 and [x['step'] for x in steps]==list(range(1,129))
 dead=steps[0]['dead_sha_before'];previous=steps[0]['model_sha_before']
 for i,x in enumerate(steps,1):
  assert x['split']=='train' and x['sample_id']==spec['train_order'][(i-1)%len(spec['train_order'])] and x['classification']=='FORMAL_RESEARCH_TRAINING'
  assert x['model_sha_before']==previous and x['model_sha_after']!=previous;previous=x['model_sha_after']
  assert x['dead_sha_before']==x['dead_sha_after']==x['isolation']['dead_sha_after']==dead and x['raw15_21_nonzero']==0
  assert all(v['nonzero']==0 for v in x['isolation']['state_after'].values())
  assert all(np.isfinite(x['raw0_15_gradient'])) and len(x['raw0_15_gradient'])==15 and np.isfinite(x['loss_before'])
  assert all(v['gradient_norm']>0 and np.isfinite(v['gradient_norm']) for v in x['module_gradients'].values())
  assert all(y['tensors']['step']['norm']==i for y in x['optimizer_state'])
  assert x['atomic_ledger']['completed']==i and x['atomic_ledger']['attempted']==i and not x['atomic_ledger']['pending']
  assert len(set(x['atomic_ledger']['done']))==i
  assert x['atomic_ledger']['done'][-1]==ledger['done'][i-1]
 points=[];model=model_for(spec,20261104);before_dead=None
 with torch.no_grad(),patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
  for k in [0,8,32,64,128]:
   path=run/('step%03d.pt'%k);side=json.loads(Path(str(path)+'.json').read_text())
   assert side['file_sha256']==sha(path) and side['cursor']==k
   blob=torch.load(path,map_location='cpu',weights_only=True);assert blob['cursor']==k
   meta=blob['metadata'];assert meta['runspec_sha256']==specsha and meta['approval_sha256']==approsha and meta['engineering_only'] is False and meta['trainer_id']==spec['trainer_id']
   model.load_state_dict(blob['model'],strict=True);msha=state_sha(model);assert msha==side['model_sha256']
   assert msha==(steps[0]['model_sha_before'] if k==0 else steps[k-1]['model_sha_after'])
   dead_params=[model.trajectory_head.mlp[-1].weight[15:],model.trajectory_head.mlp[-1].bias[15:]]
   assert sum(t.numel() for t in dead_params)==774 and digest(dead_params)==dead
   if before_dead is None:before_dead=[v.clone() for v in dead_params]
   assert all(torch.equal(v,b) for v,b in zip(dead_params,before_dead))
   for index,param in enumerate(model.parameters()):
    key=blob['optimizer']['param_groups'][0]['params'][index];state=blob['optimizer']['state'].get(key,{})
    if k:
     assert int(state['step'])==k
     recorded=steps[k-1]['optimizer_state'][index]['tensors']
     for name,value in state.items():
      if isinstance(value,torch.Tensor):
       assert digest([value])==recorded[name]['sha256']
       assert torch.isfinite(value).all()
       if param is model.trajectory_head.mlp[-1].weight or param is model.trajectory_head.mlp[-1].bias:
        if name in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:assert value[15:].count_nonzero()==0
   saved=json.loads((run/('evaluation%03d.json'%k)).read_text());assert saved['model_sha_before']==saved['model_sha_after']==msha and saved['dev_backward_calls']==0
   replay=evaluate(model,make_objective(),spec['train']+spec['dev'])
   if replay!=saved:raise ValueError('EXACT_REPLAY_MISMATCH step'+str(k))
   write(o/('REPLAY_EVALUATION%03d.json'%k),replay)
   points.append(dict(step=k,path=str(path),file_sha256=sha(path),model_sha256=msha,evaluation_exact=True,checkpoint_sidecar_sha256=sha(str(path)+'.json'),summary=replay['summary']))
   print('EXACT_REPLAY_PASS',k,replay['summary'],flush=True)
 write(o/'FORMAL128_IDENTITY_ACCEPTANCE.json',dict(status='PASS',component_id=COMPONENT_ID,verified_updates=128,new_network_updates=0,
  process=dict(path=str(process/'PROCESS_RESULT.json'),sha256=sha(process/'PROCESS_RESULT.json'),elapsed_s=p['elapsed_s'],exit_code=p['exit_code'],timed_out=p['timed_out']),
  runspec_sha256=specsha,approval_sha256=approsha,dead774_sha256=dead,shared_weight_sha_chain=True,
  historical_dev_no_backward='source audit+training-row membership+unchanged eval gradient guard;no claim of an independent historical autograd trace',points=points,
  postrun_process_exit='evidence-proven completed execution,not scientific success'))
 return points
