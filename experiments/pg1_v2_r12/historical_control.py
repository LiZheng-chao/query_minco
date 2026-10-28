"""Read-only replay of all32 historical pre-update states; no optimizer."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from query_minco.training.io import read,write,sha,digest,require
from query_minco.training.pg1.spec import qualify_bindings,lr_at
from query_minco.training.run_spec import ROOT
from query_minco.training.pg1_v2_r12.objective import forward
from query_minco.training.pg1_v2_r12.spec import R0
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
out=Path(sys.argv[1]);bbpath=Path(sys.argv[2]);start=time.monotonic()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('HISTORICAL_READONLY_NO_OPTIMIZER')) as guard:
 ds,geo,init,_=qualify_bindings(dict(path=str(bbpath),sha256=sha(bbpath)),42)
 run=ROOT/'research_outputs/pg1_task_complete_v1/local_cpu_pilot32_v1/runs/A42_PG1complete_CPU_Pilot32_run001'
 seal=read(run/'RUN_SEAL.json');require(all(sha(run/p)==h for p,h in seal['files'].items()),'HISTORY_SEAL')
 s=read(run/'RUN_SPEC.json');root_spec=read(run/'ROOT_SPEC.json')
 oldbinding=read(s['bindings']['path']);newbinding=read(bbpath)
 for k in oldbinding:
  if k!='code':require(oldbinding[k]==newbinding[k],'HISTORY_ASSET_'+k)
 source_equal=all(sha(ROOT/p)==h for p,h in s['code']['files'].items())
 require(source_equal,'HISTORY_OLD_SOURCE_SHA')
 require(s['method']=='query_minco_pg1_task_complete_v1' and s['seed']==42 and s['device']=='cpu' and s['cpu_updates']==32,'HISTORY_CONTRACT')
 require(s['batch_size']==8 and s['precision']=='FP32' and s['optimizer']==dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),'HISTORY_OPTIMIZER_RECIPE')
 records=[];model=QueryMincoPlanner();model.train();all_equal=True
 for i in range(1,33):
  require(time.monotonic()-start<180,'HISTORY_WALL')
  cp=run/'commits'/('step%04d'%(i-1))/('checkpoint_step%04d.pt'%(i-1))
  receipt=read(str(cp)+'.json');require(sha(cp)==receipt['sha256'],'HISTORY_CP_SHA')
  blob=torch.load(str(cp),map_location='cpu',weights_only=True)
  require(blob['spec_sha256']==digest(root_spec) and blob['completed_optimizer_updates']==i-1,'HISTORY_SPEC_CURSOR')
  model.load_state_dict(blob['state_dict'],strict=True);bh=state_sha(model)
  require(bh==receipt['weights_sha256'],'HISTORY_CP_WEIGHTS')
  if i==1:require(bh==init['weights_sha256'],'HISTORY_COMMON_C1')
  rec=read(run/'commits'/('step%04d'%i)/'UPDATE.json');ids,_=geo.orders[42].batch(i-1)
  require(rec['ids']==ids and rec['lr']==lr_at('pg1_pilot32_v1',i-1),'HISTORY_ORDER_LR')
  z=forward(model,ds.load(ids),geo,R0)
  with np.load(run/'intents'/('step%04d_ARRAYS.npz'%i),allow_pickle=False) as zz:
   eq={k:np.array_equal(z[k].detach().numpy(),zz[k]) for k in ['raw','positions','velocities','time_weights']}
   eq['coeffs']=np.array_equal(z['output'].trajectory.coeffs.detach().numpy(),zz['coeffs'])
   eq['durations']=np.array_equal(z['output'].parameters.durations.detach().numpy(),zz['durations'])
   for k in ['imitation','obstacle','goal','dynamic','domain']:
    g=torch.autograd.grad(z['terms'][k],z['raw'],retain_graph=True)[0].detach().numpy()
    eq['grad_'+k]=np.array_equal(g,zz['grad_'+k])
    if k!='imitation':eq['per_query_'+k]=np.array_equal(z['per_query'][k].detach().numpy(),zz['per_query_'+k])
   eq['total_raw_gradient']=np.array_equal(torch.autograd.grad(z['loss'],z['raw'])[0].detach().numpy(),zz['raw_grad'])
  eq['loss_exact']=float(z['loss'].detach())==rec['loss']
  eq['matches_exact']=z['matches']==rec['matches']
  require(state_sha(model)==bh,'HISTORY_NO_UPDATE')
  all_equal=all_equal and all(eq.values());records.append(dict(step=i,checks=eq,weights_unchanged=True))
 result=dict(schema='r12_historical_r0_readonly_v1',mathematical_reference_eligible=all_equal,
   same_assets=True,old_source_hashes_equal=True,old_method_not_renamed=True,
   optimizer_created=False,optimizer_steps=0,scope='historical reference eligibility, NOT new source training qualification',
   history_seal=dict(path=str(run/'RUN_SEAL.json'),sha256=sha(run/'RUN_SEAL.json')),
   historical_run=str(run),records=records,wall_s=time.monotonic()-start,source_bundle_is_same=False,
   recommendation='fresh paired R0 preferred for identical new runtime bundle; old reference may be cited only with this explicit equivalence audit')
 write(out,result);require(guard.call_count==0,'HISTORY_OPTIMIZER_GUARD')
 print('HISTORICAL_REFERENCE',all_equal,len(records),round(time.monotonic()-start,2))
