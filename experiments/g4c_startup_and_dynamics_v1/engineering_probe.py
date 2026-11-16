"""At most TWO engineering AdamW updates;never a formal training entry."""
from pathlib import Path
import sys,json,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.training.k1_forest_v1.trainer import verify,model_for
from query_minco.training.k1_forest_v1.data import load
from query_minco.training.k1_physics_trainer_v1.optimizer import DeadRowGuard
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha,group_stats,save_checkpoint,result_numeric,metric_terms,optimizer_report
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.peak_dynamics_proxy_v1.proxy import diagnostic_objective
from query_minco.audit.run_provenance_v1 import AtomicAttemptLedger
from diagnose_dynamics import continuous
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
def put(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def measured(model,obj,row):
 with torch.no_grad():
  b=load(row,'train');out=obj(model(b.depth,b.state),b);loss,peak=diagnostic_objective(out)
  c=continuous(out.trajectory.coeffs[0,0].double().numpy(),out.trajectory.durations[0,0].double().numpy())
  return dict(continuous=c,original_loss=float(out.loss),diagnostic_loss=float(loss),terms=metric_terms(out))
def main():
 torch.set_num_threads(1)
 assert json.loads((O/'UNIT_TEST_RESULT.json').read_text())['status']=='PASS'
 summary=json.loads((O/'DYNAMICS_ZERO_UPDATE_SUMMARY.json').read_text());assert summary['status']=='PASS'
 run=O/'engineering_peak_probe';run.mkdir()
 spec=verify(R/'research_outputs/g4a_forest_shadow_20261111_v1/K1_FOREST_TRAINING_RUNSPEC.json')
 model=model_for(spec,20261104).eval();ck=R/'research_outputs/g4a_forest_shadow_20261111_v1/formal/A_reasonable_seed20261104/step128.pt'
 model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'],strict=True);obj=make_objective()
 wanted=[r['sample_id'] for r in summary['all_records'] if r['step']==128 and r['split']=='train' and r['continuous']['continuous_violation']>1e-6][:2]
 rows=[next(r for r in spec['train'] if r['sample_id']==k) for k in wanted]
 put(run/'FROZEN_SELECTION.json',dict(selection='first2 violating TRAIN in frozen order',ids=wanted,optimizer=dict(lr=1e-5,weight_decay=0.,clip_norm=10),starting_model_sha=state_sha(model),formal_updates=0))
 opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.,foreach=False);guard=DeadRowGuard(model);atomic=AtomicAttemptLedger(run/'ATOMIC_LEDGER.json',2)
 initial={r['sample_id']:measured(model,obj,r) for r in spec['train']};records=[]
 for i,row in enumerate(rows):
  before_eval=measured(model,obj,row);model_before=state_sha(model);before=[p.detach().clone() for p in model.parameters()]
  opt.zero_grad(set_to_none=True);batch=load(row,'train');raw=model(batch.depth,batch.state);raw.retain_grad();out=obj(raw,batch);loss,peak=diagnostic_objective(out)
  assert torch.isfinite(loss);loss.backward()
  assert torch.isfinite(raw.grad).all() and not raw.grad[...,15:].count_nonzero()
  gradients=group_stats(model);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True))
  token=atomic.reserve();compat=run/('guard_attempt%d.json'%i);put(compat,dict(attempted=0,completed=0))
  protection=guard.step(opt,compat,1);ledger=atomic.complete(token)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  after_eval=measured(model,obj,row);delta=after_eval['continuous']['continuous_violation']-before_eval['continuous']['continuous_violation']
  rec=dict(engineering_optimizer_step=i+1,sample_id=row['sample_id'],classification='ENGINEERING_ONLY_NOT_DEPLOYABLE',model_sha_before=model_before,model_sha_after=state_sha(model),
   before=before_eval,after=after_eval,continuous_violation_delta=delta,loss=float(loss.detach()),raw0_15_gradient=raw.grad[0,0,:15].tolist(),
   raw15_21_gradient_nonzero=0,module_gradients=gradients,module_updates=group_stats(model,before),clip_input_norm=norm,dead_rows=protection,optimizer=optimizer_report(opt),ledger=ledger)
  rec['solver_errors_before'],rec['hard_before']=result_numeric(out,batch)
  save_checkpoint(run/('engineering_step%d.pt'%(i+1)),model,opt,guard,dict(engineering_only=True,method_id='peak_proxy_engineering_probe_v1',deployable=False),i+1)
  put(run/('STEP%d.json'%(i+1)),rec);records.append(rec);print('ENGINEERING_STEP',i+1,'CONTINUOUS_DELTA',delta,flush=True)
  if delta>=-1e-7:break
 final={r['sample_id']:measured(model,obj,r) for r in spec['train']}
 initialJ=sum(r['continuous']['continuous_violation'] for r in initial.values());finalJ=sum(r['continuous']['continuous_violation'] for r in final.values())
 good=len(records)==2 and all(r['continuous_violation_delta']<-1e-7 for r in records) and finalJ<initialJ
 put(O/'ENGINEERING_RESULT.json',dict(status='PASS' if good else 'NO_GO',engineering_optimizer_steps=len(records),formal_research_steps=0,
  selected_train=wanted,initial_train=initial,final_train=final,initial_train_continuous_J_sum=initialJ,final_train_continuous_J_sum=finalJ,records=records,dev_backward='NOT_RUN'))
 print('ENGINEERING_QUALIFICATION',good,'STEPS',len(records))
if __name__=='__main__':main()
