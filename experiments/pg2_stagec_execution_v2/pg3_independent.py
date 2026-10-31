"""Disjoint TRAIN scene pilot, frozen C1/Mode/optimizer. No training checkpoint publication."""
from pathlib import Path
import json,time,copy,gc,sys,os,subprocess,hashlib
import numpy as np,torch
from query_minco.training.io import read,write,require,digest
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_next_exact_v1.minimal import MinimalEngine
from query_minco.training.pg3_next_exact_v1.proxy import Parameters,proxy,semantic_sha
from query_minco.training.pg3_next_exact_v1.projection import project,numerical_attempt
from query_minco.training.pg3_preservation_v1.mechanism import local_direction
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,step_guard_v21
from query_minco.training.pg3_v2_no_corridor.executor_v21 import wrap_pg2
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,scene_from_field
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg2_stagec_pg3_preservation_v1.counterfactual import context,sigs
BASE=Path('/home/lizheng/query_minco_mainline/research_outputs/pg2_stagec_execution_v2')
ALPHAS=[1.,.5,.25,.125,.0625,.03125]
def run():
 require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']),'SINGLE_THREAD');torch.set_num_interop_threads(1)
 out=BASE/'PG3_INDEPENDENT001';out.mkdir(exist_ok=False);spec,ctx,cfg=context(out);ds,geo,init,_,train,dev=ctx
 groups={}
 for s,f,meta in sorted(train,key=lambda x:x[0].sample_id):
  key=f.row.get('scene_id')
  if not key:key=digest(scene_from_field(f))
  groups.setdefault(str(key),(s.sample_id,str(key)))
 selected=list(groups.values())[:64];batches=[selected[i:i+8] for i in range(0,len(selected)-7,8)];require(batches,'INSUFFICIENT_DISTINCT_TRAIN_SCENES')
 protocol=dict(selection='first sorted TRAIN SID per distinct authenticated scene_id; first64 such scenes; groups8; before loss evaluation; no Dev selection',cases=batches,case_count=len(batches),unique_samples=len({sid for batch in batches for sid,scene in batch}),unique_scenes=len({scene for batch in batches for sid,scene in batch}),seed=42,checkpoint='same C1 initialization; fresh identical AdamW moments each independent group',alphas=ALPHAS,loss='unchanged PG2 Safe + frozen Minimal Mode/Fair',case_cpu_wall_ceiling_s=30,methods=['ADAMW','GLOBAL_BACKTRACK','EXISTING_QP','FROZEN_QUERY_LOCAL'],cost_protocol='all include common real Adam/audit cost; report all actual proposal/validator CPU; compare decision frontier at common cost budgets, never padded duplicates or fictitious equal work',cost_grid_extra_cpu_s=[1.,2.,4.,8.,16.,30.],independence_limit='different samples/scenes, same two generator families and same C1/seed; not independent seed replicates',Dev_used=False,formal_updates=0)
 write(out/'PREREGISTRATION.json',protocol);results=[]
 for index,batch in enumerate(batches):
  case=out/('case%02d'%index);case.mkdir();ids=[sid for sid,scene in batch];require(not(set(ids)&{x[0].sample_id for x in dev}),'DEV_LEAK');samples=ds.load(ids);e=MinimalEngine(model_from(init),spec,cfg);initial=copy.deepcopy(e.model.state_dict());before_opt=copy.deepcopy(e.opt.state_dict());pack=Parameters(e.model);v=pack.values();start=time.process_time();wall=time.monotonic();z=e.objective(e.model,samples,geo,spec['pg2_weights'],TeacherGuidance(),0);audit,_,_=audit_v21(e.model,z,False);step_guard_v21(audit,cfg);e.exposure.rollback(0)
  passed=z['labels']['passed'].detach().numpy();goals=z['goal_labels'].detach().numpy();matched=z['matched'].numpy();scenes=[scene_from_field(f) for f in z['pg2']['fields']];before_sig=sigs(z,scenes);selected_rows=sorted({int(q) for a in z['assignments'] for q in a['selected']});rec=e.step(samples,geo);adam=pack.values()-v;common_cpu=time.process_time()-start;common_wall=time.monotonic()-wall;e.model.load_state_dict(initial)
  write(case/'AUDIT.json',dict(ids=ids,scene_ids=[scene for sid,scene in batch],audit=audit,guard=rec['guard'],before=passed.tolist(),before_goals=goals.tolist(),matched=matched.tolist(),selected_rows=selected_rows,common_cpu_s=common_cpu,common_wall_s=common_wall,real_AdamW_delta_sha256=semantic_sha(adam),initial_model=state_sha(e.model),formal_checkpoint=False))
  def trial(delta):
   e.model.load_state_dict(initial);pack.set_disposable(v+torch.as_tensor(delta));tick=time.process_time()
   with torch.no_grad():zz=wrap_pg2(pg2_forward(e.model,samples,geo,'query_minco_pg2_guided_v2_safe_v1',TeacherGuidance(),spec['pg2_weights']))
   p=zz['labels']['passed'].numpy();g=zz['goal_labels'].numpy();sign=sigs(zz,scenes);valid=bool(zz['output'].trajectory.solve_valid.all())
   return dict(all_numerical_valid=valid,protected_pass=bool(p[passed].all()),protected_goal_pass=bool(g[passed&goals].all()),accepted=bool(valid and p[passed].all() and g[passed&goals].all()),pre_safe_pass_to_fail=int((passed&~p).sum()),pre_fail_to_pass=int((~passed&p).sum()),matched_safe_degraded=int((passed&matched&~p).sum()),oracle_before=int((passed&goals).any(-1).sum()),oracle_after=int((p&g).any(-1).sum()),known_modes_before=sum(len({x for x in row if x is not None}) for row in before_sig),known_modes_after=sum(len({x for x in row if x is not None}) for row in sign),physics_after=int(p.sum()),goals_after=int(g.sum()),relative_update_norm=float(np.linalg.norm(delta)/adam.norm()),validation_cpu_s=time.process_time()-tick)
  def qp():
   e.model.load_state_dict(initial);fresh=wrap_pg2(pg2_forward(e.model,samples,geo,'query_minco_pg2_guided_v2_safe_v1',TeacherGuidance(),spec['pg2_weights']));G=[]
   require(np.array_equal(fresh['labels']['passed'].detach().numpy(),passed),'QP_BEFORE_LABEL_REPLAY')
   for b,q in zip(*np.nonzero(passed)):
    value,_=proxy(fresh['output'].trajectory.coeffs[b,q],fresh['output'].parameters.durations[b,q],scenes[b]['obstacles']);gg=pack.gradient(value);require(bool(torch.isfinite(gg).all()) and float(gg.norm())>1e-12,'PROXY_GRADIENT');G.append(gg.numpy())
   d,meta=project(adam.numpy(),np.stack(G) if G else np.zeros((0,len(adam))));return d,meta
  methods={}
  for name,fn,alphas in [('ADAMW',lambda:(adam.numpy(),{}),[1.]),('GLOBAL_BACKTRACK',lambda:(adam.numpy(),{}),ALPHAS),('EXISTING_QP',qp,ALPHAS),('FROZEN_QUERY_LOCAL',lambda:(lambda pair:(pair[0].numpy(),pair[1]))(local_direction(pack,adam,selected_rows)),ALPHAS)]:
   t=time.process_time();w=time.monotonic();trials=[];meta={}
   try:
    direction,meta=fn();require(np.isfinite(direction).all() and np.linalg.norm(direction)>0,'NO_NONZERO_DIRECTION')
    for alpha in alphas:
     require(time.process_time()-t<30 and time.monotonic()-w<30,'COST_BUDGET');r=numerical_attempt(trial,alpha*direction);r.update(alpha=alpha,cumulative_extra_cpu_s=time.process_time()-t,cumulative_total_cpu_s=common_cpu+time.process_time()-t);trials.append(r)
     if r.get('accepted'):break
    chosen=next((r for r in trials if r.get('accepted')),None);result=dict(status='ACCEPTED' if chosen else 'REJECTED',chosen=chosen,trials=trials,meta=meta)
   except Exception as ex:result=dict(status='BLOCKED',reason=str(ex),chosen=None,trials=trials)
   result.update(extra_cpu_s=time.process_time()-t,total_cpu_s=common_cpu+time.process_time()-t,extra_wall_s=time.monotonic()-w,common_cpu_s=common_cpu);methods[name]=result;write(case/(name+'.json'),result);print('INDEPENDENT',index,name,result['status'],flush=True)
  e.model.load_state_dict(initial);e.opt.load_state_dict(copy.deepcopy(before_opt));require(state_sha(e.model)==spec['initialization']['weights_sha256'] and semantic_sha(e.opt.state_dict())==semantic_sha(before_opt),'DISPOSABLE_ISOLATION')
  frontiers=[]
  for budget in protocol['cost_grid_extra_cpu_s']:
   point=dict(extra_cpu_budget_s=budget,total_cpu_budget_s=common_cpu+budget,methods={})
   for name,r in methods.items():
    available=[x for x in r['trials'] if x['cumulative_extra_cpu_s']<=budget];chosen=next((x for x in available if x.get('accepted')),None);point['methods'][name]=dict(available_validated_update=chosen is not None,hard_repairs=None if chosen is None else chosen['pre_fail_to_pass'],safe_losses=None if chosen is None else chosen['pre_safe_pass_to_fail'])
   frontiers.append(point)
  result=dict(case=index,ids=ids,status='COMPLETE',methods=methods,matched_cost_frontiers=frontiers,isolation='PASS',formal_updates=0);write(case/'RESULT.json',result);results.append(result);del e,z;gc.collect()
 write(out/'RESULT.json',dict(cases=results,status='COMPLETE',decision='NO_GO_UNLESS_INCREMENT_PROVED_IN_REPORT',formal_updates=0,Dev_used=False,case_protocol=protocol))
if __name__=='__main__':run()
