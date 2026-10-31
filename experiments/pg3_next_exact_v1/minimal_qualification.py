"""Qualification only; there is deliberately no formal training action."""
import copy
import torch
from query_minco.training.io import read,write,digest,require,artifact
from query_minco.training.pg3_next_exact_v1.minimal import MinimalObjective,MinimalEngine,ID,WEIGHTS
from query_minco.training.pg3_v2_no_corridor.executor_v21 import Engine,publish_checkpoint,restore_checkpoint,inspect_chain
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,step_guard_v21
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.initialization import configure
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_v2_no_corridor.formal_v21 import state_fingerprint
from .common import runpath,snapshot,ref

def run(out,context,prep,code,config):
    folder=out/'MINIMAL';folder.mkdir(exist_ok=False);(folder/'RUN_SPECS').mkdir();(folder/'short').mkdir();ds,geo,init,qualification,train,dev=context
    model=model_from(init);initial=state_sha(model);obj=MinimalObjective(model);records=[]
    for i in range(8):
        ids,_=geo.orders[42].batch(i);z=obj(model,ds.load(ids),geo,read(runpath('A')/'RUN_SPEC.json')['pg2_weights'],TeacherGuidance(),0);r,vectors,total=audit_v21(model,z,False);guard=step_guard_v21(r,config);require(r['ratios']['R_D']<=10,'NEXT_MINIMAL_CPU0_GT10')
        require(z['base_loss'] is z['pg2']['loss'] and r['components']['pg2/braking']['effective_weight']==r['braking_weight'],'NEXT_MINIMAL_BRAKING_CHANGED')
        require(all(r['components'][k]['parameter_norm']==0 for k in ('geom','stop','keep','div')),'NEXT_MINIMAL_FORBIDDEN_GRADIENT')
        records.append(dict(batch=i,ids=ids,audit=r,guard=guard));obj.assignment.rollback(0);print('MINIMAL_CPU',i,r['ratios'],flush=True)
    require(state_sha(model)==initial and obj.assignment.state_dict()['history']=={},'NEXT_MINIMAL_CPU_MUTATION')
    write(folder/'GRADIENT_AUDIT.json',dict(status='VERIFIED',records=records,weights=WEIGHTS,code=code,uses_development=False,recalibrated=False))
    write(folder/'CPU0.json',dict(status='PASS',ratios=records[0]['audit']['ratios'],threshold=10,increment_limit=2,weights=WEIGHTS,code=code,weights_before=initial,weights_after=state_sha(model),optimizer_steps=0))
    specs={}
    for arm in ('A','MINIMAL'):
        base=copy.deepcopy(read(runpath('A')/'RUN_SPEC.json'));base.update(schema='pg3_next_minimal_qualification_v1',arm=arm,code=code,sources=snapshot(),bindings=prep['bindings'],weights=WEIGHTS if arm=='MINIMAL' else {k:0. for k in WEIGHTS},method_id=ID if arm=='MINIMAL' else base['method_id'],preparation=str(folder),output_root=str(out/'FORMAL_NOT_AUTHORIZED'),run_id='A42_'+arm+'_NEXT_EXACT128_run001')
        base['next_exact_qualification']=str(folder/'FINAL_QUALIFICATION.json');base['execution']=dict(base['execution'],limit=128,purpose='PG3_V21_PAIRED128');base['formal_authorization_required']=True;base['formal_authorized']=False
        write(folder/'RUN_SPECS'/(arm+'_FORMAL_PLAN.json'),base);specs[arm]=base
        fingerprints={};monitors={}
        for suffix in ('continuous4','resume2plus2'):
            spec=copy.deepcopy(base);spec.update(run_id='A42_'+arm+'_'+suffix+'_run001',output_root=str(folder/'short'));spec['execution']=dict(spec['execution'],limit=4,purpose='PG3_V21_SHORT4')
            write(folder/'RUN_SPECS'/(arm+'_'+suffix+'.json'),spec);path=folder/'short'/spec['run_id'];path.mkdir();write(path/'RUN_SPEC.json',spec)
            configure(dict(seed=42,threads=1,device='cpu'));factory=MinimalEngine if arm=='MINIMAL' else Engine;e=factory(model_from(init),spec,config);last=publish_checkpoint(path,e,dict(completed=0))
            while e.completed<4:
                ids,_=geo.orders[42].batch(e.completed);record=e.step(ds.load(ids),geo);last=publish_checkpoint(path,e,record,ref(last/'COMMIT.json')['sha256'])
                if suffix=='resume2plus2' and e.completed==2:
                    e=factory(model_from(init),spec,config);last=restore_checkpoint(path,e)
            inspect_chain(path,spec);fingerprints[suffix]=state_fingerprint(last/'checkpoint.pt');monitors[suffix]=read(last/'MONITOR.json')
        require(len(set(fingerprints.values()))==1 and monitors['continuous4']==monitors['resume2plus2'],'NEXT_MINIMAL_EXACT_RESUME')
        write(folder/(arm+'_SHORT.json'),dict(status='PASS',fingerprints=fingerprints,monitor_rng_equal=True,continuous_updates=4,resume='2+2',schedule='first4 of original128-index cosine'))
    write(folder/'QUALIFICATION.json',dict(status='PREQUALIFIED_PENDING_T01_T12',method_id=ID,code=code,sources=snapshot(),gradient_audit=ref(folder/'GRADIENT_AUDIT.json'),cpu0=ref(folder/'CPU0.json'),short=ref(folder/'MINIMAL_SHORT.json'),baseline_short=ref(folder/'A_SHORT.json'),formal_plans={a:ref(folder/'RUN_SPECS'/(a+'_FORMAL_PLAN.json')) for a in specs},formal_training='NOT_RUN',formal_authorized=False,old_M_reused=False,original_braking_retained=True,no_extra_stop_computed=True,weights=WEIGHTS))
