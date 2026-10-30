"""Frozen-checkpoint gradient replay; no optimizer, no archived parameter writes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,require
from query_minco.training.pg3_v2_no_corridor.executor_v21 import wrap_pg2
from query_minco.training.pg3_v2_no_corridor.objective import ObjectiveV21,KEYS_V21
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg1_v2_r12.diagnostics import flat,cosine
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.evaluation.signed_geometry import fine_geometry
from query_minco.inference.identity import state_sha
from .ablation_v21 import load_context,bind_context,HISTORY
from experiments.pg3_feasible_mode_v1.common import model_from
SID='f00_narrow3d_g5_v1'
def clearance_at_frozen_branch(o,scene,q=3):
    c=o.trajectory.coeffs[0,q];d=o.parameters.durations[0,q]
    diag=fine_geometry(c.detach().numpy(),d.detach().numpy(),scene)
    t=diag['closest_time_s'];cum=np.cumsum(d.detach().numpy());piece=min(int(np.searchsorted(cum,t,side='right')),2)
    local=c.new_tensor(t)-d[:piece].sum();point=(c[piece]*local**torch.arange(6,dtype=c.dtype)).sum(-1)
    box=next(b for b in scene['obstacles'] if b['name']==diag['closest_box']);center=point.new_tensor(np.add(box['min'],box['max'])/2);half=point.new_tensor(np.subtract(box['max'],box['min'])/2)
    v=(point-center).abs()-half;clear=v.clamp_min(0).norm()+v.max().clamp_max(0)-.12
    return clear,dict(diag,piece=piece,frozen_global_time_s=t)
def replay(out):
    out=Path(out);require(out.is_dir(),'LOCAL_ATTR_OUTPUT');binding=bind_context(out);prep,(ds,geo,init,qualification,train,dev)=load_context(binding);sample,field,meta=next(x for x in train if x[0].sample_id==SID);results=[]
    for arm in ('A','D'):
        run=HISTORY/'runs'/('A42_'+arm+'_PG3V21_formal_run001');spec=read(run/'RUN_SPEC.json')
        for update in (1,92,118,128):
            before=torch.load(run/'commits'/('step%04d'%(update-1))/'checkpoint.pt',weights_only=True);after=torch.load(run/'commits'/('step%04d'%update)/'checkpoint.pt',weights_only=True)
            m=model_from(init);reference_sha=state_sha(m);obj=ObjectiveV21(m,reference_sha,spec['weights']) if arm=='D' else None
            m.load_state_dict(before['model']);m.train();torch.set_rng_state(before['torch_rng'])
            if obj:obj.assignment.load_state_dict(before['exposure'])
            ids,_=geo.orders[42].batch(update-1);samples=ds.load(ids);context=TeacherGuidance()
            z=obj(m,samples,geo,spec['pg2_weights'],context,update-1) if obj else wrap_pg2(pg2_forward(m,samples,geo,spec['method_id'],context,spec['pg2_weights']))
            audit,vectors,total=audit_v21(m,z,False);record=read(run/'commits'/('step%04d'%update)/'UPDATE.json')
            require(abs(float(total.norm())-record['gradient_norm_before_clip'])<2e-5,'LOCAL_REPLAY_GRADIENT_MISMATCH')
            pp=list(m.parameters());raw=m(depth_input=sample.depth_input[None],state=sample.state[None]);o=TrajectoryPostprocessor()(raw,sample.state[None]);clear,branch=clearance_at_frozen_branch(o,meta['scene'])
            cg=flat(torch.autograd.grad(clear,pp,retain_graph=True,allow_unused=True),pp).double()
            delta=torch.cat([(after['model'][n]-before['model'][n]).flatten() for n,p in m.named_parameters()]).double()
            comps={};selected={k:vectors[k]*audit['components'][k]['effective_weight'] for k in vectors if k.startswith('pg2/') or k in KEYS_V21};selected['total']=total;selected['actual_adam_delta']=delta
            offset=0;groups={}
            for n,p in m.named_parameters():
                group='learnable_queries' if n=='query_decoder.learned_queries' else n.split('.')[0]
                groups.setdefault(group,[]).append((offset,offset+p.numel()));offset+=p.numel()
            for name,v in selected.items():
                vec=v.double();comps[name]=dict(norm=float(vec.norm()),local_clearance_directional_derivative=float(cg@ (vec if name=='actual_adam_delta' else -vec)),module_norm={g:float(torch.cat([vec[a:b] for a,b in parts]).norm()) for g,parts in groups.items()},module_clearance_derivative={g:sum(float(cg[a:b]@(vec[a:b] if name=='actual_adam_delta' else -vec[a:b])) for a,b in parts) for g,parts in groups.items()})
            events=[]
            for b,sid in enumerate(ids):
                if sid!=SID:continue
                events.append(dict(batch=b,matched=bool(z['matched'][b,3]),hard_safe=bool(z['labels']['passed'][b,3]),component_raw_gradient_norms={k:audit['components'][k]['raw_per_query'][b][3] if audit['components'][k]['raw_per_query'] is not None else None for k in ('mode','geom','stop','keep')},geometry=[x for x in z['geometry'] if x['batch']==b and x['query']==3]))
            results.append(dict(arm=arm,update=update,sample_id=SID,query=3,before_model_sha256=before['model_sha256'],after_model_sha256=after['model_sha256'],replayed_gradient_norm=float(total.norm()),recorded_gradient_norm=record['gradient_norm_before_clip'],clearance_m=float(clear.detach()),branch=branch,actual_delta_first_order_clearance_change_m=float(cg@delta),actual_parameter_delta_norm=float(delta.norm()),components=comps,query_in_training_batch=SID in ids,training_events=events,parameter_source_unchanged=state_sha(m)==before['model_sha256']))
            if obj:obj.assignment.rollback(update-1)
            print('LOCAL_REPLAY',arm,update,'clearance',float(clear.detach()),'delta derivative',float(cg@delta),flush=True)
    write(out/'LOCAL_GRADIENT_REPLAY.json',dict(status='PASS',records=results,semantics='First-order frozen-time nearest-box sampled-clearance derivatives. Component derivatives use negative effective component gradient, not Adam counterfactual; actual delta uses archived AdamW update. Local evidence only; no full-training causal claim. No optimizer created or source checkpoint modified.'))
    text=['# Frozen local gradient replay','','No optimizer was created. Archived checkpoint parameters were loaded read-only into disposable models. The gradient norm of each replay matches its historical update. Learnable-query norms are measured separately from the remaining Decoder in this replay.','','| Arm | Update | Q3 in batch | Before clearance m | First-order change along actual Adam delta m |','|---|---:|---|---:|---:|']
    for r in results:text.append('| %s | %d | %s | %.9f | %.9f |'%(r['arm'],r['update'],r['query_in_training_batch'],r['clearance_m'],r['actual_delta_first_order_clearance_change_m']))
    text+=['','Per-loss and Encoder/Decoder/Learnable Queries/Head norms and directional contributions are in LOCAL_GRADIENT_REPLAY.json. These are local derivatives of a sampled nearest-box branch, not continuous-clearance certificates. Negative component derivatives or gradient cosines do not prove long-run harm; archived Adam momentum/preconditioning prevents treating the negative component gradient as the actual optimizer change.']
    (out/'LOCAL_GRADIENT_REPLAY.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='research_outputs/pg3_v21_attribution_v1');a=p.parse_args();replay(a.output)
