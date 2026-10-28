"""Frozen384 CPU-only R1.1 diagnostic, no optimizer/registration."""
import argparse,json,time,resource,traceback,subprocess,collections,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from query_minco.training.io import read,write,sha,artifact,digest
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg1.objective import forward
from query_minco.training.losses.set_physical_risk import physical_loss
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss,bottleneck_safety_loss
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.evaluation.goal_config import CONFIG
from experiments.pg1_v2_r1.cpu0_readonly import closest,metrics,TARGETS,OLDROOT
from experiments.pg1_v2_r11.math_probes import aggregate,current_time_quality,project_quality,mark_first_visits
TERMS=('imitation','obstacle','goal','limit','domain','QA','QJ','QT','lse','tail',
       'mean_plain','mean_rho','lse_rho','tail_rho','QA_current','QJ_current')
AGGS=('obstacle','mean_plain','mean_rho','lse','lse_rho','tail','tail_rho')
PROBES=AGGS+('QA','QJ','QT','QA_current','QJ_current','safety_priority_QA')
def norm(x):return float(np.linalg.norm(x))
def cos(a,b):
 n=norm(a)*norm(b)
 return float(np.dot(a,b)/n) if n>1e-24 else None
def arr(x):return x.detach().numpy()
def flattened(gs,params):
 return np.concatenate([arr(g if g is not None else torch.zeros_like(p)).ravel() for g,p in zip(gs,params)])
def budget(start,output):
 if time.monotonic()-start>1800:raise RuntimeError('WALL_CAP_1800')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4*1024**2:raise RuntimeError('RSS_CAP_4GiB')
 mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')}
 if mem['MemAvailable']<3*1024**2:raise RuntimeError('MEM_AVAILABLE_LT3GiB')
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>512*1024**2:raise RuntimeError('OUTPUT_CAP')
def graph(model,samples,geo):
 z=forward(model,samples,geo,'query_minco_pg1_task_complete_v1')
 raw=z['raw'];tr=z['output'].trajectory
 fields=geo.load(samples);boxes=[f.boxes for f in fields]
 qa=dynamic_quality_loss(tr)
 safe=bottleneck_safety_loss(z['positions'],z['time_weights'],boxes,geometry_bound=[True]*len(fields),
    certified_empty=[f.row['empty_certified'] for f in fields],temperature=.025,tail_fraction=.1)
 obs,diag=physical_loss(z['positions'],z['velocities'],z['time_weights'],boxes,geometry_bound=[True]*len(fields),
    radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
 assert torch.equal(obs,z['per_query']['obstacle'])
 aa=aggregate(safe['point_risk'],z['time_weights'],diag['risk_weights'])
 per=dict(obstacle=obs,goal=z['per_query']['goal'],limit=z['per_query']['dynamic'],domain=z['per_query']['domain'],
    QA=qa['terms']['acceleration'],QJ=qa['terms']['jerk'],QT=qa['terms']['time'],**aa)
 per.update(current_time_quality(per['QA'],per['QJ'],qa['total_time_s']))
 im=raw[...,0]*0
 for b,(cost,m) in enumerate(zip(z['costs'],z['matches'])):
  qq=m['queries'];tt=m['teachers']
  assert len(qq) in (1,4)
  im[b,qq]=cost[qq,tt]*6/len(qq)
 per['imitation']=im
 scalars={k:v.mean() for k,v in per.items()}
 assert torch.allclose(scalars['imitation'],z['terms']['imitation'],rtol=2e-6,atol=1e-9)
 assert torch.allclose(per['mean_rho'],obs,rtol=2e-6,atol=1e-8)
 return z,per,safe,diag,fields
def scene(field,geo):
 row=field.row;s=geo._read(row['scene_source'])
 for key in row['scene_pointer'].strip('/').split('/'):s=s[int(key)] if isinstance(s,list) else s[key]
 return s
def param_grads(raw,g,params):
 if not bool(torch.count_nonzero(g)):
  return np.zeros(sum(p.numel() for p in params),dtype=np.float32)
 return flattened(torch.autograd.grad(raw,params,grad_outputs=g,retain_graph=True,allow_unused=True),params)
def save_model_snapshot(output,modelname,payload,models,elapsed):
 for name,records in payload.items():
  write(output/'runs'/(modelname+'_'+name+'.json'),dict(records=records))
 write(output/'audit'/('PROGRESS_'+modelname+'.json'),dict(models=models,wall_s=elapsed))

def main(output,models_requested=None,directions_from=None):
 start=time.monotonic();thread_start=time.thread_time()
 assert output.is_absolute() and output.parent==ROOT/'research_outputs/pg1_v2_r11'
 assert not (output/'audit/STARTED.json').exists(),'RUN_ALREADY_STARTED'
 write(output/'audit/STARTED.json',dict(monotonic=start))
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 oldread=ROOT/'research_outputs/pg1_v2_r1/cpu0_002/_preparation/BINDINGS_CPU0_ONLY.json'
 binding=read(oldread);binding['code']=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=source_files(),allow_uncommitted_sources=True)
 bpath=output/'_preparation/CPU0_ONLY_BINDINGS.json';write(bpath,binding)
 ds,geo,ini,qual=qualify_bindings(dict(path=str(bpath),sha256=sha(bpath)),42);write(output/'audit/QUALIFICATION.json',qual)
 seen=set();batches=[]
 for i in range(len(geo.orders[42].ids)):
  ids,roles=geo.orders[42].batch(i);batches.append(ids);seen.update(ids)
  if seen==set(ds.rows):break
 assert len(ds.rows)==384 and len(batches)==96 and len(seen)==384
 # Independent selection BEFORE reading losses: first two per M x Near/Far.
 strata=collections.defaultdict(list)
 for ids in batches:
  for sid in ids:
   row=ds.rows[sid];k='M%d_%s'%(row['M'],row['regime'])
   if sid not in strata[k] and len(strata[k])<2:strata[k].append(sid)
 selected=[sid for v in strata.values() for sid in v]
 write(output/'_preparation/COVERAGE_CLARIFICATION.json',dict(reason='first48 has288 unique, actualFrozenOrder has repeats; no artificial epoch',batches=96,visits_per_model=768,unique_per_model=384,ordered_batches=batches,order_digest=digest(batches),training_direction_subset=strata,subset_ids=selected,selected_before_losses=True))
 dev=None
 checkpoints=[('C1_step0',OLDROOT/'A42_E2paired_CPU_Pilot32_run001/checkpoints/checkpoint_step0000.pt'),
              ('E2_step32',OLDROOT/'A42_E2paired_CPU_Pilot32_run001/checkpoints/checkpoint_step0032.pt'),
              ('PG1_step32',OLDROOT/'A42_PG1complete_CPU_Pilot32_run001/checkpoints/checkpoint_step0032.pt')]
 if models_requested is not None:checkpoints=[x for x in checkpoints if x[0] in models_requested]
 models=[];summary=[];direction_rows=[];allcases=[];alltasks=[];timings=[]
 for modelname,cp in checkpoints:
  model=QueryMincoPlanner();model.load_state_dict(torch.load(str(cp),map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval()
  bh=state_sha(model);params=list(model.parameters());cache={};visited=set()
  if directions_from:
   source_rows=[json.loads(l) for l in (directions_from/'runs'/(modelname+'_VALUES.jsonl')).read_text().splitlines()]
   for row in source_rows:
    cache.setdefault(row['sample_id'],[None if c is None else dict(surface_clearance_m=c,time_s=t) for c,t in zip(row['surface_clearances_m'],row['closest_times_s'])])
   visited=set(cache);assert len(visited)==384
  gradfile=output/'runs'/(modelname+'_GRADIENTS.jsonl');valfile=output/'runs'/(modelname+'_VALUES.jsonl')
  with gradfile.open('x') as fg,valfile.open('x') as fv:
   for bi,ids in enumerate([] if directions_from else batches):
    budget(start,output);bt=time.monotonic();samples=ds.load(ids)
    z,per,safe,diag,fields=graph(model,samples,geo);raw=z['raw'];mask=z['matched'];B=len(samples)
    new=torch.tensor(mark_first_visits(ids,visited),dtype=torch.bool)
    c=arr(z['output'].trajectory.coeffs);d=arr(z['output'].trajectory.durations)
    clear=np.zeros((B,6));tasks=[]
    for b,(sample,field) in enumerate(zip(samples,fields)):
     sc=scene(field,geo);sid=sample.sample_id
     if sid not in cache:
      if field.row['empty_certified']:cc=[None]*6
      else:cc=[closest(c[b,q],d[b,q],sc) for q in range(6)]
      cache[sid]=cc
     # Numeric continuous classifier, not certificate.
     clear[b]=[float('inf') if x is None else x['surface_clearance_m'] for x in cache[sid]]
     t=dict(model=modelname,batch=bi,sample_id=sid,first_visit=bool(new[b]),M=len(sample.teacher_coeffs),regime=ds.rows[sid]['regime'],family=field.row['family'],matched=mask[b].tolist(),
       surface_clearances_m=[None if x is None else x['surface_clearance_m'] for x in cache[sid]],closest_times_s=[None if x is None else x['time_s'] for x in cache[sid]],
       terms={k:arr(v[b]).tolist() for k,v in per.items()})
     tasks.append(t);fv.write(json.dumps(t,allow_nan=False)+'\n')
     if bool(new[b]):alltasks.append(t)
    group=dict(all=torch.ones_like(mask),matched=mask,unmatched=~mask,
       M1=torch.tensor([len(s.teacher_coeffs)==1 for s in samples])[:,None].expand_as(mask),
       M4=torch.tensor([len(s.teacher_coeffs)==4 for s in samples])[:,None].expand_as(mask),
       Near=torch.tensor([ds.rows[s.sample_id]['regime']=='near' for s in samples])[:,None].expand_as(mask),
       Far=torch.tensor([ds.rows[s.sample_id]['regime']=='far' for s in samples])[:,None].expand_as(mask),
       bottleneck=torch.from_numpy(clear<=.08),wide=torch.from_numpy(clear>.15),transition=torch.from_numpy((clear>.08)&(clear<=.15)))
    rg={}
    for k in TERMS:
     g=torch.autograd.grad(per[k].mean(),raw,retain_graph=True)[0]
     assert torch.isfinite(g).all() and not torch.count_nonzero(g[...,19:21]),('RAW',k)
     rg[k]=g.detach()
    # Raw per-query cosines and losses for UNIQUE sample stats.
    rawstats={}
    for k in TERMS:
     g=rg[k];row=dict(norm=arr(g.norm(dim=-1)).tolist(),time15_19=[float(g[...,j].norm()) for j in range(15,19)],raw19_21_zero=True)
     for ref in ['imitation','obstacle']:
      dot=(g*rg[ref]).sum(-1);den=g.norm(dim=-1)*rg[ref].norm(dim=-1)
      row['cos_'+ref]=[[float(dot[b,q]/den[b,q]) if den[b,q]>1e-20 else None for q in range(6)] for b in range(B)]
     rawstats[k]=row
    pvec={};decomp={};rawdump={k:arr(g) for k,g in rg.items()}
    np.savez_compressed(output/'runs'/('%s_batch%03d_RAW.npz'%(modelname,bi)),raw=arr(raw),state=torch.stack([s.state for s in samples]).numpy(),coeffs=c,durations=d,matched=mask.numpy(),first_visit=new.numpy(),**{'g_'+k:v for k,v in rawdump.items()})
    for groupname,gm in group.items():
     if not gm.any():continue
     pv={k:param_grads(raw,rg[k]*gm[...,None],params) for k in TERMS}
     if groupname=='all':pvec=pv
     if groupname in ('matched','unmatched'):decomp[groupname]=pv
     for k,g in pv.items():
      imn=norm(pv['imitation']);obn=norm(pv['obstacle'])
      rec=dict(model=modelname,batch=bi,group=groupname,term=k,query_count=int(gm.sum()),sample_count=int(gm.any(-1).sum()),
       parameter_norm=norm(g),ratio_imitation=norm(g)/imn if imn>1e-20 else None,ratio_obstacle=norm(g)/obn if obn>1e-20 else None,
       cos_imitation=cos(g,pv['imitation']),cos_obstacle=cos(g,pv['obstacle']),
       raw_time15_19=[float((rg[k]*gm[...,None])[...,j].norm()) for j in range(15,19)],
       group_scalar_contribution=float((per[k].detach()*gm).sum()/(B*6)),conditional_query_mean=float(per[k].detach()[gm].mean()),
       group_values_nonzero=int((per[k].detach()[gm]!=0).sum()),
       reduction='partition of same full Batch8 denominator; group not independently renormalized')
      fg.write(json.dumps(rec,allow_nan=False)+'\n')
    for k in TERMS:
     np.testing.assert_allclose(decomp['matched'][k]+decomp['unmatched'][k],pvec[k],atol=2e-6,rtol=3e-4)
    # First-order total budget, no optimizer; each quality component gets shared .1 of I norm.
    im=pvec['imitation'];obs=pvec['obstacle'];base=im+obs+pvec['goal']+.1*pvec['limit']+pvec['domain']
    qcombo=np.zeros_like(im);weights={}
    for k in ['QA','QJ','QT']:
     weights[k]=.1*norm(im)/(3*norm(pvec[k])) if norm(pvec[k]) else 0.
     qcombo+=weights[k]*pvec[k]
    projected=arr(project_quality(torch.from_numpy(pvec['QA']),torch.from_numpy(obs)))
    s=.1*norm(im)/norm(projected) if norm(projected) else 0.
    summary.append(dict(model=modelname,batch=bi,raw=rawstats,ids=ids,first_visit=new.tolist(),budget=dict(
       QA_QJ_QT_weights=weights,new_norm=norm(qcombo),base_norm=norm(base),total_norm=norm(base+qcombo),
       total_physics_to_imitation=norm(base-im+qcombo)/norm(im) if norm(im) else None,
       new_cos_obstacle=cos(qcombo,obs),QA_projected_norm=norm(projected),QA_projected_cos_obstacle=cos(projected,obs),
       projected_combo_norm=norm(obs+s*projected),projection_weight=s)))
    timings.append(dict(model=modelname,batch=bi,wall_s=time.monotonic()-bt))
    if bi%4==0:print('FULL',modelname,bi+1,'/96 unique',len(visited),'wall',round(time.monotonic()-start,1),flush=True)
  assert len(visited)==384
  # Per-model training-only deterministic direction subset and fixed historical examples.
  if dev is None:
   from query_minco.training.pilot_evaluation import load_rows
   dev={s.sample_id:(s,m) for s,m in load_rows(binding['development']) if s.sample_id in TARGETS}
  contexts=[('TRAIN',sid,ds.load([sid])[0],None) for sid in selected]+[('HISTORICAL_DEV',sid,s,m) for sid,(s,m) in dev.items()]
  for split,sid,sample,meta in contexts:
   budget(start,output)
   if split=='TRAIN':
    z,per,safe,diag,fields=graph(model,[sample],geo);sc=scene(fields[0],geo);raw=z['raw'];qs=range(6)
   else:
    # No synthetic development sidecar: use already authenticated manifest rows.
    from experiments.pg1_v2_r1.cpu0_readonly import feature
    from query_minco.training.losses.imitation import cost_matrix
    from query_minco.training.matching.hungarian import match
    raw=model(depth_input=sample.depth_input[None],state=sample.state[None]);sc=meta['scene']
    out,p,v,w,old,diag,quality,safe=feature(raw,sample.state[None],[raw.new_tensor([[b['min'],b['max']] for b in sc['obstacles']])],[False])
    per=dict(obstacle=old,QA=quality['terms']['acceleration'],QJ=quality['terms']['jerk'],QT=quality['terms']['time'],
      **aggregate(safe['point_risk'],w,diag['risk_weights']))
    per.update(current_time_quality(per['QA'],per['QJ'],quality['total_time_s']))
    cost,_=cost_matrix(out,sample,0);il,qq,tt=match(cost,'B_set');per['imitation']=il.reshape(1,1).expand(1,6)
    z=dict(output=out,time_weights=w);qs=TARGETS[sid]
   pgr={}
   for k in set(PROBES)-{'safety_priority_QA'}:
    pgr[k]=flattened(torch.autograd.grad(per[k].mean(),params,retain_graph=True,allow_unused=True),params)
   gi=flattened(torch.autograd.grad(per['imitation'].mean(),params,retain_graph=True,allow_unused=True),params)
   proj=arr(project_quality(torch.from_numpy(pgr['QA']),torch.from_numpy(pgr['obstacle'])))
   scale=.1*norm(gi)/norm(proj) if norm(proj) else 0.
   pgr['safety_priority_QA']=pgr['obstacle']+scale*proj
   times=(z['output'].trajectory.durations[...,None]*torch.linspace(0,1,64)+torch.cat([z['output'].trajectory.durations[...,:1]*0,z['output'].trajectory.durations.cumsum(-1)[...,:-1]],-1)[...,None]).flatten(2,3)[0].detach().numpy()
   focus={}
   for k in AGGS:
    if k=='obstacle':wg=safe['normalized_time_mass']*diag['risk_weights']
    else:wg=torch.autograd.grad(per[k].sum(),safe['point_risk'],retain_graph=True)[0]
    focus[k]=arr(wg[0])
   baselines={q:metrics(raw.detach(),sample,sc,q) for q in qs}
   for q,base in baselines.items():
    if split=='TRAIN':
     cc=cache[sid][q]
     assert cc is not None and abs(cc['surface_clearance_m']-base['clearance_m'])<3e-5
    rr=dict(model=modelname,split=split,sample_id=sid,query=q,baseline=base,
       per_terms={k:float(v[0,q].detach()) for k,v in per.items()},
       focus={k:float(g[q][np.abs(times[q]-base['closest_time_s'])<=.05].sum()/g[q].sum()) for k,g in focus.items()})
    allcases.append(rr)
   named=list(model.named_parameters());buffers=dict(model.named_buffers())
   for term,g in pgr.items():
    gn=norm(g)
    if gn==0:continue
    unit={};offset=0
    for name,p in named:
     n=p.numel();unit[name]=torch.from_numpy(g[offset:offset+n].reshape(tuple(p.shape)))/gn;offset+=n
    def changed(eta):
     mapping={name:p.detach()-eta*unit[name] for name,p in named};mapping.update(buffers)
     with torch.no_grad():return functional_call(model,mapping,(),dict(depth_input=sample.depth_input[None],state=sample.state[None]))
    probe=changed(.0001);speed=float((probe-raw.detach()).abs().max())/.0001
    eta=min(.001,.001/speed) if speed else .001
    rr=changed(eta);radius=float((rr-raw.detach()).abs().max())
    for q,base in baselines.items():
     after=metrics(rr,sample,sc,q)
     direction_rows.append(dict(model=modelname,split=split,sample_id=sid,query=q,term=term,parameter_l2=eta,raw_max_delta=radius,
       radius_over_nominal=radius>.00105,baseline=base,after=after,loss_before=float(per[term][0,q].detach()) if term in per else None,
       projection_scale=scale if term=='safety_priority_QA' else None,
       projected_first_order_obstacle_dot=float(np.dot(proj,pgr['obstacle'])) if term=='safety_priority_QA' else None))
   assert state_sha(model)==bh,'MODEL_MUTATED'
   print('DIRECTION',modelname,split,sid,'wall',round(time.monotonic()-start,1),flush=True)
  models.append(dict(model=modelname,checkpoint=dict(path=str(cp),sha256=sha(cp)),weights_before=bh,weights_after=state_sha(model),unique_tasks=384,visits=768))
  save_model_snapshot(output,modelname,dict(BATCH_RAW_AND_BUDGET=summary,DIRECTIONS=direction_rows,CASES=allcases,UNIQUE_TASKS=alltasks),models,time.monotonic()-start)
 for name,rr in [('BATCH_RAW_AND_BUDGET',summary),('DIRECTIONS',direction_rows),('CASES',allcases),('UNIQUE_TASKS',alltasks)]:write(output/'runs'/(name+'.json'),dict(records=rr))
 write(output/'audit/CPU0_RESULT.json',dict(status='COMPLETE',models=models,mode='DIRECTION_REPLAY_ONLY' if directions_from else 'FULL_AUDIT',full_stream_source=str(directions_from) if directions_from else None,batches=0 if directions_from else len(models)*96,unique_task_models=len(alltasks),candidate_visits=len(models)*768*6,
    unique_candidate_models=len(models)*384*6,direction_records=len(direction_rows),optimizer_created=False,optimizer_steps=0,
    GPU_jobs=0,training_qualified=False,wall_s=time.monotonic()-start,thread_cpu_s=time.thread_time()-thread_start,
    max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,timings=timings))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--models',nargs='+',choices=['C1_step0','E2_step32','PG1_step32']);p.add_argument('--directions-from',type=Path);a=p.parse_args()
 try:main(a.output,a.models,a.directions_from)
 except BaseException:
  write(a.output/'audit/FAILURE.json',dict(status='FAIL',error=traceback.format_exc(),optimizer_steps=0));raise
