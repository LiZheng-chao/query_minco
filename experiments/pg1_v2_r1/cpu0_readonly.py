"""Predeclared CPU0 R1 probes, read-only frozen models; never a training entry."""
import ast,argparse,json,math,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,write,artifact,sha
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings,recipe
from query_minco.training.pg1.objective import forward as old_forward
from query_minco.training.pilot_evaluation import load_rows,_certified_empty
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss,reduce_set
from query_minco.training.pg1.losses import analytic_samples,goal_task_loss,dynamic_limit_loss,center_domain_loss
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss,bottleneck_safety_loss
from query_minco.evaluation.validation import evaluate,roots,norm_max,validate_teacher
from query_minco.evaluation.geometry import point_box_distance
from query_minco.evaluation.config import CONFIG
from query_minco.evaluation.goal import task_metrics
TARGETS={'f02_down_only_g3_v0':[1,3],'f02_multi3d_g3_v0':[1]}
NAMES=('imitation','obstacle','limit','acceleration','jerk','time','lse','tail')
DIRECTIONS=('obstacle','acceleration','jerk','time','lse','tail')
OLDROOT=ROOT/'research_outputs/pg1_task_complete_v1/local_cpu_pilot32_v1/runs'
SOURCE_GEOMETRY=ROOT/'research_outputs/pg1_v2_diagnostics_v1/_preparation/diagnose.py'
# Reuse the existing, source-hashed diagnostic geometry functions WITHOUT running
# that CLI or writing into its sealed output directory.
scope=dict(np=np,math=math,evaluate=evaluate,roots=roots,norm_max=norm_max,point_box_distance=point_box_distance,CONFIG=CONFIG)
tree=ast.parse(SOURCE_GEOMETRY.read_text())
defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('extrema','nearest_piece','closest')]
assert len(defs)==3
exec(compile(ast.Module(body=defs,type_ignores=[]),str(SOURCE_GEOMETRY),'exec'),scope)
closest=scope['closest']
def norm(x):return float(np.linalg.norm(x))
def cosine(x,y):return float(np.dot(x,y)/(norm(x)*norm(y))) if norm(x) and norm(y) else None
def numpy_integral(c,d,order):
 result=np.zeros(c.shape[:2])
 for b in range(len(c)):
  for q in range(6):
   for cf,T in zip(c[b,q],d[b,q]):
    for axis in cf:
     dc=np.array([float(axis[k])*math.factorial(k)/math.factorial(k-order) for k in range(order,6)])
     anti=np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(dc,dc))
     result[b,q]+=np.polynomial.polynomial.polyval(float(T),anti)
 return result
def feature(raw,state,boxes,cert_empty):
 out=TrajectoryPostprocessor()(raw,state,num_samples=16)
 p,v,w=segment_samples(out.trajectory,64)
 old,diag=physical_loss(p,v,w,boxes,geometry_bound=[True]*len(p),radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
 q=dynamic_quality_loss(out.trajectory)
 safety=bottleneck_safety_loss(p,w,boxes,geometry_bound=[True]*len(p),certified_empty=cert_empty,temperature=.025,tail_fraction=.1)
 return out,p,v,w,old,diag,q,safety
def metrics(raw,sample,scene,q):
 with torch.no_grad():
  out,p,v,w,old,diag,quality,safety=feature(raw,sample.state[None],[raw.new_tensor([[b['min'],b['max']] for b in scene['obstacles']])],[False])
  par=out.parameters;c=out.trajectory.coeffs[0,q].numpy();d=par.durations[0,q].numpy()
  hard=validate_teacher(c,d,par.head_pva[0,q].numpy(),par.tail_pva[0,q].numpy(),par.inner_pts[0,q].numpy(),scene)
  cl=closest(c,d,scene)
  goal=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),sample.state.numpy()[6:9])
  cross=None
  for cf,T in zip(c,d):
   poly=cf[0].astype(float).copy();poly[0]-=scene['crossing_x_m']
   hits=roots(poly,float(T))
   if hits:cross=evaluate(cf,sorted(hits)[0]).tolist();break
  mode=None
  if cross is not None:
   y,z=cross[1:];mode=('left' if y>=0 else 'right') if abs(y)>=abs(z) else ('up' if z>=0 else 'down')
  values=dict(obstacle=old,acceleration=quality['terms']['acceleration'],jerk=quality['terms']['jerk'],time=quality['terms']['time'],lse=safety['losses']['lse'],tail=safety['losses']['tail'])
 return dict(clearance_m=cl['surface_clearance_m'],closest_time_s=cl['time_s'],closest_box=cl['box_name'],hard_pass=hard['passed'],hard_reasons=hard['reasons'],
   vmax=hard['max_speed_mps'],amax=hard['max_acceleration_mps2'],jmax=hard['max_jerk_mps3'],time_s=hard['total_time_s'],goal=goal,
   crossing_position=cross,heuristic_mode=mode,
   per_query_terms={k:float(v[0,q]) for k,v in values.items()},scene_terms={k:float(v.mean()) for k,v in values.items()})
def run(output):
 started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert output.is_absolute() and output.parent==ROOT/'research_outputs/pg1_v2_r1'
 before=read(output/'audit/PROTECTION_BEFORE.json')
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=str(ROOT),text=True)
 old=read(artifact(before['binding']));old['code']=dict(git_commit=before['HEAD'],files=source_files(),allow_uncommitted_sources=True)
 bbpath=output/'_preparation/BINDINGS_CPU0_ONLY.json';write(bbpath,old);bb=dict(path=str(bbpath),sha256=sha(bbpath))
 ds,geo,init,qualification=qualify_bindings(bb,42)
 write(output/'audit/R1_G2_CPU0_QUALIFICATION.json',qualification)
 write(output/'audit/SOURCE_IDENTITY.json',dict(kind='UNREGISTERED_EXPERIMENTAL_CPU0_ONLY',code=old['code'],geometry_analysis_source=dict(path=str(SOURCE_GEOMETRY),sha256=sha(SOURCE_GEOMETRY)),training_qualified=False))
 dev={s.sample_id:(s,m) for s,m in load_rows(old['development']) if s.sample_id in TARGETS}
 grads=[];cases=[];directions=[];numerics=[];models=[];inputs={str(SOURCE_GEOMETRY):sha(SOURCE_GEOMETRY)}
 direction_protocol=dict(raw_candidate_l2_deltas=[.001,.003],parameter_unit_probe_l2=.0001,parameter_eta_cap=.001,target_scene_raw_max_delta=.001,
  calibration='one fixed finite probe, not loss tuning or optimization; record actual raw radius',focus_window_halfwidth_s=.05,selection='none; all planned directions recorded including harms')
 write(output/'_preparation/DIRECTION_PROTOCOL.json',direction_protocol)
 for arm,name in [('E2','E2paired'),('PG1','PG1complete')]:
  runpath=OLDROOT/('A42_'+name+'_CPU_Pilot32_run001');seal=read(runpath/'RUN_SEAL.json')
  assert all(sha(runpath/n)==v for n,v in seal['files'].items())
  inputs[str(runpath/'RUN_SEAL.json')]=sha(runpath/'RUN_SEAL.json')
  for step in (0,32):
   cp=runpath/'checkpoints'/('checkpoint_step%04d.pt'%step);inputs[str(cp)]=sha(cp)
   model=QueryMincoPlanner();model.load_state_dict(torch.load(str(cp),map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval();basehash=state_sha(model)
   # Training-only calibration probe: one predeclared actual FrozenOrder batch.
   trainids,_=geo.orders[42].batch(0);ts=ds.load(trainids)
   contexts=[('TRAIN_FIRST_BATCH',trainids,ts,None)]+[(sid,[sid],[s],m) for sid,(s,m) in dev.items()]
   for context,ids,samples,meta in contexts:
    assert time.monotonic()-started<900,'CPU0_WALL_CAP'
    model.zero_grad(set_to_none=True)
    state=torch.stack([s.state for s in samples]);depth=torch.stack([s.depth_input for s in samples])
    raw=model(depth_input=depth,state=state)
    if meta is None:
     fields=geo.load(samples);boxes=[f.boxes for f in fields];empty=[f.row['empty_certified'] for f in fields]
    else:
     boxes=[raw.new_tensor([[b['min'],b['max']] for b in meta['scene']['obstacles']])];empty=[_certified_empty(meta['scene'])]
    out,p,v,w,obs,diag,quality,safety=feature(raw,state,boxes,empty)
    costs=[];matches=[];imloss=[]
    mask=torch.zeros(raw.shape[:2],dtype=torch.bool)
    for b,sample in enumerate(samples):
     cost,_=cost_matrix(out,sample,b);value,qq,tt=match(cost,'B_set')
     costs.append(cost);imloss.append(value);mask[b,qq]=True;matches.append(dict(queries=qq.tolist(),teachers=tt.tolist()))
    ana=analytic_samples(out.trajectory,64);limit,_=dynamic_limit_loss(ana['v'],ana['a'],ana['jerk'],ana['weights'])
    old_obstacle=reduce_set(obs,mask,1.,1.)
    terms=dict(imitation=torch.stack(imloss).mean(),obstacle=old_obstacle,limit=limit.mean(),**{k:v.mean() for k,v in quality['terms'].items()},lse=safety['losses']['lse'].mean(),tail=safety['losses']['tail'].mean())
    per=dict(obstacle=obs,limit=limit,**quality['terms'],lse=safety['losses']['lse'],tail=safety['losses']['tail'])
    # Repeat historical calls bit-for-bit, independent of new prototypes.
    again,_=physical_loss(p,v,w,boxes,geometry_bound=[True]*len(samples),radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
    assert torch.equal(obs,again)
    assert torch.equal(limit,dynamic_limit_loss(ana['v'],ana['a'],ana['jerk'],ana['weights'])[0])
    if meta is None:
     legacy=old_forward(model,samples,geo,'query_minco_pg1_e2_paired_v1')
     assert torch.equal(legacy['raw'],raw) and torch.equal(legacy['terms']['obstacle'],old_obstacle) and torch.equal(legacy['terms']['dynamic'],limit.mean())
    c=out.trajectory.coeffs;d=out.parameters.durations;par=list(model.parameters())
    pg={};rg={};records={}
    arrays=dict(raw=raw.detach().numpy(),state=state.numpy(),coeffs=c.detach().numpy(),durations=d.detach().numpy(),matched=mask.numpy())
    for term,value in terms.items():
     gg=torch.autograd.grad(value,[raw,c,d]+par,retain_graph=True,allow_unused=True)
     gr=[torch.zeros_like(t) if g is None else g for g,t in zip(gg,[raw,c,d]+par)]
     assert all(torch.isfinite(g).all() for g in gr),('NONFINITE',term)
     assert not torch.count_nonzero(gr[0][...,19:21]),('RAW19_21',term)
     pv=torch.cat([g.detach().flatten() for g in gr[3:]]).numpy();pg[term]=pv;rg[term]=gr[0].detach().numpy()
     records[term]=dict(value=float(value.detach()),parameter_norm=norm(pv),raw_norm=norm(rg[term]),coeff_norm=norm(gr[1].detach().numpy()),duration_norm=norm(gr[2].detach().numpy()),
       raw_time15_19_norms=[norm(rg[term][...,j]) for j in range(15,19)],
       matched_raw_norm=norm(rg[term][mask.numpy()]),unmatched_raw_norm=norm(rg[term][~mask.numpy()]))
     arrays['raw_grad_'+term]=rg[term];arrays['coeff_grad_'+term]=gr[1].detach().numpy();arrays['duration_grad_'+term]=gr[2].detach().numpy()
    for term in records:
     records[term].update(cosine_imitation=cosine(pg[term],pg['imitation']),cosine_obstacle=cosine(pg[term],pg['obstacle']),parameter_ratio_to_imitation=norm(pg[term])/norm(pg['imitation']) if norm(pg['imitation']) else None)
    # Independent NumPy polynomial squared-energy integrals.
    errors={}
    for term,order in [('acceleration',2),('jerk',3)]:
     independent=numpy_integral(c.detach().numpy(),d.detach().numpy(),order)
     actual=quality['integrals_si'][term].detach().numpy()
     assert np.allclose(actual,independent,rtol=2e-4,atol=1e-4),(term,np.max(abs(actual-independent)))
     errors[term]=float(np.max(abs(actual-independent)))
    # Independent NumPy weighted LSE and fractional weighted-tail reconstruction.
    point=safety['point_risk'].detach().numpy();alpha=safety['normalized_time_mass'].detach().numpy()
    z=point/.025;mx=z.max(-1,keepdims=True);lse=.025*(mx[...,0]+np.log((alpha*np.exp(z-mx)).sum(-1)))
    index=np.argsort(-point,axis=-1,kind='stable');val=np.take_along_axis(point,index,-1);mass=np.take_along_axis(alpha,index,-1)
    taken=np.minimum(mass,np.maximum(.1-(np.cumsum(mass,-1)-mass),0));tail=(taken*val).sum(-1)/.1
    assert np.allclose(lse,safety['losses']['lse'].detach().numpy(),atol=1e-6,rtol=2e-5)
    assert np.allclose(tail,safety['losses']['tail'].detach().numpy(),atol=1e-6,rtol=2e-5)
    numerics.append(dict(arm=arm,step=step,context=context,quality_max_abs_SI_error=errors,lse_numpy=True,tail_numpy=True,original_losses_bitwise_equal=True))
    grads.append(dict(arm=arm,step=step,context=context,ids=ids,matches=matches,matched_count=int(mask.sum()),unmatched_count=int((~mask).sum()),terms=records))
    for k,vv in per.items():arrays['per_'+k]=vv.detach().numpy()
    np.savez_compressed(output/'runs'/(arm+'_step%04d_'%step+context+'.npz'),**arrays)
    if meta is not None:
     sample=samples[0];scene=meta['scene']
     globaltimes=(d[...,None]*torch.linspace(0,1,64)+torch.cat([d[...,:1]*0,d.cumsum(-1)[...,:-1]],-1)[...,None]).flatten(2,3).detach().numpy()[0]
     focus={}
     for name,value in [('lse',safety['losses']['lse']),('tail',safety['losses']['tail'])]:
      focus[name]=torch.autograd.grad(value.sum(),safety['point_risk'],retain_graph=True)[0].detach().numpy()[0]
     focus['obstacle']=(safety['normalized_time_mass'][0]*diag['risk_weights'][0]).detach().numpy()
     for q in TARGETS[context]:
      base=metrics(raw.detach(),sample,scene,q)
      sensitivity={}
      for temp in (.01,.05):
       value=bottleneck_safety_loss(p,w,boxes,geometry_bound=[True],certified_empty=empty,temperature=temp,tail_fraction=.1)
       sensitivity['lse_tau_'+str(temp)]=float(value['losses']['lse'][0,q].detach())
      for frac in (.05,.2):
       value=bottleneck_safety_loss(p,w,boxes,geometry_bound=[True],certified_empty=empty,temperature=.025,tail_fraction=frac)
       sensitivity['tail_fraction_'+str(frac)]=float(value['losses']['tail'][0,q].detach())
      case=dict(arm=arm,step=step,sample_id=context,query=q,matched=bool(mask[0,q]),teacher_index=next((t for qq,t in zip(matches[0]['queries'],matches[0]['teachers']) if qq==q),None),baseline=base,
        quality={k:float(v[0,q].detach()) for k,v in quality['terms'].items()},safety_sensitivity=sensitivity,
        bottleneck_gradient_mass_in_plusminus_50ms={name:float(weight[q][abs(globaltimes[q]-base['closest_time_s'])<=.05].sum()) for name,weight in focus.items()})
      cases.append(case)
      for term in DIRECTIONS:
       # Candidate-raw derivative, not a fictitious network update.
       target=per[term][0,q]
       g=torch.autograd.grad(target,raw,retain_graph=True)[0].detach()[0,q];gn=float(g.norm())
       if gn==0:
        directions.append(dict(arm=arm,step=step,sample_id=context,query=q,term=term,mode='RAW_CANDIDATE',status='ZERO_GRADIENT'));continue
       for delta in (.001,.003):
        for sign in (-1,1):
         rr=raw.detach().clone();rr[0,q]+=sign*delta*g/gn
         try:after=metrics(rr,sample,scene,q);status='MEASURED'
         except ValueError as e:after=None;status='DECODE_REJECTED:'+str(e)
         directions.append(dict(arm=arm,step=step,sample_id=context,query=q,term=term,mode='RAW_CANDIDATE',sign=sign,l2_delta=delta,status=status,
           base=base,after=after))
    # Scene-mean parameter direction, evaluated functionally, for all three target queries.
    if meta is not None:
     named=list(model.named_parameters());buffers=dict(model.named_buffers())
     for term in DIRECTIONS:
      grad=pg[term];gn=norm(grad)
      if not gn:continue
      offset=0;unit={}
      for name,param in named:
       count=param.numel();unit[name]=torch.from_numpy(grad[offset:offset+count].reshape(tuple(param.shape))).to(param)/gn;offset+=count
      def changed(eta):
       mapping={name:param.detach()-eta*unit[name] for name,param in named};mapping.update(buffers)
       with torch.no_grad():return functional_call(model,mapping,(),dict(depth_input=depth,state=state))
      probe=changed(.0001);speed=float((probe-raw.detach()).abs().max())/.0001
      eta=min(.001,.001/speed) if speed else .001
      changedraw=changed(eta);actualraw=float((changedraw-raw.detach()).abs().max())
      assert actualraw<=.00105,('FUNCTIONAL_RADIUS',actualraw)
      assert state_sha(model)==basehash,'FUNCTIONAL_MODEL_MUTATION'
      for q in TARGETS[context]:
       base=next(z['baseline'] for z in cases if (z['arm'],z['step'],z['sample_id'],z['query'])==(arm,step,context,q))
       try:after=metrics(changedraw,samples[0],meta['scene'],q);status='MEASURED'
       except ValueError as e:after=None;status='DECODE_REJECTED:'+str(e)
       directions.append(dict(arm=arm,step=step,sample_id=context,query=q,term=term,mode='FUNCTIONAL_SCENE_PARAMETER',sign=-1,parameter_l2_delta=eta,actual_scene_raw_max_delta=actualraw,status=status,base=base,after=after))
    assert state_sha(model)==basehash,'MODEL_WEIGHT_CHANGED'
    print('CPU0',arm,step,context,flush=True)
   models.append(dict(arm=arm,step=step,weights_before=basehash,weights_after=state_sha(model)))
 write(output/'runs/GRADIENTS.json',dict(records=grads))
 write(output/'runs/CASES.json',dict(records=cases))
 write(output/'runs/DIRECTIONS.json',dict(records=directions))
 write(output/'audit/NUMERICS.json',dict(records=numerics))
 write(output/'audit/INPUTS.json',dict(files=inputs))
 write(output/'audit/CPU0_RESULT.json',dict(status='COMPLETE',models=models,contexts=len(grads),candidate_cases=len(cases),direction_records=len(directions),optimizer_created=False,optimizer_steps=0,GPU_jobs=0,wall_s=time.monotonic()-started))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:run(a.output)
 except BaseException:
  write(a.output/'audit/CPU0_FAILURE.json',dict(error=traceback.format_exc(),optimizer_steps=0));raise
