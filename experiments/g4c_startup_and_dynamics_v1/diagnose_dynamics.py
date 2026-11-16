"""Frozen Step0/64/128 dynamics audit;TRAIN autograd only,DEV forward-only."""
from pathlib import Path
import sys,json,math
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from query_minco.training.peak_dynamics_proxy_v1.proxy import peak_proxy,diagnostic_objective
from query_minco.training.k1_forest_v1.trainer import verify,model_for
from query_minco.training.k1_forest_v1.data import load
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.physics_only_v1_2_gate1.objective import feasibility_math
from query_minco.training.pg1.losses import analytic_samples
from query_minco.evaluation.validation import norm_max,evaluate,roots
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1';A=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def continuous(c,dt):
 maxima=[];durations=[]
 for order,limit in [(1,3.),(2,8.),(3,100.)]:
  maxima.append(max(norm_max(poly,t,order) for poly,t in zip(c,dt)));length=0.
  for poly,T in zip(c,dt):
   q=np.stack([poly[:,i]*math.factorial(i)/math.factorial(i-order) for i in range(order,6)],1);squared=np.zeros(1)
   for axis in q:squared=np.polynomial.polynomial.polyadd(squared,np.polynomial.polynomial.polymul(axis,axis))
   squared[0]-=limit**2;cut=sorted(set([0.,float(T)]+roots(squared,float(T))))
   length+=sum(b-a for a,b in zip(cut[:-1],cut[1:]) if np.linalg.norm(evaluate(poly,(a+b)/2,order))>limit+1e-10)
  durations.append(length)
 ratios=np.array(maxima)/[3.,8.,100.]
 return dict(maxima=maxima,exceed_duration_s=durations,continuous_violation=float(np.maximum(ratios-1,0).dot(np.maximum(ratios-1,0))))
def trajectory(raw,batch,objective):
 p=objective.objective.decoder(raw,batch.state)
 return objective.objective.adapter(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
def losses(tr,batch,spec):
 ana=analytic_samples(tr,64);old=feasibility_math(ana,[g.domain for g in batch.geometry],spec);peak=peak_proxy(tr.coeffs,tr.durations)
 return old,peak
def cosine(a,b):
 n=float(a.norm()*b.norm());return float((a*b).sum())/n if n>0 else None
def main():
 torch.set_num_threads(1);spec=verify(A/'K1_FOREST_TRAINING_RUNSPEC.json')
 write(O/'QUALIFICATION_RULES.json',dict(primary_raw_step=1e-3,secondary_roundoff_step=1e-4,
  reliable='at least90% of violated TRAIN cases strictly decrease continuous violation by1e-7 at primary step;no increase>2e-6',
  engineering='only after zero-update/unit contracts pass;first2 violated TRAIN at Step128,max2 attempts,noDEVgradient',
  tolerances='FP32 raw directionalFD abs5e-5+3percent relative;coefficientFP64 FD separate'))
 records=[];arrays={};checks=[]
 with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
  for step in [0,64,128]:
   model=model_for(spec,20261104).eval();path=A/'formal/A_reasonable_seed20261104'/('step%03d.pt'%step);side=json.loads(Path(str(path)+'.json').read_text());assert sha(path)==side['file_sha256']
   model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True);before=state_sha(model);objective=make_objective()
   for row in spec['train']+spec['dev']:
    batch=load(row,row['split']);raw=model(batch.depth,batch.state)
    if row['split']=='dev':
     with torch.no_grad():raw=model(batch.depth,batch.state);out=objective(raw,batch);newloss,peak=diagnostic_objective(out)
    else:out=objective(raw,batch);newloss,peak=diagnostic_objective(out)
    tr=out.trajectory;c=tr.coeffs[0,0].detach().double().numpy();dt=tr.durations[0,0].detach().double().numpy();hard=continuous(c,dt)
    peak32=peak_proxy(tr.coeffs.detach(),tr.durations,accumulation=torch.float32)
    record=dict(step=step,sample_id=row['sample_id'],split=row['split'],kind=row['kind'],old_loss=float(out.loss.detach()),diagnostic_loss=float(newloss.detach()),
     F_parts={k:float(out.metrics[k].detach().mean()) for k in ['F','F_dyn','F_domain','F_speed','F_acceleration','F_jerk']},
     sample_peaks=[float(out.metrics[k].detach().mean()) for k in ['speed_sample_peak','acceleration_sample_peak','jerk_sample_peak']],
     continuous=hard,upper_ratios={k:float(v.detach()) for k,v in peak['upper_ratios'].items()},
     error_bounds_ratio={k:float(v.detach()) for k,v in peak['error_bounds_ratio'].items()},
     peak_terms={k:float(v.detach()) for k,v in peak['terms'].items()},
     FP32_vs_FP64_upper_ratio_difference={k:float((peak32['upper_ratios'][k].double()-v.detach()).abs()) for k,v in peak['upper_ratios'].items()},
     gradients='NOT_RUN_DEV_FORWARD_ONLY',FP64_MINCO='NOT_RUN_UNSUPPORTED_BY_ORIGINAL_ADAPTER')
    key='%03d__%s'%(step,row['sample_id']);arrays[key+'_coeff']=c;arrays[key+'_durations']=dt;arrays[key+'_raw21']=raw.detach().numpy()
    if row['split']=='train':
     names={'F_speed':out.metrics['F_speed'].mean(),'F_acceleration':out.metrics['F_acceleration'].mean(),'F_jerk':out.metrics['F_jerk'].mean(),
      'C_viol':out.metrics['C_viol'].mean(),'G_task':out.metrics['G_task_train'].mean(),'shape_weighted':.1*out.metrics['C_shape'].mean(),
      'progress_weighted':.025*out.metrics['P'].mean(),'smooth_weighted':.1*out.metrics['S'].mean(),'domain':out.metrics['F_domain'].mean(),
      'old_L0':out.loss,'peak':peak['F_peak'].mean(),'diagnostic_L0':newloss}
     grads={name:torch.autograd.grad(value,raw,retain_graph=True)[0].detach() for name,value in names.items()}
     if any(not torch.isfinite(g).all() or g[...,15:].count_nonzero() for g in grads.values()):raise ValueError('RAW_GRADIENT_CONTRACT')
     fd={};counter=[]
     for name in ['F_speed','F_acceleration','F_jerk','peak']:
      g=grads[name];n=float(g.norm());direction=g/(n if n else 1.)
      if n==0:
       fd[name]=dict(status='ZERO_LOCAL_GRADIENT',norm=0.);continue
      values=[]
      with torch.no_grad():
       for sign in [-1,1]:
        tt=trajectory(raw.detach()+sign*1e-3*direction,batch,objective);of,pf=losses(tt,batch,objective.objective.spec)
        values.append(float(pf['F_peak'].mean()) if name=='peak' else float(of[name].mean()))
      derivative=(values[1]-values[0])/.002;ok=abs(derivative-n)<=5e-5+.03*max(abs(derivative),n)
      fd[name]=dict(status='PASS' if ok else 'FD_MISMATCH',autograd_directional=n,finite_difference=derivative,values=values)
     norm=float(grads['peak'].norm())
     if norm>0:
      with torch.no_grad():
       for eps in [1e-4,1e-3]:
        tt=trajectory(raw.detach()-eps*grads['peak']/norm,batch,objective);cc=continuous(tt.coeffs[0,0].double().numpy(),dt)
        counter.append(dict(epsilon=eps,continuous=cc,delta=cc['continuous_violation']-hard['continuous_violation']))
     record['gradients']=dict(norms={k:float(v.norm()) for k,v in grads.items()},vectors_raw15={k:v[0,0,:15].tolist() for k,v in grads.items()},
      cosines_vs_peak={k:cosine(v,grads['peak']) for k,v in grads.items()},directional_fd=fd,negative_peak_gradient_counterfactual=counter)
    records.append(record);print('DYNAMICS_DIAGNOSED',step,row['sample_id'],hard['maxima'],flush=True)
   assert state_sha(model)==before and all(p.grad is None for p in model.parameters());checks.append(dict(step=step,model_sha_before=before,model_sha_after=state_sha(model),file_sha256=sha(path)))
   write(O/('DYNAMICS_STEP%03d.json'%step),dict(records=[x for x in records if x['step']==step],model_identity=checks[-1],network_updates=0))
 np.savez_compressed(O/'DYNAMICS_FROZEN_ARRAYS.npz',**arrays)
 violated=[x for x in records if x['split']=='train' and x['continuous']['continuous_violation']>1e-6]
 primary=[next((y for y in x['gradients']['negative_peak_gradient_counterfactual'] if y['epsilon']==1e-3),None) for x in violated]
 deltas=[x['delta'] for x in primary if x is not None];strict=sum(d<-1e-7 for d in deltas);bad=sum(d>2e-6 for d in deltas)
 signal=bool(deltas) and len(deltas)==len(violated) and strict/len(deltas)>=.9 and bad==0
 fd_checks=[v for x in records if x['split']=='train' for v in x['gradients']['directional_fd'].values() if v['status']!='ZERO_LOCAL_GRADIENT']
 fd_bad=[x for x in fd_checks if x['status']!='PASS']
 write(O/'DYNAMICS_ZERO_UPDATE_SUMMARY.json',dict(status='PASS' if signal and not fd_bad else 'NO_GO',violated_train_cases=len(violated),strict_decreases=strict,clear_increases=bad,
  directional_signal=signal,raw_directional_fd_failures=len(fd_bad),fd_failures=fd_bad,network_updates=0,dev_gradient='NOT_RUN',
  source_precision='network/MINCO FP32;peak defaultFP64,independent continuous normmax on frozen coefficientsFP64',
  model_identities=checks,all_records=records))
 print('ZERO_UPDATE_SIGNAL',signal,'FD_BAD',len(fd_bad))
if __name__=='__main__':main()
