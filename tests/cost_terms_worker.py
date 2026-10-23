"""Read saved synthetic curves; decompose six costs without model inference."""
import sys,json
from pathlib import Path
import numpy as np,torch
N=Path(__file__).resolve().parents[1];sys.path.insert(0,str(N))
from query_minco.trajectory import MincoTrajectory,TrajectorySampler
from query_minco.training.teacher.sampling import teacher_evaluate
O=N/'research_outputs/migration_v1';out=Path(sys.argv[1])
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
allterms=[];counts=[]
for label in ['training_old002','training_new001']:
 with np.load(O/'runs'/label/'training_arrays.npz',allow_pickle=False) as z:a={k:torch.from_numpy(z[k].copy()) for k in z.files}
 fields={}
 for b in range(2):
  tr=MincoTrajectory(a['coefficients'][b:b+1],a['durations'][b:b+1],torch.ones(1,6,dtype=torch.bool),{})
  T=tr.durations[0].sum(-1);sampler=TrajectorySampler();pred_end=sampler(tr,times=T[None,:,None])
  tc=a['teacher_coeff_'+str(b)];td=a['teacher_durations_'+str(b)]
  rows=[[] for _ in range(6)];c4=[];extra=[]
  for m in range(len(td)):
   total=td[m].sum();times=torch.minimum(T,total)[...,None]*torch.linspace(0,1,32)
   pred=sampler(tr,times=times[None])
   tp=teacher_evaluate(tc[m],td[m],times,0);tv=teacher_evaluate(tc[m],td[m],times,1)
   ep=teacher_evaluate(tc[m],td[m],total.expand(6,1),0)[:,0]
   ev=teacher_evaluate(tc[m],td[m],total.expand(6,1),1)[:,0]
   ea=teacher_evaluate(tc[m],td[m],total.expand(6,1),2)[:,0]
   p=(pred.p[0]-tp).square().mean((-2,-1));v=(pred.v[0]-tv).square().mean((-2,-1))
   end=(pred_end.p[0,:,0]-ep).square().mean(-1);duration=(T-total).square()
   ve=.1*(pred_end.v[0,:,0]-ev).square().mean(-1);ac=.01*(pred_end.a[0,:,0]-ea).square().mean(-1)
   for rs,value in zip(rows,[p,.1*v,.5*end,.2*duration,ve,ac]):rs.append(value)
   c4.append(p+.1*v+.5*end+.2*duration);extra.append(ve+ac)
  reconstructed=torch.stack(c4,1)+torch.stack(extra,1)
  assert reconstructed.numpy().tobytes()==a['cost_'+str(b)].numpy().tobytes()
  for k,rs in enumerate(rows):fields['task_%d_term_%d'%(b,k)]=torch.stack(rs,1).numpy()
 with (out/(label+'_terms.npz')).open('xb') as f:np.savez(f,**fields)
 allterms.append(fields)
assert all(allterms[0][k].tobytes()==allterms[1][k].tobytes() for k in allterms[0])
with (out/'RESULT.json').open('x') as f:json.dump(dict(passed=True,term_matrices=12,
 old_new_bitwise_equal=True,actual_cost_reconstruction_bitwise_equal=True,model_forward_calls=0),f,indent=2)
print('SIX_TERM_DIAGNOSTICS_PASS')
