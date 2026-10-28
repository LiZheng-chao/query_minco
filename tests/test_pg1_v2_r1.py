"""Math/CPU0 only: no optimizer object, no parameter updates."""
import copy,math,unittest
from types import SimpleNamespace
import numpy as np
import torch
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss as quality,bottleneck_safety_loss as safety,weighted_lse,weighted_upper_tail
from query_minco.training.pg1.losses import analytic_samples,dynamic_limit_loss
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss,box_sdf
from query_minco.trajectory import TrajectoryPostprocessor

def tr(c=None,d=None):
 if c is None:
  torch.manual_seed(21);c=torch.randn(1,6,3,3,6,dtype=torch.float64)*.05
 if d is None:d=torch.tensor([.4,.5,.6],dtype=c.dtype).expand(1,6,3).clone()
 return SimpleNamespace(coeffs=c,durations=d,solve_valid=torch.ones(c.shape[:2],dtype=torch.bool))
def raw_fixture():
 r=torch.zeros(1,6,21)
 for piece,x in enumerate((.8,1.6,2.4)):r[:,:,piece*3]=math.atanh(2*x/10-1)
 r[:,:,15]=math.log((1.6-.8)/(2.2-1.6))
 r[:,:,16:19]=torch.tensor([-.1,0,.1]);r[:,:,19:21]=.7
 state=torch.tensor([[.8,0,0,0,0,0,2.4,0,0,0,0,-1.]])
 return r,state
def np_integral(c,d,order):
 result=np.zeros(c.shape[:2])
 for b in range(len(c)):
  for q in range(6):
   for cf,T in zip(c[b,q],d[b,q]):
    for axis in cf:
     dc=np.array([axis[k]*math.factorial(k)/math.factorial(k-order) for k in range(order,6)])
     p=np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(dc,dc))
     result[b,q]+=np.polynomial.polynomial.polyval(T,p)
 return result
class Quality(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_known_units_and_exact_polynomial_integrals(self):
  c=torch.zeros(1,6,3,3,6,dtype=torch.float64);c[:,:,:,0,2]=.5;c[:,:,:,1,3]=1/6
  t=tr(c);z=quality(t)
  expected_a=sum(float(T+T**3/3) for T in t.durations[0,0])
  expected_j=float(t.durations[0,0].sum())
  self.assertTrue(torch.allclose(z['integrals_si']['acceleration'],torch.full((1,6),expected_a,dtype=c.dtype),atol=1e-12,rtol=1e-12))
  self.assertTrue(torch.allclose(z['integrals_si']['jerk'],torch.full((1,6),expected_j,dtype=c.dtype),atol=1e-12,rtol=1e-12))
  for k,o in [('acceleration',2),('jerk',3)]:
   self.assertTrue(np.allclose(z['integrals_si'][k].numpy(),np_integral(c.numpy(),t.durations.numpy(),o),rtol=1e-12,atol=1e-12))
 def test_independent_numpy_random_and_ref_scaling(self):
  t=tr();a=quality(t);b=quality(t,a_ref=16,j_ref=200,time_ref=4.4)
  for k,o in [('acceleration',2),('jerk',3)]:
   self.assertTrue(np.allclose(a['integrals_si'][k].numpy(),np_integral(t.coeffs.numpy(),t.durations.numpy(),o),rtol=1e-11,atol=1e-12))
   self.assertTrue(torch.allclose(b['terms'][k]*8,a['terms'][k]))
  self.assertTrue(torch.allclose(b['terms']['time']*2,a['terms']['time']))
 def test_time_reparameterization_physical_scaling(self):
  a=tr(d=torch.tensor([.3,.4,.5],dtype=torch.float64).expand(1,6,3).clone());scale=1.2
  b=tr(a.coeffs/scale**torch.arange(6,dtype=a.coeffs.dtype),a.durations*scale)
  x=quality(a);y=quality(b)
  self.assertTrue(torch.allclose(y['terms']['acceleration'],x['terms']['acceleration']/scale**3,atol=1e-12,rtol=1e-11))
  self.assertTrue(torch.allclose(y['terms']['jerk'],x['terms']['jerk']/scale**5,atol=1e-12,rtol=1e-11))
  self.assertTrue(torch.allclose(y['terms']['time'],x['terms']['time']*scale))
 def test_leaf_coeff_and_duration_gradcheck(self):
  t=tr();c=t.coeffs.requires_grad_();d=t.durations.requires_grad_()
  def f(c,d):
   z=quality(tr(c,d))['terms'];return sum(z.values())
  self.assertTrue(torch.autograd.gradcheck(f,(c,d),eps=1e-6,atol=1e-6,rtol=1e-4,fast_mode=True))
 def test_enabled_ablations_no_hidden_final_weights(self):
  t=tr();allterms=quality(t)['terms']
  for enabled in [(),('acceleration','time'),('jerk','time'),('acceleration','jerk','time')]:
   z=quality(t,enabled=enabled)['terms']
   for k,v in z.items():self.assertTrue(torch.equal(v,allterms[k] if k in enabled else allterms[k]*0))
 def test_invalid_duration_solve_reference(self):
  for value in (-.1,0.,float('nan'),1.):
   t=tr();t.durations.fill_(value)
   with self.assertRaises(ValueError):quality(t)
  t=tr();t.solve_valid.fill_(False)
  with self.assertRaises(ValueError):quality(t)
  for kw in [dict(a_ref=0),dict(j_ref=float('inf')),dict(time_ref=-1),dict(enabled=['bad']),dict(enabled=['time','time'])]:
   with self.assertRaises(ValueError):quality(tr(),**kw)
 def test_raw_time_decoder_minco_gradient_and_unsupervised_rows(self):
  r,s=raw_fixture();r.requires_grad_();out=TrajectoryPostprocessor()(r,s,num_samples=8)
  z=quality(out.trajectory)['terms']
  for name,v in z.items():
   g=torch.autograd.grad(v.mean(),r,retain_graph=True)[0]
   self.assertTrue(torch.isfinite(g).all());self.assertFalse(torch.count_nonzero(g[...,19:21]))
   self.assertGreater(abs(float(g[0,0,15])),1e-8)
   for index in range(15,19):
    h=.003;plus=r.detach().clone();minus=r.detach().clone();plus[0,0,index]+=h;minus[0,0,index]-=h
    fp=quality(TrajectoryPostprocessor()(plus,s,num_samples=8).trajectory)['terms'][name].mean()
    fm=quality(TrajectoryPostprocessor()(minus,s,num_samples=8).trajectory)['terms'][name].mean()
    fd=float((fp-fm)/(2*h));ad=float(g[0,0,index])
    self.assertLess(abs(fd-ad),.015*max(abs(fd),abs(ad),1e-3),(name,index,ad,fd))
   if name=='time':self.assertLess(float(g[...,16:19].abs().max()),1e-6)
   else:self.assertGreater(float(g[...,16:19].norm()),1e-8)

class Bottleneck(unittest.TestCase):
 def fixture(self,dtype=torch.float64):
  p=torch.zeros(1,6,4,3,dtype=dtype);p[:,:,:,0]=torch.tensor([.2,.3,.45,.7],dtype=dtype)
  w=torch.tensor([.11,.19,.27,.43],dtype=dtype).expand(1,6,4).clone()
  b=torch.tensor([[[-.1,-.1,-.1],[.1,.1,.1]]],dtype=dtype)
  return p,w,b
 def test_physical_time_tail_and_stable_lse_numpy(self):
  v=torch.tensor([.0,.1,.2,.3],dtype=torch.float64).expand(1,6,4);w=torch.tensor([.11,.19,.27,.43],dtype=torch.float64).expand(1,6,4)
  tau=.025;expected=tau*np.log(np.sum(w.numpy()*np.exp(v.numpy()/tau),axis=-1))
  self.assertTrue(np.allclose(weighted_lse(v,w,tau).numpy(),expected,atol=1e-12))
  expected_tail=(.43*.3+.07*.2)/.5
  self.assertTrue(torch.allclose(weighted_upper_tail(v,w,.5),torch.full((1,6),expected_tail,dtype=v.dtype),atol=1e-12))
  self.assertFalse(np.isclose(expected_tail,(.3+.2)/2))
  self.assertTrue(torch.allclose(weighted_upper_tail(v,w,1.),(v*w).sum(-1)))
 def test_mass_split_invariance_and_constants(self):
  v=torch.tensor([0.,.1,.2,.3],dtype=torch.float64).expand(1,6,4);w=torch.tensor([.11,.19,.27,.43],dtype=v.dtype).expand_as(v)
  for fn,k in [(weighted_lse,.025),(weighted_upper_tail,.5)]:
   self.assertTrue(torch.allclose(fn(v,w,k),fn(v.repeat_interleave(2,-1),w.repeat_interleave(2,-1)/2,k),atol=1e-12))
   self.assertTrue(torch.allclose(fn(torch.ones_like(v)*.7,w,k),torch.ones(1,6,dtype=v.dtype)*.7,atol=1e-12))
 def test_safety_positions_weights_f64_gradcheck(self):
  p,w,b=self.fixture();p.requires_grad_();w.requires_grad_()
  def f(p,w):return safety(p,w,[b],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.5)['losses']['lse']+safety(p,w,[b],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.5)['losses']['tail']
  self.assertTrue(torch.autograd.gradcheck(f,(p,w),eps=1e-6,atol=1e-6,rtol=1e-4,fast_mode=True))
 def test_empty_certified_mixed_and_no_fake_clearance(self):
  p,w,b=self.fixture();p=p.repeat(2,1,1,1).requires_grad_();w=w.repeat(2,1,1)
  z=safety(p,w,[b.new_empty(0,2,3),b],geometry_bound=[True,True],certified_empty=[True,False],temperature=.025,tail_fraction=.1)
  self.assertIsNone(z['surface_clearance'][0]);self.assertFalse(z['certificate'])
  for v in z['losses'].values():
   self.assertTrue(torch.equal(v[0],torch.zeros_like(v[0])))
   g=torch.autograd.grad(v.sum(),p,retain_graph=True)[0];self.assertTrue(torch.isfinite(g).all());self.assertFalse(torch.count_nonzero(g[0]))
 def test_reject_unknown_empty_invalid_geometry_and_threshold(self):
  p,w,b=self.fixture()
  for field,cert,bound in [(b.new_empty(0,2,3),False,True),(b,True,True),(b,False,False),(b.flip(1),False,True),(b.clone().requires_grad_(),False,True)]:
   with self.assertRaises(ValueError):safety(p,w,[field],geometry_bound=[bound],certified_empty=[cert],temperature=.025,tail_fraction=.1)
  for kw in [dict(temperature=0),dict(tail_fraction=0),dict(tail_fraction=1.1),dict(radius=.1),dict(margin=.01)]:
   args=dict(temperature=.025,tail_fraction=.1);args.update(kw)
   with self.assertRaises(ValueError):safety(p,w,[b],geometry_bound=[True],certified_empty=[False],**args)
  with self.assertRaises(ValueError):safety(p,w,None,geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)
 def test_zero_mass_negative_mass_nonfinite(self):
  p,w,b=self.fixture()
  for ww in [w*0,-w,w*float('nan')]:
   with self.assertRaises(ValueError):safety(p,ww,[b],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)
 def test_zero_weight_samples_and_large_risk_no_nan(self):
  p,w,b=self.fixture(torch.float32);p[:,:,:,0]=.05;p.requires_grad_();w[:,:,0]=0
  z=safety(p,w,[b],geometry_bound=[True],certified_empty=[False],temperature=.00001,tail_fraction=.1)
  for v in z['losses'].values():
   self.assertTrue(torch.isfinite(v).all());g=torch.autograd.grad(v.sum(),p,retain_graph=True)[0];self.assertTrue(torch.isfinite(g).all())


 def test_safety_raw_decoder_minco_local_fd_convergence(self):
  r,state=raw_fixture()
  for piece,y in enumerate((.05,.12,.02)):r[:,:,piece*3+1]=math.atanh(y/4.)
  r.requires_grad_()
  field=r.new_tensor([[[-2.,.3,-2.],[4.,.6,2.]]])
  def values(rr):
   out=TrajectoryPostprocessor()(rr,state,num_samples=8);p,v,w=segment_samples(out.trajectory,64)
   return safety(p,w,[field],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)['losses']
  z=values(r)
  for name in ('lse','tail'):
   g=torch.autograd.grad(z[name].mean(),r,retain_graph=True)[0]
   self.assertFalse(torch.count_nonzero(g[...,19:21]))
   for index in (1,4,15,16,17,18):
    errors=[]
    for h in (.003,.001,.0003):
     p=r.detach().clone();m=r.detach().clone();p[0,0,index]+=h;m[0,0,index]-=h
     fd=float((values(p)[name].mean()-values(m)[name].mean())/(2*h));ad=float(g[0,0,index])
     errors.append(abs(fd-ad)/max(abs(fd),abs(ad),1e-3))
    # Local derivative, not a coarse secant across multiple tail-membership changes.
    # The coarse .003 failure is retained in the experiment audit, not relabelled PASS.
    self.assertLess(errors[-1],.015,(name,index,errors))
    if name=='tail' and index==1:self.assertLess(errors[-1],errors[0])

 def test_safety_coeff_duration_gradcheck(self):
  c=torch.zeros(1,6,3,3,6,dtype=torch.float64);c[:,:,:,0,0]=.25;c[:,:,:,0,1]=.07
  c[:,:,:,1,0]=torch.tensor([.01,.02,.03],dtype=c.dtype)
  c.requires_grad_();d=torch.tensor([.4,.5,.6],dtype=c.dtype).expand(1,6,3).clone().requires_grad_()
  field=c.new_tensor([[[-.2,-2.,-2.],[.1,2.,2.]]])
  def f(c,d):
   p,v,w=segment_samples(tr(c,d),16)
   z=safety(p,w,[field],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)
   return z['losses']['lse']+z['losses']['tail']
  self.assertTrue(torch.autograd.gradcheck(f,(c,d),eps=1e-6,atol=1e-6,rtol=1e-4,fast_mode=True))

 def test_original_losses_bitwise_unchanged_and_raw19_zero(self):
  r,s=raw_fixture();r.requires_grad_();out=TrajectoryPostprocessor()(r,s,num_samples=8)
  p,v,w=segment_samples(out.trajectory,64);field=p.new_tensor([[[1.,.3,-.2],[1.2,.5,.2]]])
  old,_=physical_loss(p,v,w,[field],geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
  ana=analytic_samples(out.trajectory,64);limit,terms=dynamic_limit_loss(ana['v'],ana['a'],ana['jerk'],ana['weights'])
  baseline=old.detach().clone();limitbefore=limit.detach().clone()
  z=safety(p,w,[field],geometry_bound=[True],certified_empty=[False],temperature=.025,tail_fraction=.1)
  q=quality(out.trajectory)
  for value in list(z['losses'].values())+list(q['terms'].values()):
   g=torch.autograd.grad(value.mean(),r,retain_graph=True)[0];self.assertTrue(torch.isfinite(g).all());self.assertFalse(torch.count_nonzero(g[...,19:21]))
  again,_=physical_loss(p,v,w,[field],geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
  self.assertTrue(torch.equal(baseline,again));self.assertTrue(torch.equal(limitbefore,dynamic_limit_loss(ana['v'],ana['a'],ana['jerk'],ana['weights'])[0]))
  with self.assertRaises(ValueError):box_sdf(p,field.new_empty(0,2,3))
if __name__=='__main__':unittest.main(verbosity=2)
