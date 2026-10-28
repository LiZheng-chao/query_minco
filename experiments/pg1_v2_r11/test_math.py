import unittest
import torch
import numpy as np
from experiments.pg1_v2_r11.math_probes import aggregate,current_time_quality,project_quality,unique_task_rows
class Math(unittest.TestCase):
 def test_independent_mass(self):
  z=torch.linspace(0,.1,12,dtype=torch.double).reshape(1,6,2)
  a=torch.tensor([.3,.7],dtype=torch.double).expand_as(z);rho=torch.tensor([1.,2.],dtype=torch.double).expand_as(z)
  y=aggregate(z,a,rho)
  m=(a*rho)/(a*rho).sum(-1,keepdim=True)
  zz=z.numpy();mm=m.numpy();lse=.025*np.log((mm*np.exp(zz/.025)).sum(-1))
  np.testing.assert_allclose(y['lse_rho'],lse,rtol=1e-12)
  torch.testing.assert_close(y['mean_rho'],(m*z).sum(-1))
 def test_gradcheck(self):
  torch.manual_seed(2)
  z=(torch.rand(1,6,7,dtype=torch.double)+.1).requires_grad_()
  a=(torch.rand_like(z)+.1).requires_grad_();rho=torch.rand_like(z)+1
  for name in aggregate(z,a,rho):
   self.assertTrue(torch.autograd.gradcheck(lambda zz,aa:aggregate(zz,aa,rho)[name],(z,a),eps=1e-6,atol=1e-5,rtol=1e-3))
 def test_rho_one(self):
  z=torch.rand(1,6,5,dtype=torch.double);a=torch.rand_like(z)
  y=aggregate(z,a,torch.ones_like(z))
  for k in ['mean','lse','tail']:torch.testing.assert_close(y[k+'_plain' if k=='mean' else k],y[k+'_rho'])
 def test_split_mass(self):
  z=torch.rand(1,6,5,dtype=torch.double);a=torch.rand_like(z);r=torch.rand_like(z)+1
  y=aggregate(z,a,r);yy=aggregate(z.repeat_interleave(2,-1),a.repeat_interleave(2,-1)/2,r.repeat_interleave(2,-1))
  for k in y:torch.testing.assert_close(y[k],yy[k])
 def test_quality_time_derivative(self):
  t=torch.tensor([1.4],dtype=torch.double,requires_grad=True);qa=t**-3;qj=t**-5
  z=current_time_quality(qa,qj,t)
  self.assertAlmostEqual(float(torch.autograd.grad(z['QA_current'],t,retain_graph=True)[0]),float(-4*2.2*t**-5))
  self.assertAlmostEqual(float(torch.autograd.grad(z['QJ_current'],t)[0]),float(-6*2.2*t**-7))
 def test_projection(self):
  gs=torch.tensor([1.,2.]);gq=-gs.clone();p=project_quality(gq,gs)
  self.assertGreaterEqual(float(p.dot(gs)),-1e-6)
  torch.testing.assert_close(project_quality(gs,gs),gs)
  torch.testing.assert_close(project_quality(gs,gs*0),gs)
 def test_exclusive_multimodel_journal(self):
  import tempfile,json
  from pathlib import Path
  from experiments.pg1_v2_r11.audit import save_model_snapshot,ROOT
  out=ROOT/'research_outputs/pg1_v2_r11/audit003/_preparation'
  with tempfile.TemporaryDirectory(prefix='journal_test_',dir=str(out)) as td:
   pp=Path(td);(pp/'runs').mkdir();(pp/'audit').mkdir()
   for name in ['C1','E2','PG1']:save_model_snapshot(pp,name,dict(CASES=[name]),[name],1.)
   self.assertEqual(len(list((pp/'runs').glob('*.json'))),3)
   self.assertEqual(json.loads((pp/'runs/E2_CASES.json').read_text())['records'],['E2'])
   with self.assertRaises(FileExistsError):save_model_snapshot(pp,'E2',dict(CASES=['bad']),['E2'],1.)
   self.assertEqual(json.loads((pp/'runs/E2_CASES.json').read_text())['records'],['E2'])
 def test_unique_metadata(self):
  from experiments.pg1_v2_r11.math_probes import mark_first_visits
  seen=set();self.assertEqual(mark_first_visits(['a','a','b'],seen),[True,False,True]);self.assertEqual(seen,{'a','b'})
  x=dict(model='m',sample_id='a',batch=0,first_visit=True,terms={'q':[1.]})
  self.assertEqual(len(unique_task_rows([x,dict(x)])),1)
  with self.assertRaises(ValueError):unique_task_rows([x,dict(x,terms={'q':[2.]})])
 def test_reject(self):
  z=torch.ones(1,6,3)
  for a,r in [(z*-1,z),(z,z*0),(z*float('nan'),z)]:
   with self.assertRaises(ValueError):aggregate(z,a,r)
if __name__=='__main__':unittest.main(verbosity=2)
