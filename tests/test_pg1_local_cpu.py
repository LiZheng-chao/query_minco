"""CPU adapter contracts; actual real-data smoke is recorded separately."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.pg1 import local_cpu as cpu,execution as ex
from query_minco.training.io import sha,read,write
from test_certified_empty_p0 import Fixtures

BIND=Path(__file__).resolve().parents[1]/'research_outputs/pg1_task_complete_v1/formal_executor_v1/_preparation/clean_bindings_v1/BINDINGS.json'
class Contracts(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
  self.binding=self.root/'bindings.json';write(self.binding,dict(code=dict(git_commit='synthetic',files={},allow_uncommitted_sources=False)))
  self.bb=dict(path=str(self.binding),sha256=sha(self.binding))
  self.s=cpu.make_spec(self.bb,ex.SUPPORTED[0],'A42_E2_CPU_test')
 def tearDown(self):self.t.cleanup()
 def test_cpu_new_identity_does_not_open_gpu_validator(self):
  with patch.object(cpu,'verify_code'),patch.object(cpu,'environment'):
   cpu.validate(self.s)
  with self.assertRaises(ValueError):ex.validate(self.s)
  self.assertEqual(self.s['optimizer'],dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.))
  self.assertEqual(self.s['profile'],'pg1_pilot32_v1');self.assertEqual(self.s['cpu_updates'],32)
 def test_cuda_formal_threshold_and_field_tampering_rejected(self):
  for key,value in [('device','cuda'),('profile','pg1_formal2880_v1'),('cpu_updates',2880),('optimizer',{}),('parent',{'forged':True})]:
   s=copy.deepcopy(self.s);s[key]=value
   with self.assertRaises((ValueError,KeyError)),patch.object(cpu,'verify_code'),patch.object(cpu,'environment'):cpu.validate(s)
  s=dict(self.s,unexpected=True)
  with self.assertRaises(ValueError),patch.object(cpu,'verify_code'),patch.object(cpu,'environment'):cpu.validate(s)
 def test_authorization_scope_replay_and_arm_binding(self):
  a=cpu.authorization(self.s)
  with self.assertRaises(ValueError):cpu.check_auth(self.s,a)
  good=dict(a,approved=True,authorized_by='SYNTHETIC_UNIT_TEST')
  cpu.check_auth(self.s,good)
  for key,v in [('max_optimizer_updates',2880),('scope','ONE_PG1_EXECUTION_RUN'),('approved','true')]:
   with self.assertRaises(ValueError):cpu.check_auth(self.s,dict(good,**{key:v}))
  other=cpu.make_spec(self.bb,ex.SUPPORTED[1],'A42_PG1_CPU_test')
  with self.assertRaises(ValueError):cpu.check_auth(other,good)
 def test_cpu_memory_pressure_fails_closed(self):
  for m in [dict(peak_rss_mib=4097,available_mib=9999),dict(peak_rss_mib=100,available_mib=3071)]:
   with patch.object(cpu,'memory',return_value=m),self.assertRaises(ValueError):cpu.resource_check(self.s)
 def test_smoke_not_equivalent_to_gpu_preflight(self):
  self.assertEqual(cpu.make_spec(self.bb,ex.SUPPORTED[0],'A42_smoke','smoke2')['cpu_updates'],2)
  with self.assertRaisesRegex(ValueError,'SMOKE_REQUIRED'):cpu.verify_smoke(self.s)
 def test_partial_or_unapproved_run_has_no_automatic_pair_pass(self):
  with self.assertRaises(ValueError):cpu.pair([],self.root/'pair')
  with self.assertRaises((ValueError,FileNotFoundError)):cpu.audit(self.root/'missing')

 def test_paired_output_uses_real_rows_and_reports_equal_weights_honestly(self):
  runs=[]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for method in ex.SUPPORTED:
    p=root/method;p.mkdir();(p/'commits').mkdir();(p/'intents').mkdir();(p/'checkpoints').mkdir()
    write(p/'RUN_SPEC.json',dict(method=method,purpose='pilot32',bindings={'synthetic':True},code={'synthetic':True}))
    write(p/'RUN_SEAL.json',dict(synthetic=True))
    dev=dict(rows=[dict(id='shared-task',physics=[True]*6,goal=[True]*6)],physics_rate=1.,goal_rate=1.)
    for i in range(33):
     cp=p/'commits'/('step%04d'%i);cp.mkdir()
     write(cp/'UPDATE.json',dict(ids=['shared-task'],lr=.1,losses={'goal':0.},local_cpu_diagnostics={'synthetic':True},
       physical_imitation_gradient_ratio=1.,dev_baseline=dev,development=dev))
    np.savez(p/'intents/step0032_ARRAYS.npz',raw=np.zeros((1,6,21)))
    torch.save(dict(state_dict={'weight':torch.zeros(2)}),p/'checkpoints/checkpoint_step0032.pt')
    runs.append(p)
   with patch.object(cpu,'audit',return_value={'passed':True}):cpu.pair(runs,root/'pair')
   result=read(root/'pair/PAIRED_RESULTS.json')
   self.assertTrue(result['identical_final_weights']);self.assertEqual(result['final_weight_max_abs_difference'],0.)
   self.assertEqual(sorted(result['paired_development']),['0','16','32','8'])

class Numerical(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
  self.t=tempfile.TemporaryDirectory();self.f=Fixtures(Path(self.t.name)/'data',['free','free'])
 def tearDown(self):self.t.cleanup()
 def test_cpu_actual_update_and_diagnostics_both_arms(self):
  initial=copy.deepcopy(QueryMincoPlanner().state_dict())
  for method in ex.SUPPORTED:
   m=QueryMincoPlanner();m.load_state_dict(initial);o=torch.optim.AdamW(m.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
   before=state_sha(m);rec,a=cpu.cpu_update(m,o,self.f.samples,self.f.geo,method,'pg1_pilot32_v1',0)
   o.step();self.assertNotEqual(before,state_sha(m))
   self.assertEqual(rec['local_cpu_diagnostics']['matched_queries']+rec['local_cpu_diagnostics']['unmatched_queries'],12)
   self.assertEqual(rec['local_cpu_diagnostics']['activation']['goal']['all']['count'],12)
   self.assertTrue(all(int(v['step'])==1 for v in o.state.values()))
   self.assertTrue(np.isfinite(a['local_cpu_dynamic_maxima']).all())
 def test_original_second_ratio_alarm_stops_before_step(self):
  m=QueryMincoPlanner();o=torch.optim.AdamW(m.parameters());original=ex.forward
  def bad(*args,**kwargs):
   z=original(*args,**kwargs);z['terms']['obstacle']=z['terms']['imitation']*100.
   z['loss']=z['loss']+z['terms']['obstacle'];return z
  with patch.object(ex,'forward',side_effect=bad),self.assertRaisesRegex(ValueError,'TWO_STEP_GRADIENT_RATIO_GT10'):
   cpu.cpu_update(m,o,self.f.samples,self.f.geo,ex.SUPPORTED[0],'pg1_pilot32_v1',1,1)
  self.assertFalse(o.state)
 def test_nonfinite_is_rejected_before_update(self):
  m=QueryMincoPlanner();o=torch.optim.AdamW(m.parameters());original=ex.forward
  def bad(*args,**kwargs):
   z=original(*args,**kwargs);z['loss']=z['loss']*float('nan');return z
  with patch.object(ex,'forward',side_effect=bad),self.assertRaises(ValueError):
   cpu.cpu_update(m,o,self.f.samples,self.f.geo,ex.SUPPORTED[0],'pg1_pilot32_v1',0)
  self.assertFalse(o.state)
 def test_continuous_dynamic_report_reuses_original_extrema(self):
  a=dict(raw=np.zeros((1,6,21)),coeffs=np.zeros((1,6,3,3,6)),durations=np.ones((1,6,3)))
  a['coeffs'][:,:,:,0,1]=4.
  for name in ('goal','dynamic','domain','obstacle'):a['per_query_'+name]=np.ones((1,6))
  d,p=cpu.diagnostics(a,[dict(queries=[0])],[3.,8.,100.])
  self.assertEqual(d['original_minco_dynamic']['passed'],0);self.assertTrue(np.allclose(p[:,:,0],4))
  self.assertEqual(d['activation']['goal']['unmatched']['count'],5)
if __name__=='__main__':unittest.main(verbosity=2)
