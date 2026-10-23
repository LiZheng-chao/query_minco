"""Gate E CPU-only tests. Never call optimizer.step, including synthetic runs."""
import copy,json,os,pathlib,tempfile,unittest,subprocess
from unittest.mock import patch
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest
from query_minco.training.run_spec import source_files
from query_minco.training.pilot_spec import load_spec,validate,recipe,lr_at,qualify,authorize
from query_minco.training.pilot_worker import initialize,batch,preflight
from query_minco.training.pilot_audit import audit_batch,audit_run
from query_minco.training.checkpoints import save_checkpoint,load_checkpoint
from query_minco.training.recipe import optimizer
from query_minco.training import objective
from query_minco.inference.identity import state_sha

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'research_outputs/rsps_gate_e_v1/development001'
class GateE(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  cls.artifacts=pathlib.Path(os.environ['RSPS_CPU_TEST_ARTIFACTS'])
  assert cls.artifacts.parent==OUT/'audit';cls.artifacts.mkdir()
  cls.specs={}
  for arm in ['E0','E1','E2','E3']:
   s=read(OUT/('preparation/cpu_configs003/'+arm+'.json'))
   s['code']=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),files=source_files(),allow_uncommitted_sources=True)
   cls.specs[arm]=validate(s)
  write(cls.artifacts/'CPU_RUN_SPECS.json',cls.specs)
  cls.ds,cls.order,_=qualify(cls.specs['E3']);cls.samples=cls.ds.load(cls.order.batch(0)[0])
 def test_recipe_budget_and_negative_contracts(self):
  self.assertEqual([recipe('E3')['max_updates'],recipe('E3')['checkpoints']],[32,[0,8,16,32]])
  self.assertAlmostEqual(lr_at(0),2e-5);self.assertAlmostEqual(lr_at(31),1e-6)
  for i in [-1,32,2880]:
   with self.assertRaises(ValueError):lr_at(i)
  for change in [lambda s:s['recipe'].update(max_updates=960),lambda s:s.update(seed=43),
                 lambda s:s.update(training_method_id='query_minco_imitation_e1_trainer_v1'),
                 lambda s:s['sidecar'].update(sha256='0'*64)]:
   s=copy.deepcopy(self.specs['E3']);change(s)
   with self.assertRaises(ValueError):validate(s)
 def test_all_four_cpu_kernels_and_independent_audit(self):
  baseline=None
  for arm,s in self.specs.items():
   m=initialize(s);before=state_sha(m)
   rec,a=batch(m,self.samples,self.ds.geometry,s)
   result=audit_batch(s,rec,a,self.samples,self.ds.geometry,0);self.assertTrue(result['passed'])
   self.assertEqual(state_sha(m),before);self.assertEqual(np.count_nonzero(a['raw_grad'][...,19:21]),0)
   if arm=='E0':
    m.zero_grad(set_to_none=True)
    loss,records,raw,out,costs=objective.forward(m,self.samples,'B_set','cpu');loss.backward()
    self.assertEqual(rec['loss'],loss.item());self.assertTrue(np.array_equal(a['raw'],raw.detach().numpy()))
    self.assertTrue(np.array_equal(a['raw_grad'],raw.grad.numpy()))
    baseline=(a['raw'],rec['imitation'],rec['matches'])
   else:
    self.assertTrue(np.array_equal(a['raw'],baseline[0]));self.assertEqual(rec['imitation'],baseline[1]);self.assertEqual(rec['matches'],baseline[2])
    self.assertGreater(rec['unmatched_physical_gradient_norm'],0)
   bad=copy.deepcopy(rec);bad['loss']+=.1
   with self.assertRaises(ValueError):audit_batch(s,bad,a,self.samples,self.ds.geometry,0)
   bad=copy.deepcopy(a);bad['raw_grad'][0,0,19]=1
   with self.assertRaises(ValueError):audit_batch(s,rec,bad,self.samples,self.ds.geometry,0)
   print('CPU_ARM',arm,rec['loss'],rec['physical_imitation_gradient_ratio'])
 def test_preflight_has_no_optimizer_creation(self):
  s=self.specs['E3'];p=self.artifacts/'direct_preflight';p.mkdir()
  write(p/'ENVIRONMENT.json',dict(executable=os.sys.executable,torch=torch.__version__,device='cpu'))
  with patch('query_minco.training.pilot_worker.optimizer',side_effect=AssertionError('optimizer forbidden')) as factory:
   preflight(s,p)
   factory.assert_not_called()
  self.assertFalse(read(p/'PREFLIGHT.json')['optimizer_created'])
  self.assertEqual(read(p/'PREFLIGHT.json')['optimizer_updates'],0)
 def test_authorization_refuses_cpu_unapproved_and_scope(self):
  s=self.specs['E0']
  with tempfile.TemporaryDirectory(dir=str(OUT/'audit')) as d:
   p=pathlib.Path(d)/'auth.json'
   value=dict(schema=1,scope='RSPS_MANUAL_PILOT32',run_id=s['run_id'],spec_sha256=digest(s),max_updates=32,authorized_by='NEGATIVE_TEST',approved=True)
   write(p,value)
   with self.assertRaises(ValueError):authorize(s,p)
   cuda=copy.deepcopy(s);cuda['device']='cuda';cuda['code']['allow_uncommitted_sources']=False
   for change in [dict(approved=False),dict(max_updates=960),dict(spec_sha256='0'*64),dict(scope='FORMAL_TRAINING')]:
    negative=dict(value,approved=True,max_updates=32,scope='RSPS_MANUAL_PILOT32',spec_sha256=digest(cuda));negative.update(change);p.write_text(json.dumps(negative))
    with self.assertRaises(ValueError):authorize(cuda,p)
 def test_step0_checkpoint_roundtrip_no_updates(self):
  s=self.specs['E0'];m=initialize(s);opt=optimizer(m)
  self.assertFalse(opt.state)
  with tempfile.TemporaryDirectory(dir=str(OUT/'audit')) as d:
   run=pathlib.Path(d);save_checkpoint(run,m,opt,s,0,scheduler_id='pilot_spec.lr_at_v1')
   with self.assertRaises(FileExistsError):save_checkpoint(run,m,opt,s,0)
   other=initialize(s);other_opt=optimizer(other)
   blob=load_checkpoint(run/'checkpoint_step0000.pt',other,other_opt,s)
   self.assertEqual(state_sha(m),state_sha(other));self.assertEqual(blob['scheduler'],'pilot_spec.lr_at_v1')
   self.assertFalse(other_opt.state)
 def test_cpu_development_evaluation_and_alarm(self):
  from query_minco.training.pilot_evaluation import evaluate_model,metric_alarm
  s=self.specs['E0'];m=initialize(s);result=evaluate_model(m,s['development'])
  self.assertEqual(len(result['rows']),8);self.assertEqual(result['selector'],'N/A')
  write(self.artifacts/'CPU_DEV_STEP0.json',result)
  history=[];base=dict(goal_rate=1.,oracle_rate=1.,feasible_mode_count=8)
  bad=dict(goal_rate=.9,oracle_rate=.9,feasible_mode_count=6)
  self.assertTrue(metric_alarm(history,bad,base,{}))
  with self.assertRaises(ValueError):metric_alarm(history,bad,base,{})
if __name__=='__main__':unittest.main(verbosity=2)
