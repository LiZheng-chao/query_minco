"""Small CPU optimizer tests ONLY; no real checkpoint or official training profile."""
import copy,json,os,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
import torch,numpy as np
from query_minco.models import QueryMincoPlanner
from query_minco.training.io import read,write,sha,require
from query_minco.training.pg1 import execution as ex
from query_minco.training.pg1 import spec as pg
from query_minco.training.pg1.state_io import save_commit,recovery_point,restore,inspect_commit,atomic_json
from query_minco.inference.identity import state_sha
from test_certified_empty_p0 import Fixtures
class Execution(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
  self.t=None
  if os.environ.get('PG1_EXECUTOR_TEST_OUTPUT'):
   self.root=Path(os.environ['PG1_EXECUTOR_TEST_OUTPUT'])/('A42_E2paired_cpu_'+self._testMethodName+'_'+uuid.uuid4().hex[:8]);self.root.mkdir(parents=True)
  else:
   self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
  self.f=Fixtures(self.root/'fixture',['free','free'])
  self.profile=patch.dict(pg.PROFILES,{'pg1_synthetic_updates_v1':dict(updates=3,seeds=[42],checkpoints=[0,1,2,3],wall_cap_s=120)})
  self.profile.start();self.initial=copy.deepcopy(QueryMincoPlanner().state_dict())
  self.s=dict(schema='SYNTHETIC_CPU_ONLY',profile='pg1_synthetic_updates_v1',device='cpu',run_id='A42_E2_synthetic',
   resources=dict(wall_s=120,disk_cap_gib=.25),method=ex.SUPPORTED[0],seed=42)
 def tearDown(self):
  self.profile.stop()
  if self.t:self.t.cleanup()
 def model(self):
  m=QueryMincoPlanner();m.load_state_dict(self.initial);m.train()
  return m,torch.optim.AdamW(m.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 def batch(self,i):return self.f.samples,self.f.geo,ex.SUPPORTED[0]
 def runpath(self,name):p=self.root/name;p.mkdir();return p
 def test_actual_step_both_methods_fixed_head(self):
  for method in ex.SUPPORTED:
   m,o=self.model();before=state_sha(m);rows=ex.head_rows(m)
   rec,a=ex.update(m,o,self.f.samples,self.f.geo,method,'pg1_synthetic_updates_v1',0);o.step()
   self.assertNotEqual(before,state_sha(m));self.assertTrue(ex.rows_unchanged(m,rows))
   self.assertEqual(rec['completed_optimizer_updates'],1);self.assertEqual(len(o.state),len(list(m.parameters())))
   self.assertTrue(all(int(v['step'])==1 for v in o.state.values()))
 def test_atomic_no_overwrite_and_integrity(self):
  m,o=self.model();r=self.runpath('atomic');p=save_commit(r,m,o,self.s,0,dict(step=0))
  with self.assertRaises(OSError):save_commit(r,m,o,self.s,0,dict(step=0))
  inspect_commit(p,self.s)
  (p/'UPDATE.json').write_text('{}')
  with self.assertRaises(ValueError):inspect_commit(p,self.s)
 def test_cpu_resume_committed_and_staged_exact(self):
  m,o=self.model();base=self.runpath('base');ex.run_engine(self.s,base,m,o,self.batch);expected=copy.deepcopy(m.state_dict())
  for phase in ('before_step','staged','committed'):
   a,ao=self.model();parent=self.runpath('parent_'+phase)
   def fault(p,n):
    if p==phase and n==2:raise RuntimeError('injected '+phase)
   with self.assertRaises(RuntimeError):ex.run_engine(self.s,parent,a,ao,self.batch,fault=fault)
   point=recovery_point(parent,self.s);b,bo=self.model();child=self.runpath('child_'+phase)
   ex.run_engine(self.s,child,b,bo,self.batch,root_spec=self.s,resume=point)
   self.assertTrue(all(torch.equal(expected[k],b.state_dict()[k]) for k in expected),phase)
   self.assertTrue(all(int(x['step'])==3 for x in bo.state.values()))
 def test_after_step_ambiguous_recovery_refused(self):
  m,o=self.model();p=self.runpath('ambiguous')
  def fault(phase,n):
   if phase=='after_step' and n==1:raise RuntimeError('unknown durable state')
  with self.assertRaises(RuntimeError):ex.run_engine(self.s,p,m,o,self.batch,fault=fault)
  with self.assertRaisesRegex(ValueError,'AMBIGUOUS_UPDATE'):recovery_point(p,self.s)
 def test_corrupt_cursor_optimizer_spec_rejected(self):
  m,o=self.model();p=self.runpath('state');c=save_commit(p,m,o,self.s,0,dict(step=0))
  other=copy.deepcopy(self.s);other['run_id']='different'
  with self.assertRaises(ValueError):restore(c,m,o,other)
 def test_fsync_failure_leaves_no_published_state(self):
  m,o=self.model();p=self.runpath('ioerror')
  with patch('os.fsync',side_effect=OSError('injected fsync')):
   with self.assertRaises(OSError):save_commit(p,m,o,self.s,0,dict(step=0))
  self.assertFalse((p/'commits/step0000').exists())
 def test_finite_dtype_and_formula_regression(self):
  m,o=self.model();r,a=ex.update(m,o,self.f.samples,self.f.geo,ex.SUPPORTED[0],'pg1_synthetic_updates_v1',0)
  self.assertTrue(r['numpy_reconstruction']);self.assertFalse(np.count_nonzero(a['raw_grad'][...,19:21]))
  self.assertAlmostEqual(pg.lr_at('pg1_formal2880_v1',0),2e-5);self.assertAlmostEqual(pg.lr_at('pg1_formal2880_v1',2879),1e-6)
 def test_self_consistent_bad_optimizer_counter_refused(self):
  m,o=self.model();rec,arrays=ex.update(m,o,self.f.samples,self.f.geo,ex.SUPPORTED[0],'pg1_synthetic_updates_v1',0);o.step()
  run=self.runpath('bad_counter');c=save_commit(run,m,o,self.s,1,rec)
  p=c/'checkpoint_step0001.pt';blob=torch.load(str(p),map_location='cpu',weights_only=True)
  for state in blob['optimizer_state_dict']['state'].values():state['step'].fill_(99)
  with p.open('wb') as f:torch.save(blob,f)
  receipt=read(str(p)+'.json');receipt['sha256']=sha(p);Path(str(p)+'.json').write_text(json.dumps(receipt))
  commit=read(c/'COMMIT.json');commit['files'][p.name]=sha(p);commit['files'][p.name+'.json']=sha(str(p)+'.json');(c/'COMMIT.json').write_text(json.dumps(commit))
  with self.assertRaisesRegex(ValueError,'OPTIMIZER_EXACT_STEPS'):restore(c,m,o,self.s)
 def test_paused_family_and_content_audit(self):
  from types import SimpleNamespace
  from query_minco.training.io import digest
  m,o=self.model();parent=self.runpath('parent_audit')
  s=dict(self.s,recipe=pg.recipe(ex.SUPPORTED[0]),parent=None,output_root=str(self.root),code={'synthetic':True},bindings=None)
  write(parent/'RUN_SPEC.json',s);write(parent/'ROOT_SPEC.json',s)
  ex.run_engine(s,parent,m,o,self.batch,pause_at=1)
  point=recovery_point(parent,s);b,bo=self.model();child=self.runpath('child_audit')
  cs=dict(s,run_id='A42_E2_resume_test',parent=dict(path=str(parent/'RUN_SPEC.json'),sha256=sha(parent/'RUN_SPEC.json')))
  write(child/'RUN_SPEC.json',cs);write(child/'ROOT_SPEC.json',s)
  ex.run_engine(cs,child,b,bo,self.batch,root_spec=s,resume=point)
  write(child/'PROCESS_EXIT.json',dict(returncode=0))
  ids=[x.sample_id for x in self.f.samples]
  self.f.geo.orders[42]=SimpleNamespace(batch=lambda i:(ids,['synthetic']*len(ids)))
  result=ex.audit_run(child,ds=SimpleNamespace(load=lambda ids:self.f.samples),geo=self.f.geo)
  self.assertTrue(result['passed']);self.assertTrue(result['recovery_exercised'])
 def test_partial_checkpoint_never_published(self):
  m,o=self.model();run=self.runpath('partial')
  def failed(blob,handle):handle.write(b'PARTIAL');raise OSError('partial write')
  with patch('torch.save',side_effect=failed),self.assertRaises(OSError):save_commit(run,m,o,self.s,0,{})
  self.assertFalse((run/'commits/step0000').exists())
 def test_claim_replay_and_scope_rejected(self):
  m=copy.deepcopy(self.s);m['run_id']='A42_E2_synthetic_auth'
  a=dict(ex.authorization(m),approved=True,authorized_by='CPU_TEST_ONLY')
  with patch.object(ex,'validate',return_value=m),patch.object(ex,'OUTPUT',self.root):
   ex.claim(m,a)
   with self.assertRaises(FileExistsError):ex.claim(m,a)
   bad=dict(a,additional_updates=99)
   with self.assertRaises(ValueError):ex.claim(m,bad)
 def test_public_does_not_offer_cpu_formal_or_auth_bypass(self):
  with self.assertRaises((ValueError,KeyError)):ex.validate(self.s)
  p=self.root/'json';atomic_json(p,dict(ok=True))
  with self.assertRaises(FileExistsError):atomic_json(p,dict(overwrite=True))

# Further semantic recovery counterexamples live in the same synthetic-only class.

class Supervisor(unittest.TestCase):
 def test_timeout_and_nonzero_exit_are_not_pass(self):
  import sys
  from query_minco.training.pg1.executor_cli import supervise
  with tempfile.TemporaryDirectory() as d:
   r=supervise([sys.executable,'-c','import time;time.sleep(2)'],Path(d)/'timeout',.1)
   self.assertTrue(r['timeout']);self.assertNotEqual(r['returncode'],0)
   r=supervise([sys.executable,'-c','raise SystemExit(7)'],Path(d)/'failed',2)
   self.assertEqual(r['returncode'],7);self.assertFalse(r['timeout'])
class ActivationProbe(unittest.TestCase):
 def test_synthetic_active_signals_zero_updates(self):
  from query_minco.training.pg1.activation import batch_stats
  from test_certified_empty_p0 import RawModel
  with tempfile.TemporaryDirectory() as d:
   f=Fixtures(Path(d)/'data',['free','obstacle'],goal=2.4);m=RawModel();before=state_sha(m)
   r=batch_stats(m,f.samples,f.geo)
   self.assertEqual(state_sha(m),before);self.assertEqual(r['all']['queries'],12)
   self.assertGreater(r['all']['activation']['goal']['exact_nonzero'],0)

if __name__=='__main__':unittest.main(verbosity=2)
