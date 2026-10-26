"""Explicitly bound CPU-only real G2 identity/legacy counterexamples."""
import copy,json,os,tempfile,unittest
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha
from query_minco.training.run_spec import source_files
from query_minco.training.pg1.qualification import current_identity,qualify_data
from query_minco.training.pg1.legacy_c1 import load_legacy,metadata
from query_minco.registry import describe

@unittest.skipUnless(os.environ.get('PG1_G2_BINDINGS'),'explicit G2 manifest required')
class RealG2(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  cls.path=Path(os.environ['PG1_G2_BINDINGS']);cls.m=read(cls.path)
  cls.loaded={}
  for seed in (42,43,44):
   init=next(x for x in cls.m['initializations'] if x['seed']==seed)
   order=next(x['artifact'] for x in cls.m['orders'] if x['seed']==seed)
   cls.loaded[seed]=(init,order,load_legacy(init,seed,order))
 def test_three_seed_original_lineage_positive(self):
  for seed,(init,order,(b,audit)) in self.loaded.items():
   self.assertTrue(audit['passed']);self.assertFalse(audit['checkpoint_modified'])
   self.assertEqual(audit['projection']['stage_updates'],2880);self.assertEqual(seed,b['seed'])
   self.assertNotIn('completed_optimizer_updates',b);self.assertNotIn('experiment_arm',b['spec'])
 def test_current_PG1_and_historical_C1_identities_separate(self):
  _,r=current_identity(self.m['code'],self.m['historical_reference'])
  self.assertTrue(r['passed']);self.assertFalse(r['historical_runtime_identity_claimed'])
  self.assertEqual(len(r['exact_approved_differences']),1)
  with self.assertRaisesRegex(ValueError,'SOURCE_SHA'):describe()
 def test_current_source_and_historical_authority_negatives(self):
  for change in ('source','head','flag','manifest'):
   code=copy.deepcopy(self.m['code']);ref=copy.deepcopy(self.m['historical_reference'])
   if change=='source':code['files']['query_minco/training/pg1/spec.py']='0'*64
   elif change=='head':code['git_commit']='0'*40
   elif change=='flag':code['allow_uncommitted_sources']=1
   else:ref['sha256']='0'*64
   with self.subTest(change=change),self.assertRaises(ValueError):current_identity(code,ref)
 def test_no_generic_SOURCE_SHA_exemption(self):
  from unittest.mock import patch
  from query_minco.training.pg1 import qualification as q
  actual=q.sha
  for target in ['query_minco/evaluation/validation.py','query_minco/models/planner.py']:
   def changed(path):
    return '0'*64 if str(path).endswith(target) else actual(path)
   with self.subTest(target=target),patch.object(q,'sha',side_effect=changed),self.assertRaisesRegex(ValueError,'UNAPPROVED_LEGACY_SOURCE'):
    current_identity(self.m['code'],self.m['historical_reference'])
 def test_metadata_wrong_seed_arm_step_weights_source_order(self):
  for seed,(init,order,(blob,_)) in self.loaded.items():
   e=read(init['lineage']['path'])['evidence']
   receipt=read(e['receipt']['path']);cfg=read(e['run_config']['path'])
   for key in ('seed','arm','step','optimizer_updates_completed','weights_sha256','source_freeze_sha256','revision','actual_weight','order'):
    b=copy.copy(blob);b['spec']=copy.copy(blob['spec']);o=copy.deepcopy(order)
    if key=='actual_weight':
     b['state_dict']=copy.copy(blob['state_dict']);name=next(iter(b['state_dict']))
     b['state_dict'][name]=b['state_dict'][name].clone();b['state_dict'][name].view(-1)[0]+=1.
    elif key=='order':o=next(x['artifact'] for x in self.m['orders'] if x['seed']!=seed)
    else:b[key]=99 if key in ('seed','step','optimizer_updates_completed') else 'WRONG'
    with self.subTest(seed=seed,key=key),self.assertRaises(ValueError):
     metadata(b,seed,receipt,cfg,o,Path(e['source_freeze']['path']),Path(e['initialization_registry']['path']))
 def test_wrong_binding_and_missing_proof_rejected(self):
  for seed,(init,order,_) in self.loaded.items():
   for key in ('seed','weights','checkpoint_hash','checkpoint_swap','lineage_hash','missing_lineage','missing_receipt','wrong_source','wrong_seal','wrong_order','wrong_parent','wrong_immutable'):
    i=copy.deepcopy(init);o=copy.deepcopy(order)
    with tempfile.TemporaryDirectory() as d:
     if key=='seed':i['seed']=99
     elif key=='weights':i['weights_sha256']='0'*64
     elif key=='checkpoint_hash':i['checkpoint']['sha256']='0'*64
     elif key=='checkpoint_swap':i['checkpoint']=next(x['checkpoint'] for x in self.m['initializations'] if x['seed']!=seed)
     elif key=='lineage_hash':i['lineage']['sha256']='0'*64
     elif key=='missing_lineage':i['lineage']=None
     elif key=='wrong_order':o=next(x['artifact'] for x in self.m['orders'] if x['seed']!=seed)
     else:
      desc=read(i['lineage']['path'])
      if key=='missing_receipt':del desc['evidence']['receipt']
      else:
       target={'wrong_source':'source_freeze','wrong_seal':'training_seal','wrong_parent':'parent_checkpoint','wrong_immutable':'immutable_inputs'}[key]
       desc['evidence'][target]['sha256']='0'*64
      p=Path(d)/'evidence.json';write(p,desc);i['lineage']=dict(path=str(p),sha256=sha(p))
     with self.subTest(seed=seed,key=key),self.assertRaises((ValueError,KeyError)):load_legacy(i,seed,o)
 def test_frozen_data_three_order_geometry_and_isolation(self):
  ds,geo,audit=qualify_data(self.m);self.assertTrue(audit['passed'])
  self.assertEqual(len(ds.rows),384);self.assertEqual(len(geo.rows),384)
  self.assertEqual(audit['train_dev_scene_overlap'],0)
 def test_frozen_data_and_proof_negative(self):
  for key in ('dataset','sidecar','development','order'):
   m=copy.deepcopy(self.m)
   if key=='order':m['orders'][0]['artifact']['sha256']='0'*64
   else:m[key]['sha256']='0'*64
   with self.subTest(key=key),self.assertRaises(ValueError):qualify_data(m)

class ParameterAuditNegatives(unittest.TestCase):
 def test_parameter_gradient_corruption_rejected(self):
  from test_certified_empty_p0 import Fixtures,RawModel
  from query_minco.training.pg1.objective import cpu_zero_update
  from query_minco.training.pg1.audit import audit_parameter_gradients
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as d:
   f=Fixtures(Path(d)/'rows',['free','obstacle']);m=RawModel()
   method='query_minco_pg1_task_complete_v1';cpu_zero_update(m,f.samples,f.geo,method)
   r,_=audit_parameter_gradients(m,f.samples,f.geo,method);self.assertTrue(r['passed'])
   m.raw.grad[0,19]=1.
   with self.assertRaises(ValueError):audit_parameter_gradients(m,f.samples,f.geo,method)

if __name__=='__main__':unittest.main(verbosity=2)
