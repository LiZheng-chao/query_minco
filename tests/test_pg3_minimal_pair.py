"""Formal adapter contracts and evidence-bound exact-resume/metrics verification."""
import copy,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from query_minco.training.io import read,write,digest
from query_minco.training.pg3_next_exact_v1.minimal import WEIGHTS,ID
from experiments.pg3_minimal_pair_v1.executor import authorize,validate,qualified,SCOPE,run
ROOT=Path('/home/lizheng/query_minco_mainline')
def spec():
 s=copy.deepcopy(read(ROOT/'research_outputs/pg3_next_exact_v1/evidence003/MINIMAL/RUN_SPECS/MINIMAL_FORMAL_PLAN.json'))
 s.update(schema='pg3_minimal_pair_executor_v1',formal_scope=SCOPE,minimal_qualification='/nonexistent/qualification',method_id=ID,weights=dict(WEIGHTS));return s
def auth(s):return dict(approved=True,spec_sha256=digest(s),scope=SCOPE,max_updates=128,resources=s['execution'],authorized_by='test-only')
class MinimalPair(unittest.TestCase):
 def test_exact_authorization_and_resource_binding(self):
  s=spec();authorize(s,auth(s))
  for key,value in [('approved',False),('scope','PG3_V21_PAIRED128'),('max_updates',4),('spec_sha256','stale'),('resources',{})]:
   a=auth(s);a[key]=value
   with self.assertRaises(ValueError):authorize(s,a)
 def test_no_loss_or_method_drift(self):
  for k in ['stop','geom','keep','div','mode']:
   s=spec();s['weights'][k]+=1e-9
   with self.assertRaises(ValueError):validate(s)
  s=spec();s['stop_mode']='replace_full_stop'
  with self.assertRaises(ValueError):validate(s)
 def test_pg2_baseline_protocol_frozen(self):
  s=spec();s.update(arm='A',method_id='query_minco_pg2_guided_v2_safe_v1',weights={k:0. for k in WEIGHTS});validate(s)
  for k,v in [('seed',43),('batch_size',4),('updates',4),('schedule','new')]:
   z=copy.deepcopy(s);z[k]=v
   with self.assertRaises(ValueError):validate(z)
 def test_unauthorized_has_no_output_side_effect(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);s=spec();s['output_root']=str(p/'never');write(p/'s.json',s);write(p/'a.json',dict(approved=False))
   with self.assertRaises(ValueError):run(p/'s.json',p/'a.json')
   self.assertFalse((p/'never').exists())
 def test_qualification_missing_or_unbound_stops(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q.json';s=spec();s['minimal_qualification']=str(p)
   for i,q in enumerate([dict(status='BLOCKED'),dict(status='QUALIFIED',code={},sources={})]):
    p=Path(d)/('q%d.json'%i);s['minimal_qualification']=str(p);write(p,q)
    with self.assertRaises((ValueError,KeyError)):qualified(s)
 def test_evidence_real_executor_resume_and_metrics(self):
  name=os.environ.get('PG3_MINIMAL_PAIR_EVIDENCE')
  if not name:self.skipTest('new-source executor evidence required')
  p=Path(name);z=read(p/'SHORT_PAIR_RESULT.json');self.assertEqual(z['status'],'PASS')
  for arm in ('A','MINIMAL'):
   self.assertEqual(len(set(z['arms'][arm]['fingerprints'].values())),1)
   q=read(p/'short'/('A42_'+arm+'_u_run001')/'evaluation/training8_step0004_QUALITY_TRANSITIONS.json')
   self.assertEqual(q['original']['queries'],48);self.assertIn('initial_matched_safe_degraded',q['transitions']);self.assertIn('QA',q['normalized_quality'])
