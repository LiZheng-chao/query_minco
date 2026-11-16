import unittest
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v1.map import Snapshot,Config,KNOWN_FREE,OCCUPIED
from query_minco.integration.observed_space_v2.core import Prepared
from query_minco.integration.depth_minco_t2_v1.revalidate import latest_proof
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
from query_minco.integration.yopo_execution_v4.trajectory import build
class Tests(unittest.TestCase):
 def setup_case(self,oldstamp=100.):
  p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.2,0,0],100.)
  s=Snapshot(np.full((12,12,12),KNOWN_FREE,np.uint8),[-.6,-.6,-.6],Config(shape=(12,12,12)),[dict(id='SYNTHETIC_CERTIFIED_FIXTURE',stamp=oldstamp)],oldstamp,1,lambda:1)
  return p,s
 def test_latest_conflict_overrides_cached_safe_geometry(self):
  p,s=self.setup_case();pose=dict(p=[0,0,0],q=[0,0,0,1])
  birth=lambda now:BirthGrant((0,0,0),.6,100.2,'SYNTHETIC_ONLY')
  clear=[(np.full((90,160),5.,np.float32),pose,100.03,2)]
  _,proof,_=latest_proof(s,p,clear,birth,lambda:100.03)
  self.assertTrue(proof['passed'])
  blocked=[(np.full((90,160),.15,np.float32),pose,100.03,2)]
  with self.assertRaises(ValueError):latest_proof(s,p,blocked,birth,lambda:100.03)
 def test_new_frame_cannot_resurrect_expired_history(self):
  p,s=self.setup_case(97.)
  frames=[(np.full((90,160),5.,np.float32),dict(p=[0,0,0],q=[0,0,0,1]),100.03,2)]
  with self.assertRaises(ValueError):latest_proof(s,p,frames,lambda now:None,lambda:100.03)
 def test_incidence_cache_never_caches_permission(self):
  p,s=self.setup_case();prepared=Prepared(p)
  self.assertTrue(s.certify(prepared,100.)['passed'])
  states=s.states.copy();states[6,6,6]=OCCUPIED
  changed=Snapshot(states,s.origin,s.cfg,s.frames,100.,2,lambda:2)
  with self.assertRaises(ValueError):changed.certify(prepared,100.)
 def test_explicit_f0_scene_and_reference(self):
  s=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
  self.assertIn("'open' if a.stage=='D' else 'forest'",s)
  self.assertIn("'--stage','D' if stage=='F0'",s)
  self.assertIn("['D','F0']",s)

 def test_partial_birth_proof_is_containment_not_carving(self):
  from query_minco.integration.observed_space_v1.map import birth_covers_intersection
  b=BirthGrant((0,0,0),.6,100.2,'SYNTHETIC_ONLY')
  lo=np.array([[.3,.1,.3]]);hi=lo+.1;p=np.array([.4,0,0])
  self.assertTrue(birth_covers_intersection(lo,hi,p,.33,b)[0])
  self.assertFalse(birth_covers_intersection(np.array([[.7,0,0]]),np.array([[.8,.1,.1]]),np.array([.8,0,0]),.33,b)[0])
