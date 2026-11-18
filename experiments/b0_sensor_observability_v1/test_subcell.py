from pathlib import Path
import sys,unittest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from experiments.b0_sensor_observability_v1.subcell_reference import *
class SubcellSafety(unittest.TestCase):
 def test_positive_depth_subcell(self):
  r=RayCover(np.full((90,160),5,np.float32),dict(p=SPAWN,q=[0,0,0,1]));self.assertTrue(r.ray_free(SPAWN+[1,-.01,-.01],SPAWN+[1.01,.01,.01])[0])
 def test_missing_point_depth_rejected(self):
  r=RayCover(np.full((90,160),20,np.float32),dict(p=SPAWN,q=[0,0,0,1]));self.assertFalse(r.ray_free(SPAWN+[1,0,0],SPAWN+[1,0,0])[0])
 def test_fov_point_rejected(self):
  r=RayCover(np.full((90,160),5,np.float32),dict(p=SPAWN,q=[0,0,0,1]));self.assertFalse(r.ray_free(SPAWN+[.5,0,1],SPAWN+[.5,0,1])[0])
 def test_support_bound_covers_birth_intersection(self):
  r=RayCover(np.full((90,160),20,np.float32),dict(p=SPAWN,q=[0,0,0,1]));lo=np.array([.84375,-.30625,1.9484375]);hi=np.array([.8453125,-.3046875,1.95]);p=np.array([.55,-.25,2.]);radius=.3043945308714705;birth=BirthGrant(tuple(SPAWN),.6,101.,'SYNTHETIC_TEST')
  self.assertTrue(r.cover(lo,hi,p,radius,birth));self.assertEqual(r.counts['DEPTH_FULL_SUBCELL'],0)
  pts=np.array(list(itertools.product(*[np.linspace(a,b,21) for a,b in zip(lo,hi)])));inside=np.linalg.norm(pts-p,axis=1)<=radius;self.assertTrue((np.linalg.norm(pts[inside]-SPAWN,axis=1)<=.6+1e-12).all())
 def test_new_reference_does_not_claim_success(self):
  d=json.loads((O/'SUBCELL_EXACT_SET_COVER_REFERENCE.json').read_text());self.assertFalse(d['results'][0]['result']['passed']);self.assertTrue(all(not x['passed'] for x in d['results'][1:]))
 def test_uncertain_point_is_outside_birth_and_inside_query(self):
  d=json.loads((O/'SUBCELL_EXACT_SET_COVER_REFERENCE.json').read_text())
  for row in d['actual_uncovered_point_witnesses']:
   w=row['actual_uncovered_point'];self.assertIsNotNone(w);self.assertGreater(w['distance_to_birth'],.6);self.assertLess(w['distance_to_query'],w['query_radius'])
if __name__=='__main__':unittest.main(verbosity=2)
