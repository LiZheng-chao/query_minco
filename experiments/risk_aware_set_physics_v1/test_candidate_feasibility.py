import unittest,json
import numpy as np
from query_minco.inference.candidate_feasibility import fast_soft_stops,soft_jerk_witness,prepare_fast_bound_reference
from query_minco.inference.trajectory_execution import Program,MotionLimits,quintic
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
class Tests(unittest.TestCase):
 def test_many_nonzero_pva_stops_exact_full_extrema(self):
  rng=np.random.default_rng(506)
  # Nonzero accelerations including opposition and positive acceleration.
  heads=np.zeros((32,3,3));heads[:,0]=[0,0,2]
  heads[:,1]=rng.uniform(-.25,.25,(32,3));heads[:,2]=rng.uniform(-.03,.03,(32,3))
  for h,s in zip(heads,fast_soft_stops(heads)):
   s.validate(MotionLimits())
   self.assertLessEqual(s.extrema()[2],.35+1e-9)
   np.testing.assert_allclose(s.pva(0),h,atol=1e-10)
   np.testing.assert_allclose(s.pva(s.duration)[1:],0,atol=1e-10)
 def test_quartic_matches_existing_quintic_formula(self):
  from query_minco.inference.trajectory_execution import quintic
  h=np.array([[.1,.2,2],[.3,.02,0],[-.02,.01,0]])
  s=fast_soft_stops(h[None])[0];T=s.duration
  q=quintic(h,np.stack([h[0]+h[1]*T/2+h[2]*T*T/12,np.zeros(3),np.zeros(3)]),T)
  np.testing.assert_allclose(s.coefficients,q.coefficients,atol=1e-12)
 def test_necessary_soft_witness_not_hard_impossibility(self):
  c=np.zeros((1,3,6));c[0,:,0]=[0,0,2];c[0,0,1]=.8;c[0,0,2]=.5
  p=Program(c,[.02])
  self.assertIsNotNone(soft_jerk_witness(p,.35))
  self.assertIsNone(soft_jerk_witness(p,10))
 def test_full_bank_same_lifetime_and_fail_closed(self):
  c=np.zeros((1,3,6));c[0,:,0]=[0,0,2];c[0,0,1]=.4;p=Program(c,[.4])
  v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],130,'fixture'),),Scope.ANALYTIC_FIXTURE)
  b=prepare_fast_bound_reference(p,v,10,10)
  self.assertLess(b.valid_until_ros,130)
  self.assertEqual(len(b.stops),21)
  for row in b.stops:
   proof=json.loads(row.proof);self.assertEqual(proof['status'],'CERTIFIED_FREE')
   self.assertEqual(proof['certified_prefix_sha256'],p.digest)
   self.assertLessEqual(proof['valid_until'],130)
  with self.assertRaisesRegex(ValueError,'EXPIRED'):prepare_fast_bound_reference(p,v,129.5,129.5)
  with self.assertRaisesRegex(ValueError,'ACTIVATION_LEAD'):prepare_fast_bound_reference(p,v,10,10,.2)
 def test_geometry_outside_unknown_not_free(self):
  c=np.zeros((1,3,6));c[0,:,0]=[9,0,2];p=Program(c,[.4])
  v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],130,'fixture'),),Scope.ANALYTIC_FIXTURE)
  with self.assertRaisesRegex(ValueError,'PREFIX_SPACE'):prepare_fast_bound_reference(p,v,10,10)
 def test_high_speed_fails(self):
  h=np.zeros((1,3,3));h[0,1,0]=1.1
  with self.assertRaisesRegex(ValueError,'HARD_LIMIT'):fast_soft_stops(h)
if __name__=='__main__':unittest.main()
