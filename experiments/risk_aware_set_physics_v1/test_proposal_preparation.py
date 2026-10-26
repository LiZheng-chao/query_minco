import unittest,numpy as np
from query_minco.inference.proposal_preparation import prepare_bound_reference
from query_minco.inference.committed_reference import prepare_reference
from query_minco.inference.trajectory_execution import Program
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
class Tests(unittest.TestCase):
 def setUp(self):
  c=np.zeros((1,3,6));c[0,:,0]=[0,0,2];self.p=Program(c,[.4])
 def test_reproduces_and_fixes_caller_lifetime_mismatch(self):
  volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],129.99,'cpu'),),Scope.EXTERNAL_SUPERVISOR)
  with self.assertRaisesRegex(ValueError,'NO_FRESH_CONTINUOUS_VOLUME'):prepare_reference(self.p,volume,10,10)
  b=prepare_bound_reference(self.p,volume,10,10)
  self.assertLess(b.valid_until_ros,129.99)
  self.assertEqual(len(b.stops),21)
 def test_expiry_not_extended(self):
  v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],10.5,'cpu'),),Scope.EXTERNAL_SUPERVISOR)
  with self.assertRaisesRegex(ValueError,'EXPIRED'):prepare_bound_reference(self.p,v,10,10)
 def test_activation_bound_unchanged(self):
  v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],130,'cpu'),),Scope.EXTERNAL_SUPERVISOR)
  with self.assertRaisesRegex(ValueError,'ACTIVATION_LEAD'):prepare_bound_reference(self.p,v,10,10,.2)
if __name__=='__main__':unittest.main()
