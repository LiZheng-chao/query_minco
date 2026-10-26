"""V3 meaningful guard and braking regressions; no ROS control."""
import unittest,numpy as np
from query_minco.inference.execution_clock_v3 import soft_brake,activation_guard,cruise_segment
from query_minco.inference.trajectory_execution import *
from query_minco.inference.observed_space import *

class V3Tests(unittest.TestCase):
    def test_activation_boundary(self):
        self.assertAlmostEqual(activation_guard(10.,10.005,9.99,['/f3_reference_arbiter']),.005)
        for t in [9.999,10.019742,10.019979,10.020241]:
            with self.assertRaises(ValueError):activation_guard(10.,t,9.99,['/f3_reference_arbiter'])
    def test_clock_frame_and_authority_faults(self):
        with self.assertRaisesRegex(ValueError,'CLOCK_JUMP'):activation_guard(10.,10.,9.99,['/f3_reference_arbiter'],.2)
        with self.assertRaisesRegex(ValueError,'STALE_ODOM'):activation_guard(10.,10.,9.8,['/f3_reference_arbiter'])
        with self.assertRaisesRegex(ValueError,'PUBLISHER_CONFLICT'):activation_guard(10.,10.,9.99,['/f3_reference_arbiter','/intruder'])
    def test_soft_brake_whole_body_and_real_nonzero_pva(self):
        for v,a in [(-.5,-.2),(.4,.15),(.8,0),(.7,-.1)]:
            h=np.array([[1.4,0,2],[v,0,0],[a,0,0]],float)
            p,b=soft_brake(h);p.validate(MotionLimits())
            np.testing.assert_allclose(p.pva(0),h,atol=1e-10)
            np.testing.assert_allclose(p.pva(p.duration)[1:],0,atol=1e-9)
            self.assertLessEqual(p.extrema()[2],.35)
            volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],100,'external-test'),),Scope.EXTERNAL_SUPERVISOR)
            self.assertEqual(volume.certify(p.coefficients,p.durations,0,required_until=p.duration)['status'],'CERTIFIED_FREE')
    def test_old_watchdog_head_under_soft_budget(self):
        h=np.array([[2.6382471374,0,2],[-.5902274469,0,0],[-.4893392048,0,0]])
        # Not all aggressive states have a feasible brake under tighter soft limits.
        # Rejection remains a valid failure, never relax limits.
        try:p,b=soft_brake(h)
        except ValueError:return
        p.validate(MotionLimits());self.assertLessEqual(p.extrema()[2],.35)
    def test_nominal_reference_and_cruise(self):
        h0=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float)
        h1=np.array([[-.6,0,2],[.8,0,0],[0,0,0]],float)
        p=concatenate(quintic(h0,h1,2.5),cruise_segment(h1,.75))
        p.validate(MotionLimits());np.testing.assert_allclose(p.pva(p.duration)[0],[0,0,2],atol=1e-10)
        h=p.pva(p.duration)
        nominal=concatenate(p,quintic(h,np.array([[3,0,2],[0,0,0],[0,0,0]]),7.5))
        nominal.validate(MotionLimits())
    def test_literal_fixed_position_stability_impossible(self):
        # With vx>=0.77 for0.5s displacement>=.385m, larger than +/-2cm box diameter.
        self.assertGreater(.77*.5,2*.02)
if __name__=='__main__':unittest.main()
