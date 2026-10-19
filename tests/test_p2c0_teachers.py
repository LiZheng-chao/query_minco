import unittest
import numpy as np
from query_minco.p2c0.geometry import make_scene,raycast
from query_minco.p2c0.teachers import generate_teachers,aligned_distance,construct_teacher
from query_minco.p2c0.validation import validate_teacher
from query_minco.p2c0.dataset import archive_teacher_arrays


class P2C0TeacherTests(unittest.TestCase):
    def test_single_dual_zero_teachers_and_independent_reference(self):
        for kind,M in [("left_only",1),("right_only",1),("dual",2),("narrow",1),("blocked",0),("free",1)]:
            scene=make_scene(kind,1.4,.3,.18);teachers,checks=generate_teachers(scene)
            self.assertEqual(len(teachers),M,(kind,checks))
            for t in teachers:
                self.assertTrue(t["validation"]["passed"])
                self.assertEqual(t["coeffs"].shape,(3,3,6))
                self.assertTrue((t["durations"]>=.12).all())
                self.assertLess(max(t["validation"]["reference_max_abs_pva"]),.01)
            if kind=="dual":
                self.assertEqual({t["side"] for t in teachers},{"left","right"})
                self.assertGreater(min(aligned_distance(*teachers)),.25)
                self.assertTrue(any(a["status"]=="deduplicated" for a in checks["attempts"]))
            if kind=="blocked":
                self.assertTrue(all(a["status"]=="rejected" for a in checks["attempts"]))
                a=archive_teacher_arrays(teachers)
                self.assertEqual(a["teacher_coeffs"].shape,(0,3,3,6))

    def test_counterfactual_state_unchanged_depth_switch_and_wrong_route_collision(self):
        left=make_scene("left_only",1.4,.3,.18);right=make_scene("right_only",1.4,.3,.18)
        ld=raycast(left)[0];rd=raycast(right)[0]
        self.assertGreater(float((np.abs(ld-rd)>.01).mean()),.05)
        t=generate_teachers(left)[0][0]
        self.assertFalse(validate_teacher(t["coeffs"],t["durations"],t["head_pva"],t["tail_pva"],t["inner_pts"],right)["passed"])

    def test_boundary_and_time_failures_are_rejected(self):
        scene=make_scene("dual",1.4,.3,.18);t=generate_teachers(scene)[0][0]
        c=t["coeffs"].copy();c[0,0,0]+=.02
        bad=validate_teacher(c,t["durations"],t["head_pva"],t["tail_pva"],t["inner_pts"],scene)
        self.assertFalse(bad["passed"]);self.assertIn("boundary_0",bad["reasons"])
        d=t["durations"].copy();d[0]=.10
        bad=validate_teacher(t["coeffs"],d,t["head_pva"],t["tail_pva"],t["inner_pts"],scene)
        self.assertIn("time_contract",bad["reasons"])

    def test_clearance_reserve_triggers_refinement_before_acceptance(self):
        scene=make_scene("left_only",1.58,.35,.21)
        t=construct_teacher(scene,.5)
        self.assertTrue(t["validation"]["passed"])
        self.assertGreater(t["validation"]["surface_clearance_lower_bound_m"],.03)
        self.assertGreater(t["validation"]["dense_surface_clearance_m"],.19)
