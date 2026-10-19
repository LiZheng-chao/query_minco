import unittest
import numpy as np
from query_minco.p2b.diagnostics import attention_views,pairwise_distances,unique_count,toy_checks
from query_minco.p2b.toy_data import sample_for_seed


class P2BVisualizationTests(unittest.TestCase):
    def test_attention_excludes_state_token_and_preserves_flatten_order(self):
        a=np.arange(2*4*6*81,dtype=np.float32).reshape(2,4,6,81)
        visual,state=attention_views(a)
        self.assertEqual(visual.shape,(2,4,6,8,10))
        self.assertEqual(state.shape,(2,4,6))
        self.assertEqual(visual[1,2,3,2,4],a[1,2,3,24])
        self.assertEqual(state[1,2,3],a[1,2,3,80])
        with self.assertRaises(ValueError):attention_views(a[...,:80])

    def test_pairwise_duplicates_and_unique_semantics(self):
        paths=np.zeros((6,32,3));paths[3:,:,0]=1.
        distances=pairwise_distances(paths)
        self.assertEqual(unique_count(distances,.15),2)
        np.testing.assert_allclose(distances,distances.T)
        np.testing.assert_allclose(np.diag(distances),0)

    def test_plane_and_kinematic_checks_are_independent_of_network_score(self):
        sample=sample_for_seed(32000,"train")
        result=toy_checks(sample.teacher_coeffs.numpy(),sample.teacher_durations.numpy(),sample.plane_x_m)[0]
        import json
        json.dumps(result)
        self.assertTrue(result["toy_geometry_pass"])
        self.assertTrue(result["toy_kinematics_pass"])
        self.assertAlmostEqual(result["plane_clearance_m"],sample.metadata["teacher_min_plane_clearance_m"],places=4)
