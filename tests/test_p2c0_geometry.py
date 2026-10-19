import unittest
import numpy as np
from query_minco.p2c0.config import CONFIG
from query_minco.p2c0.geometry import make_scene,raycast,camera_rays,point_box_distance
from query_minco.p2c0.validation import bernstein_controls,split_controls,certify_piece,evaluate
from query_minco.p2c0.dataset import scene_specs,model_inputs


class P2C0GeometryTests(unittest.TestCase):
    def test_camera_backprojection_axes_and_nearest_surface(self):
        scene={"obstacles":[{"min":[2.,-3.,-3.],"max":[2.1,3.,3.]},
                            {"min":[4.,-3.,-3.],"max":[4.1,3.,3.]}]}
        depth,valid,network,ids=raycast(scene)
        self.assertTrue(valid.all());np.testing.assert_allclose(depth,2.)
        self.assertTrue((ids==0).all())
        points=camera_rays()*depth[...,None]
        self.assertLess(float(point_box_distance(points,scene["obstacles"][0]).max()),1e-10)
        rays=camera_rays()
        self.assertGreater(rays[64,0,1],0);self.assertLess(rays[64,159,1],0)
        self.assertGreater(rays[0,80,2],0);self.assertLess(rays[127,80,2],0)
        np.testing.assert_array_equal(network[1],valid)

    def test_invalid_no_hit_and_near_clip(self):
        depth,valid,network,ids=raycast({"obstacles":[]})
        self.assertFalse(valid.any());self.assertTrue((network==0).all());self.assertTrue((ids==-1).all())
        box={"min":[.01,-1.,-1.],"max":[.02,1.,1.]}
        self.assertFalse(raycast({"obstacles":[box]})[1].any())
        self.assertFalse(raycast({"obstacles":[{"min":[21.,-100.,-100.],"max":[22.,100.,100.]}]})[1].any())

    def test_sphere_corner_and_interior_curve_collision(self):
        box={"min":[.9,-.1,-.1],"max":[1.1,.1,.1]}
        self.assertAlmostEqual(float(point_box_distance([.8,.2,0.],box)),np.sqrt(.02))
        c=np.zeros((3,6));c[0,1]=2.
        self.assertGreater(float(point_box_distance(evaluate(c,0),box)),.15)
        self.assertGreater(float(point_box_distance(evaluate(c,1),box)),.15)
        self.assertFalse(certify_piece(c,1.,[box])["passed"])
        c[1,0]=.5
        self.assertTrue(certify_piece(c,1.,[box])["passed"])

    def test_bernstein_subdivision_matches_polynomial(self):
        rng=np.random.RandomState(4);c=rng.randn(3,6);T=.7
        controls=bernstein_controls(c,T);left,right=split_controls(controls)
        np.testing.assert_allclose(controls[0],evaluate(c,0))
        np.testing.assert_allclose(controls[-1],evaluate(c,T))
        np.testing.assert_allclose(left[-1],evaluate(c,T/2))
        np.testing.assert_allclose(right[0],evaluate(c,T/2))

    def test_grouped_unseen_geometry_splits_and_gt_exclusion(self):
        rows=scene_specs();self.assertEqual(len(rows),75)
        groups={}
        for row in rows:groups.setdefault(row["group_id"],set()).add(row["split"])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        for row in rows:
            if row["kind"]=="left_only":
                other=next(r for r in rows if r["id"]==row["id"].replace("left_only","right_only"))
                self.assertEqual(row["split"],other["split"])
            if row["kind"]=="staggered_double":self.assertEqual(row["split"],"test")
        arrays={"depth_input":np.zeros((2,128,160)),"state":np.array(CONFIG["state"]),
                "GT":object(),"seed":100,"scene":object(),"teacher_coeffs":object()}
        self.assertEqual(set(model_inputs(arrays)),{"depth_input","state"})
        self.assertNotIn("seed",model_inputs(arrays))
        with self.assertRaises(ValueError):model_inputs({"depth_input":np.zeros((1,128,160)),"state":np.zeros(12)})
