import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from query_minco.p2b.toy_data import sample_for_seed,render_plane,create_dataset
from query_minco.p2b.utf8_io import write_utf8


class P2BContractTests(unittest.TestCase):
    def test_unicode_round_trip_and_no_overwrite(self):
        text="\u6570\u636e\u5ba1\u8ba1\uff1a\u5750\u6807\u3001\u65f6\u95f4\u4e0e\u8f68\u8ff9\u3002"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"report.md"
            write_utf8(p,text,require_chinese=True)
            self.assertEqual(p.read_bytes().decode("utf-8"),text)
            self.assertNotIn("?",p.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):write_utf8(p,text)
            with self.assertRaises(ValueError):write_utf8(Path(tmp)/"bad.md","???",require_chinese=True)

    def test_plane_render_matches_known_rays(self):
        depth,valid=render_plane(9.)
        for u,v in ((0,0),(79,63),(159,127)):
            ray=np.array([1.,-(u-79.5)/80.,-(v-63.5)/80.])
            point=ray*depth[v,u]
            self.assertAlmostEqual(point[0],9.)
            self.assertAlmostEqual(79.5-80*point[1]/point[0],u)
            self.assertAlmostEqual(63.5-80*point[2]/point[0],v)
        self.assertTrue(valid.all())

    def test_state_teacher_depth_and_time_contract(self):
        torch.set_num_threads(1)
        for seed in (32000,32011,32048):
            s=sample_for_seed(seed,"train")
            self.assertEqual(tuple(s.depth_input.shape),(2,128,160))
            self.assertEqual(tuple(s.state.shape),(12,))
            self.assertEqual(tuple(s.teacher_coeffs.shape),(1,3,3,6))
            np.testing.assert_allclose(s.teacher_coeffs[0,0,:,0].numpy(),0.,atol=1e-7)
            torch.testing.assert_close(s.teacher_coeffs[0,0,:,1],s.state[:3],atol=1e-6,rtol=0)
            torch.testing.assert_close(2*s.teacher_coeffs[0,0,:,2],s.state[3:6],atol=1e-6,rtol=0)
            torch.testing.assert_close(s.state[6:9],s.state[:3]*s.teacher_durations.sum(),atol=1e-6,rtol=0)
            self.assertGreater(s.metadata["teacher_min_plane_clearance_m"],1.)
            self.assertEqual(s.metadata["depth_time_s"],s.metadata["teacher_start_time_s"])

    def test_scene_seed_split_is_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=create_dataset(Path(tmp)/"data",train_count=3,val_count=2)
            train={s.scene_seed for s in data if s.role=="train"}
            val={s.scene_seed for s in data if s.role=="val"}
            self.assertFalse(train&val)
            self.assertEqual(len(data),5)
            same=sample_for_seed(data[0].scene_seed,data[0].role)
            self.assertTrue(torch.equal(same.state,data[0].state))
            self.assertTrue(torch.equal(same.depth_input,data[0].depth_input))
