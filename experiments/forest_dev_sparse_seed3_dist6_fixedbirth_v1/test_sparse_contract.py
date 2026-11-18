from pathlib import Path
import unittest,json,hashlib
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
class SparseContract(unittest.TestCase):
 def test_fixed_scene(self):
  d=json.loads((O/'BUILD_IDENTITY.json').read_text());self.assertEqual(d['config_changes'],{'tree_dist':[4,6]});self.assertEqual([d['config']['maze_type'],d['config']['seed'],d['config']['tree_dist']],[5,3,6])
 def test_all_frames_matched_ordered_fixed(self):
  d=json.loads((O/'capture_ordered/INPUTS.json').read_text());rows=d['records'];self.assertGreater(len(rows),1)
  for a,b in zip(rows,rows[1:]):self.assertLess(a['stamp_ns'],b['stamp_ns'])
  for x in rows:
   self.assertEqual(x['pose']['p'],[.25,-.25,2.]);self.assertEqual(x['pose']['q'],[0.,0.,0.,1.]);self.assertEqual(x['K'],[80.,0.,80.,0.,80.,45.,0.,0.,1.]);self.assertEqual(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest(),x['sha256'])
 def test_no_control_publishers(self):
  d=json.loads((O/'capture_ordered/INPUTS.json').read_text());self.assertFalse(d['plant']);self.assertFalse(d['controller']);self.assertFalse(d['ARM'])
  for topic,nodes in d['live_graph'][2][0]:self.assertFalse(any(s in topic.lower() for s in ['position_cmd','so3_cmd','arm','permit']))
if __name__=='__main__':unittest.main(verbosity=2)
