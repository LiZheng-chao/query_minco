import unittest,tempfile,json
from pathlib import Path
import numpy as np
from query_minco.integration.native_grid_truth_v1.grid import Grid
class Tests(unittest.TestCase):
 def fixture(self,r,data):
  b=r/'g';Path(str(b)+'.u8').write_bytes(data.tobytes())
  Path(str(b)+'.json').write_text(json.dumps(dict(shape=list(data.shape),origin=[0,0,0],resolution=.1)))
  return Grid(b)
 def test_occupied_cell_boundary(self):
  with tempfile.TemporaryDirectory() as d:
   data=np.zeros((30,30,30),np.uint8);data[15,15,15]=1;g=self.fixture(Path(d),data)
   self.assertLess(g.clearance([1.5,1.5,1.5]),0)
   self.assertAlmostEqual(g.clearance([1.3,1.55,1.55]),.2-1e-5,places=5)
 def test_no_drop_all_cells_and_floor(self):
  with tempfile.TemporaryDirectory() as d:
   data=np.zeros((30,30,30),np.uint8);data[0,0,0]=1;data[-1,-1,-1]=1;g=self.fixture(Path(d),data)
   self.assertLess(g.clearance([2.95,2.95,2.95]),0)
   self.assertLess(g.clearance([1,1,.05]),0)
   self.assertLess(g.clearance([-1,1,1]),0)
 def test_map_mutation_denied(self):
  with tempfile.TemporaryDirectory() as d:
   g=self.fixture(Path(d),np.zeros((30,30,30),np.uint8));g.path.write_bytes(b'x')
   with self.assertRaises(ValueError):g.clearance([1,1,1])
 def test_bounded_free_clearance(self):
  with tempfile.TemporaryDirectory() as d:
   g=self.fixture(Path(d),np.zeros((30,30,30),np.uint8))
   self.assertAlmostEqual(g.clearance([1.5,1.5,1.5]),.8-1e-5,places=5)
 def test_no_arm_stability_source(self):
  s=Path('experiments/g4cp7b_fix_v1/manual.py').read_text()
  self.assertIn("if stage!='S' and h['mode']=='ACK'",s)
  self.assertIn("if stage!='S' and h['mode']=='PREPARE'",s)
  self.assertIn("predecessor(a.prior,'S',binding)",s)
 def test_native_render_library_not_reimplemented(self):
  s=Path('research_outputs/g4cp7b_fix_20261112_v1/build_v2/renderer.cpp').read_text()
  self.assertIn('renderDepthImage(grid_map, camera, T_wc, depth_image)',s)
  self.assertNotIn('pcl::toROSMsg(*cloud, output)',s)
