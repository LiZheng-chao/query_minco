"""Native host must reject occupied output before loading or generating any map."""
from pathlib import Path
import unittest,tempfile,subprocess,json,os
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
class GenerationProtection(unittest.TestCase):
 def test_occupied_destination_rejected_without_touching_sentinel(self):
  build=json.loads((O/'build/BUILD_IDENTITY.json').read_text())
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);sentinel=out/'HISTORY_SENTINEL';sentinel.write_bytes(b'preserve-history')
   before=sentinel.stat()
   p=subprocess.run([build['binary'],str(R.parent/'YOPO'),str(O/'PREREGISTRATION.json'),str(out)],capture_output=True,text=True,env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib"))
   print('OCCUPIED_OUTPUT_TRUE_CHILD_EXIT',p.returncode,'STDOUT',repr(p.stdout),'STDERR',repr(p.stderr))
   self.assertNotEqual(p.returncode,0);self.assertIn('EXCLUSIVE_EMPTY_OUT_REQUIRED',p.stderr)
   self.assertEqual(list(out.iterdir()),[sentinel]);self.assertEqual(sentinel.read_bytes(),b'preserve-history')
   self.assertEqual(sentinel.stat().st_mtime_ns,before.st_mtime_ns)
if __name__=='__main__':unittest.main(verbosity=2)
