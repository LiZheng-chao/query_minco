"""CPU checks: native camera ray identity, Mask preservation, frozen output regression."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from query_minco.inference.shadow_camera30 import extract
class Contracts(unittest.TestCase):
    def test_native_ray_grid_and_value_provenance(self):
        full=np.arange(512*800,dtype=np.float32).reshape(512,800);q,y=extract(full)
        self.assertEqual(q.shape,(128,160));self.assertEqual(y.shape,(90,160))
        for w,h,f,cx,cy,ox,oy,step in [(160,128,100.,79.5,63.5,82,0,4),(160,90,80.,80.,45.,0,29,5)]:
            u=np.arange(w,dtype=np.float32);v=np.arange(h,dtype=np.float32)
            np.testing.assert_array_equal((u-cx)/f,(ox+step*u-400)/400)
            np.testing.assert_array_equal((v-cy)/f,(oy+step*v-254)/400)
        full[0,82]=20.;full[4,86]=np.nan;q,_=extract(full)
        self.assertEqual(q[0,0],20.);self.assertTrue(np.isnan(q[1,1]))
    def test_frozen_numerical_regression_and_goal(self):
        from query_minco.inference.shadow import FrozenPair,FROZEN,sha,make_state
        from query_minco.inference.ros_shadow30 import TimedFrozenPair
        a=FrozenPair(sha(FROZEN));b=TimedFrozenPair(sha(FROZEN))
        state,g=make_state(np.array([0.,0.,2.]),np.eye(3),np.array([.8,0,0]),np.zeros(3),np.array([30.,0,2.]))
        self.assertAlmostEqual(float(np.linalg.norm(state[6:9])),2.4,places=5)
        depth=np.stack([np.full((128,160),.25,dtype=np.float32),np.ones((128,160),dtype=np.float32)])
        x=a.forward(depth,state);y=b.forward(depth,state)
        for label in x:
            for name in x[label]:np.testing.assert_array_equal(x[label][name],y[label][name])
            self.assertGreater(b.last_timings[label],0)
if __name__=='__main__':unittest.main()
