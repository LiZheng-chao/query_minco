import unittest
import numpy as np
from query_minco.inference.low_speed_reference import ramp,reference,braking
from query_minco.inference.online_selector import select,lawful_features,Observability
from query_minco.evaluation.validation import norm_max
from pathlib import Path
class TestLowSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z=np.load(Path(__file__).resolve().parents[2]/'research_outputs/rsps_forest_shadow_v1/forest85s_shadow001/frame_000010.npz')
        cls.d=cls.z['depth_input'];cls.s=cls.z['state'];cls.o={k[3:]:cls.z[k] for k in cls.z.files if k.startswith('C1_')}
    def test_reference_pva(self):
        for t in np.linspace(.01,3.99,101):
            h=1e-5;p,v,a=ramp(t)
            self.assertAlmostEqual((ramp(t+h)[0]-ramp(t-h)[0])/(2*h),v,places=7)
            self.assertAlmostEqual((ramp(t+h)[1]-ramp(t-h)[1])/(2*h),a,places=7)
        self.assertTrue(np.allclose(reference(4),[1.6,.8,0]))
        self.assertTrue(np.allclose(braking(0,5,.8),[5,.8,0]))
        self.assertTrue(np.allclose(braking(4,5,.8),[6.6,0,0]))
    def test_unknown_is_never_permission(self):
        r=select(self.d,self.s,self.o,.01)
        self.assertIsNone(r['selected_query']);self.assertEqual(r['certified_count'],0)
        self.assertTrue(all(c['observability']=='UNKNOWN' for c in r['candidates']))
    def test_raw_scores_ignored(self):
        a=select(self.d,self.s,self.o,.01);out=dict(self.o)
        out['raw']=self.o['raw'].copy();out['raw'][:,19:]=np.arange(12).reshape(6,2)*10000
        self.assertEqual(a,select(self.d,self.s,out,.01))
    def test_stale_nan_and_bad_durations(self):
        for age in [-.1,.251,float('nan')]:
            self.assertEqual(select(self.d,self.s,self.o,age)['input_gate'],'BLOCKED')
        s=self.s.copy();s[0]=np.nan
        self.assertEqual(select(self.d,s,self.o,.01)['input_gate'],'BLOCKED')
        out=dict(self.o);out['durations']=np.zeros((6,3))
        self.assertTrue(all('TIME_CONTRACT' in c['reasons'] for c in select(self.d,self.s,out,.01)['candidates']))
    def test_head_and_piece_discontinuity(self):
        out=dict(self.o);out['coefficients']=self.o['coefficients'].copy()
        out['coefficients'][0,0,0,0]+=1.
        out['coefficients'][1,1,0,0]+=1.
        r=select(self.d,self.s,out,.01)
        self.assertIn('HEAD_PVA',r['candidates'][0]['reasons'])
        self.assertIn('PVA_CONTINUITY_0',r['candidates'][1]['reasons'])
        self.assertIsNone(r['selected_query'])
    def test_observed_rays_do_not_certify_body(self):
        d=self.d.copy();d[0]=.99;d[1]=1.
        r=select(d,self.s,self.o,.01)
        self.assertEqual(r['certified_count'],0)
        self.assertTrue(all(c['observability']=='UNKNOWN' for c in r['candidates']))
    def test_interior_extrema(self):
        # v=4t(1-t), v endpoints=0, exact interior max=1.
        c=np.zeros((3,6));c[0,2]=2;c[0,3]=-4/3
        self.assertAlmostEqual(norm_max(c,1.,1),1.,places=9)
    def test_determinism_and_no_gt_inputs(self):
        self.assertEqual(select(self.d,self.s,self.o,.01),select(self.d,self.s,self.o,.01))
        self.assertEqual(lawful_features(self.d,self.s,self.o).shape,(6,65,27))
if __name__=='__main__':unittest.main(verbosity=2)
