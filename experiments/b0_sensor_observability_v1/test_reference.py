import sys,unittest,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from experiments.b0_sensor_observability_v1.reference import *
from query_minco.integration.observed_space_v1.map import Frame
class ReferenceSafety(unittest.TestCase):
 def test_cap_only_recovery_is_detected(self):
  g=project(np.array([SPAWN+[.4,0,0]]),dict(p=SPAWN,q=[0,0,0,1]));depth=np.full((90,160),5,np.float32)
  self.assertGreater(g['radius'][0],24);self.assertFalse(evaluate(depth,g,'current')[0][0]);self.assertTrue(evaluate(depth,g,'uncapped_square')[0][0])
 def test_real_native_mask_matches_production(self):
  item=json.loads((BASE/'capture_ordered/INPUTS.json').read_text())['records'][0];depth=np.load(item['path'])['depth'];m=ObservedMap(CFG);t=item['stamp'];m.add(depth,item['pose'],t,t,t);s=m.snapshot(t);ids=np.indices(CFG.shape).transpose(1,2,3,0);centers=(s.origin+(ids+.5)*.1).reshape(-1,3);mask=evaluate(depth,project(centers,item['pose']),'current')[0];self.assertTrue(np.array_equal(mask.reshape(CFG.shape)&(s.states!=2),s.states==1))
 def test_synthetic_minimum_filter_equivalence(self):
  rng=np.random.RandomState(3);depth=rng.uniform(.2,7,(90,160)).astype(np.float32);pose=dict(p=SPAWN,q=[0,0,0,1]);origin=SPAWN-[1.6,1.2,.8];centers=(origin+(np.indices(CFG.shape).transpose(1,2,3,0)+.5)*.1).reshape(-1,3);g=project(centers,pose);reference=evaluate(depth,g,'current')[0];f=Frame(depth,pose,100.,100.,100.,'synthetic','camera');actual=f.carve(centers,origin,CFG,time.perf_counter()+5)[0].ravel();self.assertTrue(np.array_equal(reference,actual))
 def test_outside_fov_cannot_be_free(self):
  g=project(np.array([SPAWN+[.5,0,.5]]),dict(p=SPAWN,q=[0,0,0,1]));depth=np.full((90,160),5,np.float32)
  for mode in ['current','uncapped_square','conservative_rectangle']:self.assertFalse(evaluate(depth,g,mode)[0][0])
 def test_missing_depth_cannot_be_free(self):
  g=project(np.array([SPAWN+[1,0,0]]),dict(p=SPAWN,q=[0,0,0,1]))
  for val in [0.,20.,float('nan'),float('inf')]:
   for mode in ['current','uncapped_square','conservative_rectangle']:self.assertFalse(evaluate(np.full((90,160),val,np.float32),g,mode)[0][0])
 def test_near_obstacle_cannot_be_free(self):
  g=project(np.array([SPAWN+[1,0,0]]),dict(p=SPAWN,q=[0,0,0,1]));depth=np.full((90,160),.3,np.float32)
  for mode in ['current','uncapped_square','conservative_rectangle']:self.assertFalse(evaluate(depth,g,mode)[0][0])
 def test_no_birth_or_free_override_of_occupied(self):
  states=np.full(CFG.shape,2,np.uint8);s=Snapshot(states,SPAWN-[1.6,1.2,.8],CFG,[dict(id='x',stamp=100.)],100.,1,lambda:1);ref=make_snapshot(s,np.ones(states.size,bool),100.);self.assertTrue((ref.states==2).all());self.assertFalse(ref.ball(SPAWN,.3,100.)['passed'])
 def test_snapshot_expiry_is_not_extended(self):
  s=Snapshot(np.zeros(CFG.shape,np.uint8),SPAWN-[1.6,1.2,.8],CFG,[dict(id='x',stamp=100.)],100.,1,lambda:1);ref=make_snapshot(s,np.zeros(s.states.size,bool),100.)
  with self.assertRaises(Exception):ref.current(100.201)
 def test_no_native_single_view_success_fabricated(self):
  result=json.loads((O/'B0_EXACT_RAY_REFERENCE.json').read_text());self.assertTrue(result['voxel_diagnosis']['reference_current_free_mask_equals_production'])
  for values in result['certified_counts'].values():self.assertEqual(values,{'0.6':0,'0.8':0})
 def test_virtual_views_do_not_claim_execution(self):
  result=json.loads((O/'B0_SENSOR_OBSERVABILITY_RESULT.json').read_text());self.assertFalse(result['actual_body_turning_proven']);self.assertFalse(result['expiry_tests'][1]['old_views_retained']);self.assertTrue(result['expiry_tests'][0]['rejected'])
if __name__=='__main__':unittest.main(verbosity=2)
