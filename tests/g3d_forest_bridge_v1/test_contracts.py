"""Contract tests for the new forest bridge; no optimizer, no model selection."""
import unittest,json,tempfile,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from query_minco.evaluation.forest_adapter_v1 import ForestAdapter,compact_source,verify_files
from query_minco.evaluation.mirror_holdout_v1 import sha
from query_minco.trajectory.state_goal_reference_v1 import reference_raw
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import evaluate
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
R=Path(__file__).resolve().parents[2]
O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
class ForestContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.index=json.loads((O/'dataset/FOREST_INDEX.json').read_text())
  cls.maps=json.loads((O/'dataset/MAP_INDEX.json').read_text())
  cls.results=json.loads((O/'FOREST_RESULTS.json').read_text())
  cls.arrays=np.load(O/'FOREST_TRAJECTORIES.npz')
 def test_map_seed_isolation_and_counts(self):
  self.assertEqual(len(self.maps),6);self.assertEqual(len({m['seed'] for m in self.maps}),6)
  self.assertEqual(Counter(m['density'] for m in self.maps),dict(sparse=2,medium=2,dense=2))
  q=json.loads((O/'dataset/FOREST_QUALIFICATION.json').read_text())
  self.assertEqual(q['planned_snapshots'],len(self.index)+sum(x['planned_snapshots'] for x in q['excluded']))
  self.assertEqual(q['independently_feasible'],20);self.assertEqual(q['undetermined'],2)
  self.assertEqual(q['infeasible'],0)
  for row in self.index:
   m=json.loads(Path(row['manifest']).read_text());self.assertEqual(m['split'],'forest_test')
   self.assertFalse(m['teacher_loaded'])
 def test_exact_native_occupancy_all_maps(self):
  for m in self.maps:
   folder=Path(m['map_manifest']).parent
   proof=json.loads((folder/'COMPACTION_PROOF.json').read_text())
   full=np.fromfile(folder/'FULL_SOURCE_COUNTS.i32','<i4')
   compact=np.fromfile(folder/'COMPACT_COUNTS.i32','<i4')
   self.assertTrue(np.array_equal(full>0,compact>0));self.assertEqual(int((full>0).sum()),proof['occupied_voxels'])
   self.assertEqual(hashlib.sha256(np.asarray(full>0,dtype=np.uint8).tobytes()).hexdigest(),proof['mask_sha256'])
   self.assertTrue(proof['min_max_preserved']);self.assertEqual(proof['threshold'],0)
   manifest=json.loads(Path(m['map_manifest']).read_text());verify_files(manifest['files_sha256']);verify_files(manifest['yopo_sources_sha256'])
 def test_compaction_synthetic_extrema_and_duplicate_cells(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cloud=np.array([[0,0,0],[.01,.01,.01],[.5,.5,.5],[.51,.51,.51],[.99,.99,.99]],'<f4')
   cloud.tofile(p/'raw');counts,meta,proof=compact_source(p/'raw',p)
   c=np.fromfile(p/'compact_cloud.f32','<f4').reshape(-1,3)
   np.testing.assert_array_equal(c.min(0),cloud.min(0));np.testing.assert_array_equal(c.max(0),cloud.max(0))
   self.assertTrue(proof['positive_occupancy_exact']);self.assertLess(len(c),len(cloud))
 def test_strict_sha_and_schema_rejection(self):
  row=self.index[0]
  with self.assertRaisesRegex(ValueError,'FOREST_MANIFEST_SHA'):
   ForestAdapter.load_snapshot(row['manifest'],'0'*64)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.json';p.write_text(json.dumps(dict(schema='wrong',component_id='wrong')))
   with self.assertRaisesRegex(ValueError,'FOREST_SCHEMA'):ForestAdapter.load_snapshot(p,sha(p))
   with self.assertRaisesRegex(ValueError,'FOREST_SOURCE_SHA'):verify_files({str(p):'0'*64})
 def test_render_depth_mask_pose_geometry_and_unknown(self):
  for row in self.index:
   batch=ForestAdapter.load_snapshot(row['manifest'],row['manifest_sha256'])
   self.assertEqual(tuple(batch.depth.shape),(1,2,128,160));self.assertEqual(tuple(batch.state.shape),(1,12))
   self.assertTrue(torch.isfinite(batch.depth).all());self.assertTrue(torch.isfinite(batch.state).all())
   depth=batch.depth[0];mask=depth[1]
   self.assertTrue(((mask==0)|(mask==1)).all());self.assertTrue((depth[0][mask==0]==0).all())
   self.assertTrue((mask[:19]==0).all());self.assertTrue((mask[109:]==0).all())
   m=json.loads(Path(row['manifest']).read_text());pd=Path(m['pose_folder']);f=Path(m['map_folder'])
   cam=json.loads((pd/'CAMERA_BINDING.json').read_text());self.assertLessEqual(cam['independent_replay_max_error_m'],cam['replay_tolerance_m'])
   g=batch.geometry[0];np.testing.assert_array_equal(g.rotation,np.eye(3))
   native=np.fromfile(pd/'render/occupancy_counts.i32','<i4')
   np.testing.assert_array_equal(native,np.fromfile(f/'COMPACT_COUNTS.i32','<i4'))
   unknown=np.array(g.roi[1])+1
   local=(unknown-g.origin)@g.rotation
   with self.assertRaisesRegex(ValueError,'UNKNOWN_OUTSIDE_CERTIFIED_ROI'):g.sdf(torch.tensor(local[None],dtype=torch.float64))
 def test_witness_qualification_is_prior_and_not_infeasibility_claim(self):
  for row in self.index:
   m=json.loads(Path(row['manifest']).read_text());q=json.loads((Path(m['folder'])/'PROBLEM_QUALIFICATION.json').read_text())
   feasible=any(w['hard']['passed'] for w in q['witnesses'])
   self.assertEqual(row['classification']=='CERTIFIED_FEASIBLE',feasible);self.assertFalse(q['certified_infeasible'])
   state=np.fromfile(Path(m['folder'])/'state12.f32','<f4')
   for w in q['witnesses']:
    self.assertFalse(w['used_as_training_label']);c=np.array(w['coefficients']);dt=np.array(w['durations'])
    np.testing.assert_allclose(dt,np.full(3,2/3),atol=0,rtol=0)
    np.testing.assert_allclose(evaluate(c[0],0,1),state[:3],atol=1e-10)
    np.testing.assert_allclose(evaluate(c[0],0,2),state[3:6],atol=1e-10)
    for order in range(5):
     for i in range(2):np.testing.assert_allclose(evaluate(c[i],dt[i],order),evaluate(c[i+1],0,order),atol=1e-8)
 def test_original_decode_reference_boundary_and_invalid_inputs(self):
  batch=ForestAdapter.load_snapshot(self.index[0]['manifest'],self.index[0]['manifest_sha256'])
  dec=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1);minco=MincoAdapter(num_candidates=1)
  for kind in ['fixed_forward','state_goal_quintic']:
   raw=reference_raw(batch.state,kind);self.assertEqual(tuple(raw.shape),(1,1,21));self.assertTrue((raw[...,9:]==0).all())
   p=dec(raw,batch.state);tr=minco(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
   c=tr.coeffs[0,0].detach().numpy();dt=tr.durations[0,0].numpy()
   for order in range(3):
    np.testing.assert_allclose(evaluate(c[0],0,order),p.head_pva[0,0,order].numpy(),atol=2e-5)
    np.testing.assert_allclose(evaluate(c[-1],dt[-1],order),p.tail_pva[0,0,order].numpy(),atol=2e-5)
   for bad in [np.array([0,1,1]),np.array([-1,1,1]),np.array([np.nan,1,1]),np.array([np.inf,1,1]),np.ones((1,3)),np.ones(2)]:
    h=validate_polynomial(c,bad,batch.geometry[0],batch.state[0,6:9].numpy());self.assertFalse(h['passed'])
  for s in [torch.zeros(12),torch.zeros(1,12,dtype=torch.float64),torch.full((1,12),float('nan'))]:
   with self.assertRaisesRegex(ValueError,'REFERENCE_STATE'):reference_raw(s,'fixed_forward')
  with self.assertRaisesRegex(ValueError,'REFERENCE_GOAL'):reference_raw(torch.zeros(1,12),'state_goal_quintic')
  with self.assertRaisesRegex(ValueError,'REFERENCE_KIND'):reference_raw(batch.state,'oracle')
  bad=batch.state.clone();bad[0,6:9]=torch.tensor([-2.,0,0])
  with self.assertRaisesRegex(ValueError,'OUTSIDE_ORIGINAL_A_SUPPORT'):reference_raw(bad,'state_goal_quintic')
 def test_frozen_eval_denominator_raw_pva_solver_and_no_updates(self):
  d=self.results;self.assertEqual(d['network_updates'],0);self.assertEqual(d['leaf_updates'],0)
  self.assertTrue(d['known_feasible_denominator_fixed_before_eval'])
  self.assertEqual(len(d['records']),88)
  for ident in d['checkpoint_identity']:self.assertEqual(sha(ident['path']),ident['file_sha256'])
  for row in d['records']:
   self.assertEqual(np.array(row['raw21']).shape,(1,1,21))
   self.assertTrue(np.isfinite(row['loss']));self.assertEqual(row['hard_task'],row['hard']['passed'] and row['hard']['goal']['passed'])
   np.testing.assert_array_less(row['hard']['independent_minco_max_abs_pva'],[.002,.01,.05])
   self.assertEqual(row['hard']['reference_radius_m'],.12);self.assertEqual(row['hard']['margin_m'],.03)
  for method,groups in d['summary'].items():
   known=[r for r in d['records'] if r['method']==method and r['preclassification']=='CERTIFIED_FEASIBLE']
   self.assertEqual(groups['CERTIFIED_FEASIBLE']['n'],len(known))
   self.assertEqual(groups['CERTIFIED_FEASIBLE']['hard_task'],sum(r['hard_task'] for r in known))
   self.assertEqual(groups['FEASIBILITY_UNDETERMINED']['n'],2)
if __name__=='__main__':unittest.main(verbosity=2)
