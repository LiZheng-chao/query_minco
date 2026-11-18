import unittest,copy,json,ast
from pathlib import Path
import numpy as np
from experiments.sim_dev_oracle_v1.guardian import Guardian
from experiments.sim_dev_oracle_v1.planner import CandidatePlanner
from experiments.sim_dev_oracle_v1.evidence import validate_oracle
from experiments.sim_dev_oracle_v1.manual import qualification
from query_minco.integration.yopo_execution_v4.trajectory import build,evaluate
from query_minco.integration.f0_continuity_v1.execution_evidence import validate_observed_authorization
class Empty:
 map_id='SYNTHETIC_NOT_FLIGHT';
 def clearance(self,p):return 10.
class Blocked(Empty):
 def clearance(self,p):return 0.
class Safety(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=build(np.array([.25,-.25,2.]),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.85,-.25,2.]),100.);cls.state=dict(p=[.25,-.25,2.],v=[0,0,0],a=[0,0,0],q=[0,0,0,1])
 def proof(self):return Guardian(Empty(),'N').check(self.plan.record,self.state,100.,100.,1)
 def test_prefix_brake_positive_synthetic_only(self):
  p=self.proof();self.assertFalse(p['strict_depth_qualified']);self.assertEqual(p['max_delay'],.35);self.assertGreater(p['brake_extra_margin'],0)
 def test_continuous_brake_ends_stopped(self):
  p=self.proof();c=np.array(p['brake_coefficients']);T=p['brake_duration'];np.testing.assert_allclose(evaluate(c,T,1),0,atol=1e-10);np.testing.assert_allclose(evaluate(c,T,2),0,atol=1e-10)
 def test_robust_brake_limits(self):
  p=self.proof();self.assertTrue(all(x<=y for x,y in zip(p['robust_brake_maxima'],[1,8,100])))
 def test_new_gt_obstacle_rejected(self):
  with self.assertRaisesRegex(ValueError,'ORACLE_(PREFIX_OR_BRAKE_OCCUPIED|FAULT_STOP_REACHABILITY)'):Guardian(Blocked(),'N').check(self.plan.record,self.state,100.,100.,1)
 def test_stale_depth_rejected(self):
  with self.assertRaisesRegex(ValueError,'STALE'):Guardian(Empty(),'N').check(self.plan.record,self.state,100.21,100.,1)
 def test_tracking_deviation_rejected(self):
  s=copy.deepcopy(self.state);s['p'][0]+=.16
  with self.assertRaisesRegex(ValueError,'TRACKING'):Guardian(Empty(),'N').check(self.plan.record,s,100.,100.,1)
 def test_wrong_delay_not_accepted(self):
  with self.assertRaisesRegex(ValueError,'CONTRACT'):Guardian(Empty(),'N').check(self.plan.record,self.state,100.,100.,1,delay=0.)
 def test_expired_lease_rejected(self):
  g=Guardian(Empty(),'N');p=g.check(self.plan.record,self.state,100.,100.,1);self.assertFalse(g.valid(p,self.plan.id,100.101,1,'N'))
 def test_wrong_revision_rejected(self):
  g=Guardian(Empty(),'N');p=g.check(self.plan.record,self.state,100.,100.,1);self.assertFalse(g.valid(p,self.plan.id,100.01,2,'N'))
 def test_wrong_nonce_rejected(self):
  g=Guardian(Empty(),'N');p=g.check(self.plan.record,self.state,100.,100.,1);self.assertFalse(g.valid(p,self.plan.id,100.01,1,'OTHER'))
 def test_wrong_plan_rejected(self):
  g=Guardian(Empty(),'N');p=g.check(self.plan.record,self.state,100.,100.,1);self.assertFalse(g.valid(p,'BAD',100.01,1,'N'))
 def test_tampered_proof_rejected(self):
  g=Guardian(Empty(),'N');p=g.check(self.plan.record,self.state,100.,100.,1);p['expires']+=1;self.assertFalse(g.valid(p,self.plan.id,100.01,1,'N'))
 def test_strict_validator_cannot_accept_oracle(self):
  p=self.proof();command=dict(plan_id=self.plan.id,authorization=dict(type='SIM_DEV_ORACLE_PREFIX_BRAKE',oracle=p));self.assertIsNotNone(validate_observed_authorization(command,self.plan.record))
 def test_guardian_proof_not_a_fake_qualification(self):
  with self.assertRaises(PermissionError):qualification(Path('research_outputs/sim_dev_oracle_20261118_v1/runtime'),{})
 def test_planner_has_no_gt_inputs(self):
  p=Path('experiments/sim_dev_oracle_v1/planner.py');tree=ast.parse(p.read_text());names=[getattr(x,'module','') or '' for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)];self.assertFalse(any('truth' in x or 'guardian' in x for x in names))
 def test_real_cuda_counterexample(self):
  d=json.loads(Path('research_outputs/sim_dev_oracle_20261118_v1/CUDA_FIXTURE_RESULT_v4.json').read_text());cases={x['name']:x for x in d['cases']};self.assertEqual(cases['empty_ray']['depth'],20);self.assertEqual(cases['thin_grazing_occupied_voxel_missed_by_sampling']['depth'],20);self.assertLess(cases['near_tree_occupied_voxel']['depth'],1)
 def test_generated_adapters_keep_original_imports_separate(self):
  root=Path('research_outputs/sim_dev_oracle_20261118_v1/generated_v2');s=(root/'supervisor.py').read_text();self.assertIn('from experiments.sim_dev_oracle_v1.bridge import SupervisorBridge',s);self.assertIn("runtime.get('experiment_mode')!='SIM_DEV_ORACLE'",s);self.assertIn('from query_minco.integration.depth_minco_t2_v1.ros_bridge import SupervisorBridge',Path('ros/g4cp7b_fix2_v1/supervisor.py').read_text())
 def test_sim_evidence_rejects_wrong_mode(self):
  p=self.proof();p['mode']='STRICT_DEPTH';c=dict(authorization=dict(oracle=p,geometry='SIM_DEV_ORACLE_PREFIX_BRAKE'),plan_id=self.plan.id,certificate_id=p['identity']);self.assertEqual(validate_oracle(c,self.plan.record),'ORACLE_MODE')
 def test_latest_hit_cache_matches_exhaustive_group_queries(self):
  from experiments.sim_dev_oracle_v1.planner import LatestHitSpace
  pose=dict(p=[.25,-.25,2.],q=[0,0,0,1],frame='world');space=LatestHitSpace(np.full((90,160),5,np.float32),pose,100.)
  for point in [np.array([x,-.25,2.]) for x in [.25,.45,.65,.85]]:
   expected=all(tree.query(point,p=np.inf)[0]>radius+.33 for radius,tree,lo,hi in space.groups);self.assertEqual(space.ball_free(point,.33),expected);self.assertEqual(space.ball_free(point,.33),expected)
 def test_batch_sampler_matches_original_polynomial(self):
  from query_minco.integration.observed_space_v2.core import sample_batch
  ts=np.linspace(0,1,31);np.testing.assert_allclose(sample_batch(self.plan,ts),np.stack([self.plan.sample(t) for t in ts]),atol=1e-10,rtol=0)
 def test_fresh_invalid_depth_rejected(self):
  from experiments.sim_dev_oracle_v1.planner import validate_input
  with self.assertRaisesRegex(ValueError,'MISSING'):validate_input(np.full((90,160),np.nan,np.float32),dict(p=[.25,-.25,2.],q=[0,0,0,1]),100.)
 def test_new_finite_near_obstacle_vetoes(self):
  from experiments.sim_dev_oracle_v1.planner import validate_input
  with self.assertRaisesRegex(ValueError,'OCCUPIED'):validate_input(np.full((90,160),.15,np.float32),dict(p=[.25,-.25,2.],q=[0,0,0,1]),100.,[.25,-.25,2.])
 def test_owner_only_routes_sim_stage_to_sim_reference(self):
  source=Path('research_outputs/sim_dev_oracle_20261118_v1/generated_v2/manual.py').read_text();tree=ast.parse(source);episode=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='episode');assignment=next(x for x in ast.walk(episode) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reference' for t in x.targets));expr=assignment.value.args[1].elts[2];compiled=compile(ast.Expression(expr),'reference_path','eval')
  for stage in ['S','C']:self.assertEqual(eval(compiled,{'stage':stage}),'ros/g4cp7b_fix2_v1/reference.py')
  self.assertTrue(eval(compiled,{'stage':'SIM_DEV_ORACLE'}).endswith('experiments/sim_dev_oracle_v1/reference.py'))
 def test_gt_version_change_revokes_cached_geometry(self):
  import tempfile,os
  with tempfile.TemporaryDirectory() as folder:
   file=Path(folder)/'data';meta=Path(folder)/'meta';file.write_bytes(b'0');meta.write_bytes(b'0');grid=Empty();grid.path=file;grid.meta_path=meta;grid.data_stat=file.stat();grid.meta_stat=meta.stat();g=Guardian(grid,'N');g.check(self.plan.record,self.state,100.,100.,1);file.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'MAP_CHANGED'):g.check(self.plan.record,self.state,100.,100.,1)
 def test_fault_stop_ball_is_checked_separately(self):
  class Narrow(Empty):
   def clearance(self,p):return .70
  with self.assertRaisesRegex(ValueError,'FAULT_STOP_REACHABILITY'):Guardian(Narrow(),'N').check(self.plan.record,self.state,100.,100.,1)
 def test_scene_label_matches_bound_sparse_config(self):
  scene=json.loads(Path('research_outputs/sim_dev_oracle_20261118_v1/runtime_v2/SCENES.json').read_text())['forest'];self.assertIn('tree_dist6',scene['scene_kind']);self.assertEqual(scene['tree_dist'],6);self.assertEqual(scene['spawn'],[.25,-.25,2.]);self.assertIn('forest_dev_sparse_seed3_dist6_fixedbirth_v1',scene['sensor_binary'])
if __name__=='__main__':unittest.main(verbosity=2)
