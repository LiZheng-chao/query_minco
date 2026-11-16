import ast,time,unittest,tempfile,json
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v1.map import ObservedMap,Config,CertificationError
from query_minco.integration.observed_space_v1.space import MapSpace
from query_minco.integration.observed_space_v1.admission import require_initial_coverage
from query_minco.integration.observed_space_v1.evidence import verify_e0
class InterfaceTests(unittest.TestCase):
 def frame(self,m,t=10.,x=0.,depth=5.):
  m.add(np.full((90,160),depth,np.float32),dict(p=[x,0,0],q=[0,0,0,1]),t,t,t)
  return m.snapshot(t)
 def test_source_gap_erases_history(self):
  m=ObservedMap();s=self.frame(m);self.frame(m,10.3)
  self.assertEqual(len(m.frames),1)
  with self.assertRaises(CertificationError):s.current(10.3)
 def test_dynamic_mode_never_fuses_historical_free(self):
  m=ObservedMap(Config(static_scene=False));self.frame(m,10.,-1.)
  s=self.frame(m,10.1,0.);self.assertFalse(s.ball([0,0,0],.3,10.1)['passed'])
 def test_rolling_rebase_preserves_world_obstacle(self):
  m=ObservedMap();a=self.frame(m,depth=1.5);self.assertGreater(a.ball([1.5,0,0],.1,10.)['occupied'],0)
  b=self.frame(m,10.1,.2,1.3);self.assertGreater(b.ball([1.5,0,0],.1,10.1)['occupied'],0)
 def test_config_cannot_raise_caps_or_nan(self):
  for cfg in [Config(max_cells=2000000),Config(update_budget_s=float('nan')),Config(query_budget_s=.1),Config(static_scene=1)]:
   with self.assertRaises(ValueError):ObservedMap(cfg)
 def test_search_budget_is_fail_closed(self):
  m=ObservedMap();self.frame(m);space=MapSpace(m,clock=lambda:10.)
  space.search_deadline=time.perf_counter()-1
  with self.assertRaises(CertificationError):space.ball_free([1,0,0],.1)
 def test_snapshot_origin_cannot_mutate(self):
  s=self.frame(ObservedMap())
  with self.assertRaises(ValueError):s.origin[:]=0
 def test_prearm_rejects_stale_bootstrap_pass(self):
  h=dict(mode='ACK',plant=dict(integration_steps=0),ros_now=10.,bootstrap=dict(status='STATIC_PASS',valid_until=10.1))
  self.assertTrue(require_initial_coverage(h,10.01))
  for now in [10.11,9.9,float('nan')]:
   with self.assertRaises(PermissionError):require_initial_coverage(h,now)
  h['plant']['integration_steps']=1
  with self.assertRaises(PermissionError):require_initial_coverage(h,10.01)
 def test_e0_incomplete_or_failed_not_authorization(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'E0_RESULT.json';p.write_text(json.dumps(dict(status='STATIC_PASS',bootstrap_status='FAIL',frames=100,live_native_sensor=True)))
   with self.assertRaises(PermissionError):verify_e0(p)
 def test_new_ros_scripts_compile_without_ros_execution(self):
  for folder in ['ros/g4cp3_observed_v1','experiments/g4cp3_observed_v1','query_minco/integration/observed_space_v1']:
   for p in Path(folder).glob('*.py'):compile(p.read_text(),str(p),'exec')
 def test_e0_has_no_publish_arm_or_controller(self):
  for p in [Path('ros/g4cp3_observed_v1/e0_collector.py'),Path('experiments/g4cp3_observed_v1/manual_e0.py')]:
   tree=ast.parse(p.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
   self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ['Publisher','publish'] for n in calls))
   text=p.read_text();self.assertNotIn('/start_permit',text);self.assertNotIn('controller_only.launch',text)
 def test_execution_has_fresh_bootstrap_fence_and_same_solver(self):
  owner=Path('experiments/g4cp3_observed_v1/manual_v5.py').read_text()
  self.assertIn('require_initial_coverage(h,time.time())',owner)
  self.assertIn("predecessor(a.prior,'C_CAMPAIGN'",owner)
  ref=Path('ros/g4cp3_observed_v1/reference_v5.py').read_text()
  self.assertIn('yopo_execution_v4.trajectory import build',ref);self.assertIn('space=observed_space',ref)
  supervisor=Path('ros/g4cp3_observed_v1/supervisor_v5.py').read_text()
  self.assertIn('space.certify(gate.plan)',supervisor)
