import unittest,tempfile,threading,time,json,copy,socket
from pathlib import Path
import numpy as np
from query_minco.integration.yopo_execution_v4.lifecycle import Journal,Lifecycle
from query_minco.integration.yopo_execution_v4.metrics import summarize
from query_minco.integration.yopo_execution_v4.trajectory import build,Plan,Admission
from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace
from query_minco.integration.yopo_execution_v4.evidence import expected_fault,predecessor,CASES
class Runtime(unittest.TestCase):
 def test_duplicate_actual_readiness(self):
  import importlib.util
  from query_minco.integration.yopo_execution_v4.readiness import Readiness
  spec=importlib.util.spec_from_file_location('fixture_p1',Path('tests/g4cp1_startup_control_v1/test_handshake.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  r=Readiness(True,True,'zero');values=m.Handshake().fixture()
  for _ in range(5):
   for kind,data in values.items():r.observe(kind,10.,10.,data)
  self.assertEqual(r.tick(10.),'HELD');self.assertTrue(all(v==1 for v in r.counts.values()))
  self.assertEqual(r.tick(10.201),'ABORT')
 def test_owned_stop_failure_has_explicit_fallback(self):
  import subprocess,sys
  from query_minco.integration.yopo_execution_v4.owned import stop_plant
  class Broken:
   def stop(self,reason):raise RuntimeError('SYNTHETIC_GUARD_FAILURE')
  child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True)
  terminal,errors=stop_plant(child,Broken(),'UNIT_ONLY')
  self.assertTrue(errors);self.assertIsNotNone(child.poll());self.assertTrue(terminal['fallback'])
 def test_owner_launch_exception_is_fail(self):
  import importlib.util
  from unittest.mock import patch
  spec=importlib.util.spec_from_file_location('owner_p2',Path('experiments/g4cp2_execution_v1/manual.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory(dir='research_outputs/g4cp2_execution_20261111_v1') as t:
   with patch.object(m.subprocess,'Popen',side_effect=RuntimeError('SYNTHETIC_LAUNCH_FAILURE')):
    result=m.episode(Path(t)/'case','C','request_timeout',{}, {},{})
   self.assertEqual(result['status'],'FAIL');self.assertFalse(result['physical_feedback'])
   Path('research_outputs/g4cp2_execution_20261111_v1/SYNTHETIC_LAUNCH_FAILURE_RESULT.json').write_text(json.dumps(dict(classification='SYNTHETIC_UNIT_ONLY',result=result),indent=2))
 def test_late_callback_after_close(self):

  with tempfile.TemporaryDirectory() as t:
   j=Journal(Path(t)/'log');j.close()
   with j.callback() as active:self.assertFalse(active)
   self.assertFalse(j.append(dict(kind='late')))
 def test_delayed_callback_close_race(self):
  with tempfile.TemporaryDirectory() as t:
   j=Journal(Path(t)/'log');started=threading.Event();release=threading.Event();errors=[]
   def callback():
    try:
     with j.callback() as active:
      started.set();release.wait(1);self.assertTrue(active);j.append(dict(kind='inflight'))
    except Exception as e:errors.append(repr(e))
   th=threading.Thread(target=callback);th.start();self.assertTrue(started.wait(1))
   close=threading.Thread(target=j.close);close.start();release.set();th.join(2);close.join(2)
   self.assertFalse(errors);self.assertFalse(th.is_alive());self.assertFalse(close.is_alive())
   rows=[json.loads(x) for x in Path(t,'log').read_text().splitlines()];self.assertEqual(rows[-1]['kind'],'JOURNAL_CLOSED')
 def completed(self):
  l=Lifecycle();l.move('QUALIFIED',1);l.move('TERMINATING',2);l.move('TERMINATED',3);return l
 def test_no_pass_from_qualification_alone(self):
  l=self.completed();r=l.result({'supervisor':dict(actual_exit_code=1)},dict(terminal_state='OWNED_PROCESS_TERMINATED'))
  self.assertEqual(r['runtime_status'],'PASS');self.assertEqual(r['cleanup_status'],'FAIL');self.assertEqual(r['status'],'FAIL')
 def test_no_pass_from_abort_flag(self):
  l=self.completed();self.assertEqual(l.result({},None)['status'],'FAIL')
 def test_runtime_fault_not_hidden_by_termination(self):
  l=self.completed();l.fault('depth_STALE_before_stop',1.5)
  self.assertEqual(l.result({},dict(terminal_state='OWNED_PROCESS_TERMINATED'))['status'],'FAIL')
 def test_state_skip(self):
  with self.assertRaises(ValueError):Lifecycle().move('TERMINATED',1)
 def test_metrics_full_window_not_last(self):
  rows=[dict(kind='odom',accepted=True,stamp=10+i*.01,ros_now=10+i*.01,monotonic=100+i*.01,payload=dict(p=[i*.1,0,2],v=[v,0,0])) for i,v in enumerate([1.5,0,0])]
  r=summarize(rows,begin=100,end=100.02,anchor=[0,0,2])
  self.assertEqual(r['maxima']['speed'],1.5);self.assertEqual(r['maxima']['position_drift'],.2)
 def test_sensor_stopped_tail(self):
  row=dict(kind='imu',accepted=True,stamp=10,ros_now=10,monotonic=100,payload=dict(a_world=[0,0,0]))
  r=summarize([row],end=100.3)
  self.assertAlmostEqual(r['intervals']['imu']['end_source_age_s'],.3);self.assertAlmostEqual(r['intervals']['imu']['end_receipt_age_s'],.3)
 def test_interval_gap_and_duplicate(self):
  rows=[dict(kind='raw',accepted=True,stamp=x,ros_now=x,monotonic=x+90,payload={}) for x in [10,10,10.3]]
  r=summarize(rows);self.assertEqual(r['intervals']['raw']['duplicates'],1);self.assertAlmostEqual(r['intervals']['raw']['max_source_gap_s'],.3)
 def test_expected_faults_not_generic(self):
  for c,r in zip(CASES,['request_STALE','depth_STALE','odom_STAMP','REQUEST ValueError','reference_EXIT','controller_EXIT']):self.assertTrue(expected_fault(c,r))
  self.assertFalse(expected_fault('depth_loss','TRUTH_COLLISION'))
 def test_fake_C_incomplete(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'result';p.write_text(json.dumps(dict(stage='C_CAMPAIGN',status='PASS',source_binding={},episodes=[])))
   with self.assertRaises(PermissionError):predecessor(p,'C_CAMPAIGN',{})
 def test_fake_D_nonphysical(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'result';p.write_text(json.dumps(dict(stage='D',status='PASS',source_binding={},physical_feedback=False)))
   with self.assertRaises(PermissionError):predecessor(p,'D',{})
 def test_listening_port_not_bypassed(self):
  with socket.socket() as a:
   a.bind(('127.0.0.1',0));a.listen();port=a.getsockname()[1]
   with socket.socket() as b:
    b.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    with self.assertRaises(OSError):b.bind(('127.0.0.1',port))
class Curves(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.plan=build(np.array([0.,0.,2.]),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,2]),100.)
 def test_rotated_world_X_and_lateral_goals(self):
  for q,goal in [([0,0,np.sin(np.pi/4),np.cos(np.pi/4)],[.4,0,2]),([0,0,0,1],[0,.4,2])]:
   p=build(np.array([0.,0.,2.]),np.zeros(3),np.zeros(3),q,np.array(goal),100.)
   np.testing.assert_allclose(p.sample(2)[0],goal,atol=.002)
 def test_real_minco_04_limits(self):
  self.assertLess(self.plan.maxima[0],.376);self.assertLess(self.plan.maxima[1],.58);self.assertLess(self.plan.maxima[2],3.01)
  np.testing.assert_allclose(self.plan.sample(0)[0],[0,0,2],atol=.002);np.testing.assert_allclose(self.plan.sample(2)[0],[.4,0,2],atol=.002)
 def test_continuity_independent(self):self.assertTrue(all(v<t for v,t in zip(self.plan.errors,[.002,.01,.05,.25,2.])))
 def test_nonzero_initial_pva(self):
  p=build(np.array([0.,0.,2.]),np.array([.1,0,0]),np.array([.01,0,0]),[0,0,0,1],np.array([.4,0,2]),100.)
  np.testing.assert_allclose(p.sample(0)[1],[.1,0,0],atol=.01);np.testing.assert_allclose(p.sample(0)[2],[.01,0,0],atol=.05)
 def test_bad_time_and_nonfinite(self):
  for value in [0.,-1.,float('nan'),float('inf')]:
   rec=copy.deepcopy(self.plan.record);rec['durations'][0]=value
   with self.assertRaises(ValueError):Plan(rec)
 def test_admission_atomic_reject_and_commit(self):
  a=Admission();s=dict(p=[0,0,2],v=[0,0,0],a=[0,0,0]);en=dict(kind='trajectory',stamp=100.,pva=self.plan.sample(0).tolist(),plan=self.plan.record)
  bad=copy.deepcopy(en);bad['pva'][0][0]=.01
  with self.assertRaises(ValueError):a.check(bad,s,100.)
  self.assertIsNone(a.plan);a.check(en,s,100.);self.assertEqual(a.plan.id,self.plan.id)
 def test_command_race_and_stale_plan(self):
  a=Admission();s=dict(p=[0,0,2],v=[0,0,0],a=[0,0,0]);e=dict(kind='trajectory',stamp=100.,pva=self.plan.sample(0).tolist(),plan=self.plan.record)
  with self.assertRaises(ValueError):a.check(e,s,100.3)
  a.check(e,s,100.);e['stamp']=103.;e['pva']=self.plan.at(103.).tolist()
  with self.assertRaises(ValueError):a.check(e,s,103.)
 def test_unknown_not_free(self):
  pose=dict(p=[0,0,2],q=[0,0,0,1])
  self.assertFalse(ObservedSpace(np.full((90,160),20,np.float32),pose,0).ball_free([2,0,2],.15))
 def test_positive_observed_sphere_and_rotation(self):
  d=np.full((90,160),5,np.float32)
  a=ObservedSpace(d,dict(p=[0,0,2],q=[0,0,0,1]),0);self.assertTrue(a.ball_free([2,0,2],.15));self.assertFalse(a.ball_free([-2,0,2],.15))
  a=ObservedSpace(d,dict(p=[0,0,2],q=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]),0);self.assertTrue(a.ball_free([0,2,2],.15))
 def test_startup_blind_zone_fails_closed(self):
  a=ObservedSpace(np.full((90,160),5,np.float32),dict(p=[0,0,2],q=[0,0,0,1]),0)
  with self.assertRaises(ValueError):a.certify(self.plan)
 def test_deterministic_search_no_anchor(self):
  a=ObservedSpace(np.full((90,160),5,np.float32),dict(p=[0,0,2],q=[0,0,0,1]),0)
  g1=a.search_goal([1,0,2],[2,0,2]);g2=a.search_goal([1,0,2],[2,0,2]);np.testing.assert_array_equal(g1,g2)
