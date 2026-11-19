"""Targeted Goal and owned ARM-helper shutdown tests. No ROS/Plant."""
import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path
from experiments.sim_dev_oracle_v1.reference import fixed_goal,source
from experiments.sim_dev_oracle_v1.arm_cleanup import capture,cleanup
from experiments.sim_dev_oracle_v1.prepare_runtime import arm_cleanup_adapter,goal_queue_adapter
from query_minco.integration.yopo_execution_v4.lifecycle import Lifecycle
R=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('method_host',Path(__file__).with_name('test_state_sync.py'));host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
TERM=dict(terminal_state='OWNED_PROCESS_TERMINATED',actual_exit_code=-9)
class Goal(unittest.TestCase):
 def test_fixed_scene_goal_with_noisy_initial_odom(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);run=root/'none';run.mkdir();(root/'SCENE_IDENTITY.json').write_text(json.dumps(dict(scene=dict(seed=3,maze_type=5,tree_dist=6,spawn=[.25,-.25,2.],goal=[3.25,-.25,2.]))))
   goal=fixed_goal(run,[.25+1e-12,-.25-1e-12,2.+1e-12]);self.assertEqual(goal,[3.25,-.25,2.]);b,c=host.bridge();b.engine.set_goal(goal,100.);self.assertEqual(b.engine.goal,(goal,100.))
   with self.assertRaisesRegex(ValueError,'SIM_FIXED_SPAWN'):fixed_goal(run,[.3,-.25,2.])
 def test_wrong_stale_future_rejected_with_raw_diagnostics(self):
  for goal,stamp,reason in [([3.26,-.25,2.],100.,'COORDINATES'),([3.25,-.25,2.],99.799,'STALE'),([3.25,-.25,2.],100.001,'FUTURE')]:
   with self.subTest(reason=reason):
    b,c=host.bridge()
    with self.assertRaisesRegex(ValueError,reason):b.engine.set_goal(goal,stamp)
    events=b.engine.take_events();self.assertEqual(events[0]['event'],'GOAL_RECEIVED');self.assertEqual(events[-1]['event'],'GOAL_REJECTED');self.assertEqual(events[-1]['age'],100.-stamp);self.assertIsNone(b.engine.goal)
 def test_actual_adapters_compile_and_preserve_failure_checks(self):
  reference=(R/'query_minco/integration/depth_minco_t2_v1/ros_bridge.py').read_text();fn=next(n for n in ast.parse(reference).body if isinstance(n,ast.FunctionDef) and n.name=='reference_main');code=ast.get_source_segment(reference,fn);compile(source(code),'<actual SIM reference>','exec')
  owner=arm_cleanup_adapter((R/'experiments/stop_latency_fix_v1/manual.py').read_text());compile(owner,'<actual SIM owner>','exec');self.assertIn('unexpected_children', (R/'query_minco/integration/yopo_execution_v4/lifecycle.py').read_text());self.assertIn("expected_children=expected_arm_exits+",owner);self.assertIn("row['arm_cleanup_evidence']",owner)
  supervisor=goal_queue_adapter((R/'ros/g4cp7b_fix2_v1/supervisor.py').read_text());self.assertIn('queue_size=(1 if forest_bridge else 10)',supervisor)
class Arm(unittest.TestCase):
 def child(self,body):
  p=subprocess.Popen([sys.executable,'-u','-c',"import sys,time,os,signal;print('READY',flush=True);"+body],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  owner=capture(p,p.args);self.assertEqual(p.stdout.readline().strip(),b'READY')
  def release():
   if p.poll() is None:p.kill();p.wait()
   for stream in [p.stdin,p.stdout,p.stderr]:stream.close()
  self.addCleanup(release);return p,owner
 def tearDown(self):
  pass
 def test_normal_zero_exit_reaped(self):
  p,owner=self.child("sys.stdin.readline();sys.exit(0)");p.stdin.write(b'go\n');p.stdin.flush();p.wait();code,proof=cleanup(p,owner,TERM);self.assertEqual(code,0);self.assertEqual(proof['events'][-1]['event'],'WAIT_REAPED');self.assertEqual(proof['residual_owned'],[])
 def test_parent_sigint_is_attributed_and_only_then_expected(self):
  p,owner=self.child('time.sleep(10)');code,proof=cleanup(p,owner,TERM);self.assertEqual(code,-signal.SIGINT);self.assertTrue(proof['parent_initiated_sigint_expected']);sig=next(e for e in proof['events'] if e.get('signal')==signal.SIGINT);self.assertTrue(sig['sent']);self.assertEqual(sig['sender_pid'],os.getpid());self.assertEqual(sig['target_start_ticks'],owner['start_ticks'])
  life=Lifecycle();life.state='TERMINATED';life.qualified=True;result=life.result({'arm_permit':{'actual_exit_code':code}},TERM,expected_killed=['arm_permit']);self.assertEqual(result['unexpected_children'],{})
 def test_unattributed_minus_two_still_fails_unexpected_children(self):
  p,owner=self.child('sys.stdin.readline();os.kill(os.getpid(),signal.SIGINT)');p.stdin.write(b'go\n');p.stdin.flush();p.wait();code,proof=cleanup(p,owner,TERM);self.assertEqual(code,-signal.SIGINT);self.assertFalse(proof['parent_initiated_sigint_expected']);life=Lifecycle();life.state='TERMINATED';life.qualified=True;self.assertIn('arm_permit',life.result({'arm_permit':{'actual_exit_code':code}},TERM)['unexpected_children'])
 def test_abnormal_exit_recorded_and_not_accepted(self):
  p,owner=self.child('sys.stdin.readline();sys.exit(7)');p.stdin.write(b'go\n');p.stdin.flush();p.wait();code,proof=cleanup(p,owner,TERM);self.assertEqual(code,7);self.assertFalse(proof['parent_initiated_sigint_expected'])
 def test_ownership_mismatch_does_not_signal(self):
  p,owner=self.child('time.sleep(10)');owner=dict(owner,owner_pid=-1)
  with self.assertRaisesRegex(RuntimeError,'OWNER_MISMATCH'):cleanup(p,owner,TERM)
  self.assertIsNone(p.poll())
if __name__=='__main__':unittest.main()
