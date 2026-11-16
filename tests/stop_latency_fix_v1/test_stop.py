import unittest,subprocess,sys,json
from pathlib import Path
from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop
class Tests(unittest.TestCase):
 def test_actual_stop_trace_and_reap(self):
  p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True)
  try:
   d=OwnedProcessStop(p).stop('UNIT_NO_ROS')
   names=[e['event'] for e in d['events']]
   self.assertIn('PAUSE_WAITPID_OBSERVED',names);self.assertIn('SIGKILL_SENT',names);self.assertIn('WAITPID_REAPED',names)
   self.assertEqual(d['actual_exit_code'],-9);self.assertIsNotNone(p.poll())
   self.assertEqual(d['events'][-1]['event'],'EXIT_CONFIRMED')
  finally:
   if p.poll() is None:p.kill();p.wait()
 def test_no_main_loop_family_scan(self):
  s=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
  loop=s[s.index('  while time.monotonic()-begin'):s.index(' except BaseException as e:reason=repr(e)')]
  self.assertNotIn('family(child.pid)',loop)
  self.assertIn('OWNER_RECEIVED_ABORT',loop)
 def test_total_acceptance_unchanged(self):
  s=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
  self.assertIn('stop_time-injected<=.35',s)
 def test_stress_records_actual_exit(self):
  d=json.loads(Path('research_outputs/stop_latency_fix_20261116_v1/FIXED_NONMOTION.json').read_text())
  self.assertEqual(len(d['cases']),12)
  self.assertTrue(all(x['actual_exit_code']==-9 and x['stop_state_observed'] for x in d['cases']))
