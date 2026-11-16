"""NONPHYSICAL: emergency cleanup before any launcher wait."""
import os,signal,subprocess,sys,time,unittest
from unittest.mock import patch
from pathlib import Path
from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop,members
from query_minco.integration.yopo_execution_v4.owned import stop_plant,plant_exit_event
class T1(unittest.TestCase):
 def child(self):
  p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],start_new_session=True)
  def clean():
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=3)
  self.addCleanup(clean)
  return p
 def test_immediate_emergency_after_primary_failure(self):
  p=self.child();g=OwnedProcessStop(p);before=len(os.listdir('/proc/self/fd'))
  begin=time.monotonic()
  with patch.object(g,'stop',return_value=dict(terminal_state='STOP_NOT_CONFIRMED',errors=['INJECTED'])):
   r,errors=stop_plant(p,g,'INJECTED')
  self.assertTrue(errors);self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
  self.assertTrue(r['emergency']['confirmed']);self.assertEqual(p.returncode,-9)
  self.assertLess(time.monotonic()-begin,1.)
  self.assertEqual(members(p.pid),[]);self.assertEqual(len(os.listdir('/proc/self/fd')),before)
  self.assertEqual(plant_exit_event(r,1.)['event'],'PLANT_STOP_NOT_CONFIRMED')
 def test_exception_enters_emergency(self):
  p=self.child();g=OwnedProcessStop(p)
  with patch.object(g,'stop',side_effect=InterruptedError()):
   r,errors=stop_plant(p,g,'EXCEPTION')
  self.assertEqual(p.returncode,-9);self.assertTrue(errors)
  self.assertTrue(r['emergency']['confirmed'])
 def test_identity_mismatch_never_signals(self):
  p=self.child();g=OwnedProcessStop(p);g.start_ticks='UNTRUSTED'
  with patch('query_minco.integration.owned_stop_v2.stop.os.killpg') as kill:
   r,errors=stop_plant(p,g,'UNTRUSTED')
  kill.assert_not_called();self.assertFalse(r['emergency']['confirmed'])
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
 def test_primary_success_never_emergency(self):
  p=self.child();g=OwnedProcessStop(p)
  with patch.object(g,'emergency_stop',side_effect=AssertionError('NOT_EXPECTED')):
   r,errors=stop_plant(p,g,'NORMAL')
  self.assertFalse(errors);self.assertEqual(r['terminal_state'],'OWNED_PROCESS_TERMINATED')
 def test_unconfirmed_plant_skips_generic_cleanup(self):
  s=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
  branch=s[s.index("   if p is plant and (not terminal"):s.index("   if p is not plant and p.poll()")]
  self.assertNotIn('killpg(',branch);self.assertNotIn('p.wait(',branch);self.assertIn('continue',branch)
