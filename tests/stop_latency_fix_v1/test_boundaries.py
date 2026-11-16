"""NONPHYSICAL only: isolated owned sessions; never ROS or Plant."""
import os, sys, signal, subprocess, unittest, time, ctypes
from unittest.mock import patch
import query_minco.integration.owned_stop_v2.stop as m
from query_minco.integration.yopo_execution_v4.owned import stop_plant, plant_exit_event

class Boundaries(unittest.TestCase):
 def spawn(self, code='import time;time.sleep(30)'):
  p=subprocess.Popen([sys.executable,'-B','-c',code],start_new_session=True,
                     stdout=subprocess.PIPE,text=True)
  self.addCleanup(self.cleanup,p)
  return p
 def cleanup(self,p):
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait(timeout=3)
  if p.stdout:p.stdout.close()
 def check_fd(self,p,context=None):
  guard=m.OwnedProcessStop(p);before=len(os.listdir('/proc/self/fd'))
  if context:
   with context:r=guard.stop('BOUNDARY_NONPHYSICAL')
  else:r=guard.stop('BOUNDARY_NONPHYSICAL')
  self.assertEqual(len(os.listdir('/proc/self/fd')),before)
  return r
 def test_single_and_nonowned_survives(self):
  sentinel=self.spawn();p=self.spawn()
  r=self.check_fd(p)
  self.assertEqual(r['terminal_state'],'OWNED_PROCESS_TERMINATED')
  self.assertEqual(r['actual_exit_code'],-9)
  self.assertIsNone(sentinel.poll())
 def group(self,exit_leader=False,stop_child=False):
  code = 'import os,time,signal; child=os.fork();'+chr(10)+'if child==0: '+('os.kill(os.getpid(),signal.SIGSTOP);' if stop_child else '')+'time.sleep(30);os._exit(0)'+chr(10)+'print(child,flush=True)'+chr(10)+'time.sleep(.2)'+chr(10)+('os._exit(0)' if exit_leader else 'time.sleep(30)')
  p=self.spawn(code);g=m.OwnedProcessStop(p);child=int(p.stdout.readline())
  return p,g,child
 def test_group(self):
  p,g,c=self.group();r=g.stop('GROUP')
  self.assertEqual(r['terminal_state'],'OWNED_PROCESS_TERMINATED')
  self.assertEqual(m.members(p.pid),[])
 def test_stopped_descendant(self):
  p,g,c=self.group(stop_child=True);r=g.stop('STOPPED_CHILD')
  self.assertEqual(r['terminal_state'],'OWNED_PROCESS_TERMINATED')
  self.assertEqual(m.members(p.pid),[])
 def test_leader_reaped_child_alive_fails_closed(self):
  p,g,c=self.group(exit_leader=True);p.wait(timeout=3)
  r=g.stop('LEADER_GONE')
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
  self.assertNotIn('EXIT_CONFIRMED',[e['event'] for e in r['events']])
  self.assertNotEqual(m.identity(c)[0],'Z')
 def test_pidfd_unavailable(self):
  class Missing:
   def syscall(self,*a):return -1
  r=self.check_fd(self.spawn(),patch.object(m.ctypes,'CDLL',return_value=Missing()))
  self.assertEqual(r['actual_exit_code'],-9)
 def test_signal_permission_error_no_leak(self):
  r=self.check_fd(self.spawn(),patch.object(m.os,'killpg',side_effect=PermissionError()))
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
 def test_sigstop_esrch_no_leak(self):
  original=os.killpg
  def raced(pid,sig):
   if sig==signal.SIGSTOP:raise ProcessLookupError()
   original(pid,sig)
  r=self.check_fd(self.spawn(),patch.object(m.os,'killpg',side_effect=raced))
  self.assertEqual(r['actual_exit_code'],-9)
 def test_sigkill_esrch_after_delivery(self):
  original=os.killpg
  def raced(pid,sig):
   original(pid,sig)
   if sig==signal.SIGKILL:raise ProcessLookupError()
  r=self.check_fd(self.spawn(),patch.object(m.os,'killpg',side_effect=raced))
  self.assertEqual(r['actual_exit_code'],-9)
 def test_eintr_retry(self):
  original=os.waitpid;calls=[0]
  def interrupted(*a):
   calls[0]+=1
   if calls[0]==1:raise InterruptedError()
   return original(*a)
  r=self.check_fd(self.spawn(),patch.object(m.os,'waitpid',side_effect=interrupted))
  self.assertEqual(r['actual_exit_code'],-9)
 def test_echild_unknown_no_leak(self):
  r=self.check_fd(self.spawn(),patch.object(m.os,'waitpid',side_effect=ChildProcessError()))
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
  self.assertIsNone(r['actual_exit_code'])
 def test_pre_reaped_single_known_status(self):
  p=self.spawn('import time;time.sleep(.1)');g=m.OwnedProcessStop(p)
  self.assertEqual(p.wait(timeout=3),0)
  self.assertEqual(g.stop('REAPED')['terminal_state'],'OWNED_PROCESS_TERMINATED')
 def test_owner_does_not_convert_error_to_success(self):
  class Broken:
   def stop(self,r):raise ChildProcessError()
  r,errors=stop_plant(self.spawn(),Broken(),'TEST')
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED');self.assertTrue(errors)
  self.assertEqual(plant_exit_event(r,1.)['event'],'PLANT_STOP_NOT_CONFIRMED')
 def test_launcher_uses_conditional_event(self):
  from pathlib import Path
  s=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
  self.assertIn('owner_events.append(plant_exit_event(terminal,stop_time))',s)
  self.assertNotIn("owner_events.append(dict(event='PLANT_EXIT_CONFIRMED'",s)

 def test_pidfd_timeout_no_leak(self):
  class TimeoutPoll:
   def register(self,*a):pass
   def poll(self,*a):return []
  r=self.check_fd(self.spawn(),patch.object(m.select,'poll',return_value=TimeoutPoll()))
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
  self.assertIsNone(r['actual_exit_code'])
 def test_sigkill_esrch_without_delivery_not_confirmed(self):
  original=os.killpg
  def missing(pid,sig):
   if sig==signal.SIGKILL:raise ProcessLookupError()
   original(pid,sig)
  r=self.check_fd(self.spawn(),patch.object(m.os,'killpg',side_effect=missing))
  self.assertEqual(r['terminal_state'],'STOP_NOT_CONFIRMED')
