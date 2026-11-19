"""SIM-only ARM helper ownership and attributable shutdown; no blanket -2 pass."""
import os
import signal
import time
import subprocess
from query_minco.integration.yopo_execution_v4.owned import identity, family, alive

def capture(process,argv):
 row=identity(process.pid)
 if row is None or os.getpgid(process.pid)!=process.pid or os.getsid(process.pid)!=process.pid:raise RuntimeError('ARM_HELPER_OWNER_UNAVAILABLE')
 return dict(pid=process.pid,start_ticks=row['start_ticks'],pgid=process.pid,session=process.pid,owner_pid=os.getpid(),captured_monotonic=time.monotonic(),argv=list(argv))

def cleanup(process,owner,terminal,timeout=5.):
 if not owner or owner['pid']!=process.pid or owner['owner_pid']!=os.getpid():raise RuntimeError('ARM_HELPER_OWNER_MISMATCH')
 events=[];members=family(process.pid);forced=False
 def send(sig):
  current=identity(process.pid)
  if current is None:return
  if current['start_ticks']!=owner['start_ticks'] or os.getpgid(process.pid)!=owner['pgid'] or os.getsid(process.pid)!=owner['session']:raise RuntimeError('ARM_HELPER_IDENTITY_CHANGED')
  event=dict(event='PARENT_SIGNAL',signal=int(sig),signal_name=signal.Signals(sig).name,sender_pid=os.getpid(),target_pid=process.pid,target_start_ticks=owner['start_ticks'],pgid=owner['pgid'],monotonic=time.monotonic(),sent=False)
  events.append(event)
  try:os.killpg(owner['pgid'],sig);event['sent']=True
  except ProcessLookupError:pass
 if process.poll() is None:send(signal.SIGINT)
 events.append(dict(event='WAIT_BEGIN',monotonic=time.monotonic()))
 try:code=process.wait(timeout=timeout)
 except subprocess.TimeoutExpired:
  forced=True;send(signal.SIGKILL);code=process.wait(timeout=2.)
 events.append(dict(event='WAIT_REAPED',monotonic=time.monotonic(),actual_exit_code=code))
 deadline=time.monotonic()+1.
 while any(alive(row) for row in members) and time.monotonic()<deadline:time.sleep(.005)
 residual=[row for row in members if alive(row)]
 stop_confirmed=bool(terminal and terminal.get('terminal_state') in ['OWNED_PROCESS_TERMINATED','OWNED_PROCESS_ALREADY_EXITED'] and terminal.get('actual_exit_code') is not None)
 attributed=bool(code==-signal.SIGINT and stop_confirmed and not forced and not residual and any(e.get('sent') and e.get('signal')==signal.SIGINT for e in events))
 return code,dict(owner=owner,events=events,stop_confirmed=stop_confirmed,forced=forced,residual_owned=residual,actual_exit_code=code,parent_initiated_sigint_expected=attributed)
