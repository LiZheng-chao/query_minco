"""Exclusive-session stop; uncertainty is never an exit confirmation."""
import os, signal, time, ctypes, select
from pathlib import Path

def identity(pid):
 try:
  s = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
  return s[0], s[19]
 except FileNotFoundError:
  return None, None

def members(pgid):
 """Live members of the owned group OR session; zombies cannot integrate."""
 rows = []
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():
   continue
  try:
   s = (path / 'stat').read_text().rsplit(')', 1)[1].split()
   if (int(s[2]) == pgid or int(s[3]) == pgid) and s[0] != 'Z':
    rows.append(int(path.name))
  except FileNotFoundError:
   pass
 return rows

class OwnedProcessStop:
 def __init__(self, process):
  if process.pid == os.getpid() or os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
   raise ValueError('EXCLUSIVE_PROCESS_GROUP_AND_SESSION')
  self.process = process
  self.start_ticks = identity(process.pid)[1]
  if self.start_ticks is None:
   raise ValueError('MISSING_INITIAL_IDENTITY')

 def stop(self, reason):
  p = self.process
  events = []
  self.events = events
  begin = time.monotonic()
  stopped = False
  fd = None
  errors = []
  deadline = begin + 3.
  def mark(name, **kw):
   events.append(dict(event=name, monotonic=time.monotonic(), state=identity(p.pid)[0], **kw))
  def reap(flags):
   while True:
    try:
     return os.waitpid(p.pid, flags)
    except InterruptedError:
     if time.monotonic() >= deadline:
      raise TimeoutError('WAITPID_EINTR_TIMEOUT')
  def status_code(status):
   if os.WIFSIGNALED(status):
    return -os.WTERMSIG(status)
   if os.WIFEXITED(status):
    return os.WEXITSTATUS(status)
   raise RuntimeError('NONTERMINAL_WAIT_STATUS')
  def send(sig):
   # Sole-reaper ownership keeps the unreaped leader PID reserved.
   if identity(p.pid)[1] != self.start_ticks or os.getpgid(p.pid) != p.pid or os.getsid(p.pid) != p.pid:
    raise RuntimeError('PID_IDENTITY_CHANGED')
   mark('SIGSTOP_BEGIN' if sig == signal.SIGSTOP else 'SIGKILL_BEGIN')
   try:
    os.killpg(p.pid, sig)
   except ProcessLookupError:
    mark('SIGNAL_ESRCH', signal=int(sig))
   else:
    mark('SIGSTOP_SENT' if sig == signal.SIGSTOP else 'SIGKILL_SENT')
  mark('STOP_REQUEST')
  try:
   # Do not poll/reap before signalling: the reserved leader protects PGID.
   state, ticks = identity(p.pid)
   if ticks is not None and ticks != self.start_ticks:
    raise RuntimeError('PID_IDENTITY_CHANGED')
   if p.returncode is None:
    if ticks is None:
     # External reaping cannot establish a trustworthy exit code.
     raise RuntimeError('LEADER_EXIT_STATUS_UNKNOWN')
    if os.uname().machine == 'x86_64':
     raw = ctypes.CDLL(None, use_errno=True).syscall(434, p.pid, 0)
     if raw >= 0:
      fd = raw
    send(signal.SIGSTOP)
    until = min(deadline, time.monotonic() + .1)
    while time.monotonic() < until:
     pid, status = reap(os.WNOHANG | os.WUNTRACED)
     if pid and os.WIFSTOPPED(status):
      stopped = True
      mark('PAUSE_WAITPID_OBSERVED')
      break
     if pid and (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
      p.returncode = status_code(status)
      mark('EXIT_REAPED_BEFORE_KILL')
      break
     time.sleep(.0005)
    if p.returncode is None:
     send(signal.SIGKILL)
     if fd is not None:
      poll = select.poll()
      poll.register(fd, select.POLLIN)
      if not poll.poll(max(0, int((deadline - time.monotonic()) * 1000))):
       raise TimeoutError('PIDFD_EXIT_TIMEOUT')
      mark('PIDFD_EXIT_READY')
     mark('WAITPID_BEGIN')
     while True:
      pid, status = reap(os.WNOHANG)
      if pid:
       p.returncode = status_code(status)
       break
      if time.monotonic() >= deadline:
       raise TimeoutError('WAITPID_EXIT_TIMEOUT')
      time.sleep(.0005)
     mark('WAITPID_REAPED')
   if p.returncode is None:
    raise RuntimeError('LEADER_EXIT_STATUS_UNKNOWN')
   # Descendants killed in the group may need scheduling time to become zombies.
   while members(p.pid):
    if time.monotonic() >= deadline:
     raise RuntimeError('OWNED_SESSION_MEMBERS_REMAIN')
    time.sleep(.0005)
   mark('EXIT_CONFIRMED', exit_code=p.returncode, scope='leader reaped; no live owned session/group member')
  except Exception as error:
   errors.append(repr(error))
   mark('STOP_NOT_CONFIRMED', error=repr(error))
  finally:
   if fd is not None:
    os.close(fd)
  return dict(reason=reason, pid=p.pid, start_ticks=self.start_ticks,
              stop_state_observed=stopped, actual_exit_code=p.returncode,
              terminal_state='STOP_NOT_CONFIRMED' if errors else 'OWNED_PROCESS_TERMINATED',
              errors=errors, elapsed_s=time.monotonic()-begin, events=events,
              criterion='trustworthy leader exit and no live owned session/group members')
