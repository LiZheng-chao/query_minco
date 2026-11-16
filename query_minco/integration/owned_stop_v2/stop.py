"""Identity-bound stop trace;exit readiness avoids Popen timeout backoff."""
import os,signal,time,ctypes,select
from pathlib import Path
def identity(pid):
 try:
  s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
  return s[0],s[19]
 except FileNotFoundError:return None,None
class OwnedProcessStop:
 def __init__(self,process):
  if process.pid==os.getpid() or os.getpgid(process.pid)!=process.pid:raise ValueError('EXCLUSIVE_PROCESS_GROUP')
  self.process=process;self.start_ticks=identity(process.pid)[1]
 def stop(self,reason):
  p=self.process;events=[];self.events=events;begin=time.monotonic();stopped=False;fd=None
  def mark(name,**kw):
   state,start=identity(p.pid);events.append(dict(event=name,monotonic=time.monotonic(),state=state,**kw))
  mark('STOP_REQUEST')
  if p.poll() is None:
   if identity(p.pid)[1]!=self.start_ticks or os.getpgid(p.pid)!=p.pid:raise ValueError('PID_IDENTITY_CHANGED')
   if os.uname().machine=='x86_64':
    raw=ctypes.CDLL(None,use_errno=True).syscall(434,p.pid,0)
    if raw>=0:fd=raw
   mark('SIGSTOP_BEGIN');os.killpg(p.pid,signal.SIGSTOP);mark('SIGSTOP_SENT')
   until=time.monotonic()+.1
   while time.monotonic()<until:
    pid,status=os.waitpid(p.pid,os.WNOHANG|os.WUNTRACED)
    if pid and os.WIFSTOPPED(status):stopped=True;mark('PAUSE_WAITPID_OBSERVED');break
    if pid and (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
     p.returncode=-os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status);mark('EXIT_REAPED_BEFORE_KILL');break
    time.sleep(.0005)
   if p.returncode is None:
    mark('SIGKILL_BEGIN')
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    mark('SIGKILL_SENT')
    try:
     if fd is not None:
      poll=select.poll();poll.register(fd,select.POLLIN)
      if not poll.poll(3000):raise TimeoutError('PIDFD_EXIT_TIMEOUT')
      mark('PIDFD_EXIT_READY');mark('WAITPID_BEGIN');p.wait();mark('WAITPID_REAPED')
     else:
      mark('WAITPID_BEGIN');until=time.monotonic()+3
      while True:
       pid,status=os.waitpid(p.pid,os.WNOHANG)
       if pid:
        p.returncode=-os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status);break
       if time.monotonic()>until:raise TimeoutError('WAITPID_EXIT_TIMEOUT')
       time.sleep(.0005)
      mark('WAITPID_REAPED')
    finally:
     if fd is not None:os.close(fd);fd=None
  if fd is not None:os.close(fd)
  mark('EXIT_CONFIRMED',exit_code=p.returncode)
  return dict(reason=reason,pid=p.pid,start_ticks=self.start_ticks,stop_state_observed=stopped,actual_exit_code=p.returncode,terminal_state='OWNED_PROCESS_TERMINATED',elapsed_s=time.monotonic()-begin,events=events,criterion='actual exit/reap,not pause substituted for exit')
