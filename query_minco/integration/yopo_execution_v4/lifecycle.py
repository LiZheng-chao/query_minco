"""Run qualification is provisional until owned termination and child reconciliation."""
import json,threading,time,math
from contextlib import contextmanager
from pathlib import Path
class Journal:
 def __init__(self,path):
  self.lock=threading.RLock();self.file=Path(path).open('x');self.closing=False;self.late_callbacks=0
 @contextmanager
 def callback(self):
  with self.lock:
   if self.closing:self.late_callbacks+=1;yield False
   else:yield True
 def append(self,event):
  with self.lock:
   if self.closing:return False
   try:text=json.dumps(event,allow_nan=False)
   except (ValueError,TypeError):text=json.dumps(dict(kind='INVALID_EVENT',payload_repr=repr(event)))
   self.file.write(text+'\n');self.file.flush();return True
 def close(self,unregister=()):
  with self.lock:self.closing=True
  errors=[]
  for fn in unregister:
   try:fn()
   except Exception as e:errors.append(repr(e))
  with self.lock:
   self.file.write(json.dumps(dict(kind='JOURNAL_CLOSED',monotonic=time.monotonic(),late_callbacks=self.late_callbacks,cleanup_errors=errors))+'\n')
   self.file.flush();self.file.close()
  return errors
class Lifecycle:
 def __init__(self):self.state='RUNNING';self.transitions=[];self.faults=[];self.qualified=False
 def move(self,state,now):
  allowed={'RUNNING':['QUALIFIED','TERMINATING'],'QUALIFIED':['TERMINATING'],'TERMINATING':['TERMINATED'],'TERMINATED':[]}
  if state not in allowed[self.state]:raise ValueError('LIFECYCLE_SKIP')
  self.transitions.append(dict(before=self.state,after=state,monotonic=now));self.state=state
  if state=='QUALIFIED':self.qualified=True
 def fault(self,reason,now):self.faults.append(dict(reason=reason,monotonic=now,state=self.state))
 def result(self,children,terminal,expected_killed=(),expected_fault=False):
  unexpected={k:v for k,v in children.items() if v['actual_exit_code'] not in ([0,-9,-15,-2]+([3] if expected_fault and k=='plant' else []) if k in expected_killed else [0])}
  cleanup='PASS' if not unexpected and terminal and terminal.get('terminal_state') in ['OWNED_PROCESS_TERMINATED','OWNED_PROCESS_ALREADY_EXITED'] else 'FAIL'
  runtime='PASS' if self.qualified and (not self.faults or expected_fault) else 'FAIL'
  return dict(status='PASS' if runtime==cleanup=='PASS' and self.state=='TERMINATED' else 'FAIL',runtime_status=runtime,cleanup_status=cleanup,
   lifecycle=self.state,transitions=self.transitions,faults=self.faults,unexpected_children=unexpected,qualified_before_cleanup=self.qualified)
def atomic_json(path,value):
 p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,allow_nan=False));temp.replace(p)
