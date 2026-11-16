"""PID-start-time-bound ownership;never pkill arbitrary ROS processes."""
from pathlib import Path
import os,signal
from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop
def identity(pid):
 try:
  x=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
  return dict(state=x[0],pid=pid,ppid=int(x[1]),start_ticks=x[19])
 except (OSError,IndexError):return None
def family(root):
 rows={int(p.name):identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()};rows={k:v for k,v in rows.items() if v}
 ids={root}
 while True:
  more={k for k,v in rows.items() if v['ppid'] in ids}
  if more<=ids:break
  ids|=more
 return [rows[k] for k in ids if k in rows]
def alive(row):
 current=identity(row['pid']);return current is not None and current['state']!='Z' and current['start_ticks']==row['start_ticks']
def kill_family(root,sig=signal.SIGKILL):
 rows=family(root)
 for row in sorted(rows,key=lambda r:r['pid']==root):
  if row['pid']!=os.getpid() and alive(row):
   try:os.kill(row['pid'],sig)
   except ProcessLookupError:pass
 return rows

def stop_plant(process,guard,reason):
 if process is None:return None,['PLANT_NOT_STARTED']
 errors=[]
 try:
  if guard is None:raise RuntimeError('NO_INITIALIZED_PROCESS_GUARD')
  result=guard.stop(reason)
  if result.get('terminal_state')=='OWNED_PROCESS_TERMINATED':
   return result,errors
  errors.extend(result.get('errors') or ['STOP_NOT_CONFIRMED'])
 except Exception as error:
  errors.append(repr(error))
  result=dict(terminal_state='STOP_NOT_CONFIRMED',actual_exit_code=None)
 # Attempt trusted emergency cleanup NOW, before launcher waits for other nodes.
 # Never convert a failed primary STOP into a passing acceptance.
 if isinstance(guard, OwnedProcessStop) and guard.process is process:
  result['emergency']=guard.emergency_stop(reason)
  if result['emergency']['confirmed']:
   result['actual_exit_code']=result['emergency']['actual_exit_code']
  else:
   errors.append(result['emergency'].get('error','EMERGENCY_STOP_NOT_CONFIRMED'))
 else:
  errors.append('NO_TRUSTED_EMERGENCY_GUARD')
 return result,errors

def plant_exit_event(terminal,monotonic):
 confirmed=bool(terminal and terminal.get('terminal_state')=='OWNED_PROCESS_TERMINATED'
                and terminal.get('actual_exit_code') is not None)
 return dict(event='PLANT_EXIT_CONFIRMED' if confirmed else 'PLANT_STOP_NOT_CONFIRMED',
             monotonic=monotonic,terminal=terminal)
