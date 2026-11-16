"""PID-start-time-bound ownership;never pkill arbitrary ROS processes."""
from pathlib import Path
import os,signal
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
 errors=[]
 if process is None:return None,['PLANT_NOT_STARTED']
 try:
  if guard is None:raise RuntimeError('NO_INITIALIZED_PROCESS_GUARD')
  result=guard.stop(reason)
  if result.get('terminal_state')!='OWNED_PROCESS_TERMINATED':
   errors.extend(result.get('errors') or ['STOP_NOT_CONFIRMED'])
  return result,errors
 except Exception as error:
  # No unverified killpg fallback or leader-only success conversion.
  return dict(terminal_state='STOP_NOT_CONFIRMED',actual_exit_code=None),[repr(error)]

def plant_exit_event(terminal,monotonic):
 confirmed=bool(terminal and terminal.get('terminal_state')=='OWNED_PROCESS_TERMINATED'
                and terminal.get('actual_exit_code') is not None)
 return dict(event='PLANT_EXIT_CONFIRMED' if confirmed else 'PLANT_STOP_NOT_CONFIRMED',
             monotonic=monotonic,terminal=terminal)
