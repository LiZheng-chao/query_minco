"""Exact run-scoped descendant cleanup; no broad pkill and no unrelated master shutdown."""
from pathlib import Path
import os,sys,time,signal,json,socket
ROOT=Path('/home/lizheng/query_minco_mainline/research_outputs/pg2_stagec_pg3_preservation_v1/SYSTEM')
ALLOWED={'moving001':11401,'watchdog001':11402,'moving002':11403,'cleanup_contract001':11404,'watchdog002':11405}
def inventory(out):
 entries=[]
 prefix='__log:='+str(out/'roslog')+'/'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
   status=(p/'status').read_text();uid=int(next(l.split()[1] for l in status.splitlines() if l.startswith('Uid:')))
   st=(p/'stat').read_text().split();state=st[2]
   if uid==os.getuid() and state!='Z' and any(x.startswith(prefix) for x in args):
    executable=' '.join(args[:3]);assert any(s in executable for s in ['rosmaster','rosout','quadrotor_simulator_so3','network_control_node']),args
    entries.append(dict(pid=int(p.name),ppid=int(st[3]),pgid=int(st[4]),starttime=st[21],args=args,uid=uid))
  except (FileNotFoundError,ProcessLookupError):pass
 return entries

def cleanup(out):
 out=Path(out).resolve();assert out.parent==ROOT and out.name in ALLOWED,'EXACT_RUN_SCOPE_REQUIRED';before=inventory(out);actions=[]
 for sig,wait_s in [(signal.SIGINT,2.),(signal.SIGTERM,1.),(signal.SIGKILL,.5)]:
  current=inventory(out)
  if not current:break
  for entry in current:
   # Re-read identity before signal to prevent PID reuse; exact args/starttime must still match.
   live={x['pid']:x for x in inventory(out)}.get(entry['pid'])
   if live is not None and live['starttime']==entry['starttime'] and live['args']==entry['args']:
    os.kill(entry['pid'],sig);actions.append(dict(pid=entry['pid'],signal=int(sig),starttime=entry['starttime']))
  deadline=time.monotonic()+wait_s
  while time.monotonic()<deadline and inventory(out):time.sleep(.05)
 after=inventory(out);sock=socket.socket();sock.settimeout(.2);closed=sock.connect_ex(('127.0.0.1',ALLOWED[out.name]))!=0;sock.close()
 result=dict(status='PASS' if not after and closed else 'FAILED',before=before,actions=actions,remaining=after,master_port_closed=closed,scope='only exact __log run path + same UID + rechecked PID identity',correction='original RUN_RESULT only counted Popen wrappers, not roslaunch setsid descendants')
 (out/'DESCENDANT_CLEANUP_AUDIT.json').write_text(json.dumps(result,indent=2));assert result['status']=='PASS',result;return result
