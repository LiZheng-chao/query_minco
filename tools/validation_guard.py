"""Narrow import-time uname exception; OS mounts remain authoritative."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def install(run,allowed,uname_sha,platform_sha):
 old=Path('/home/lizheng/reach_planner')
 platform_file=Path(sys.base_prefix)/'lib/python3.8/platform.py'
 assert sha('/usr/bin/uname')==uname_sha and sha(platform_file)==platform_sha
 assert os.environ['PATH']=='/usr/bin:/bin' and not os.environ.get('PYTHONPATH')
 events=[]
 def audit(event,args):
  if event=='os.system':raise RuntimeError('DENY_OS_SYSTEM')
  if event=='subprocess.Popen':
   exe,argv,cwd,env=args
   f=sys._getframe(1);origin=False
   while f:
    if f.f_code.co_filename==str(platform_file) and f.f_code.co_name=='_syscmd_uname':origin=True
    f=f.f_back
   if not (exe=='uname' and tuple(argv)==('uname','-p') and cwd is None and env is None and origin):
    raise RuntimeError('DENY_SUBPROCESS')
   events.append(dict(executable='/usr/bin/uname',argv=list(argv),source=str(platform_file),sha256=uname_sha))
  if event=='open' and args and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).absolute()
   if old in p.parents and p.suffix not in ['.py','.pyc','.so'] and str(p) not in allowed:
    raise RuntimeError('DENY_HISTORICAL_DATA '+str(p))
 sys.addaudithook(audit)
 return events
