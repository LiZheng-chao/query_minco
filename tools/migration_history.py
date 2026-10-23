"""Read-only old-tree byte snapshots. Outputs only in migration audit namespace."""
import os,json,hashlib,subprocess,argparse,time
from pathlib import Path
R=Path('/home/lizheng/reach_planner')
O=Path(__file__).resolve().parents[1]/'research_outputs/migration_v1/audit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def snapshot():
 rows={}
 for base,dirs,files in os.walk(R,followlinks=False):
  for n in files:
   p=Path(base)/n;s=p.lstat()
   row=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode)
   if p.is_symlink():row['link']=os.readlink(p)
   elif p.is_file():row['sha256']=sha(p)
   rows[str(p.relative_to(R))]=row
 return rows
def git():
 return {n:subprocess.check_output(['git','--no-optional-locks']+a,cwd=R,text=True) for n,a in {
 'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
 'tracked':['diff','--name-status'],'staged':['diff','--cached','--name-status'],
 'status':['status','--porcelain=v1','-uall']}.items()}
def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['before','after']);x=a.parse_args()
 t=time.time();s=snapshot();g=git()
 if x.action=='before':
  out=dict(files=s,git=g,scope='all files including .git; no symlink traversal',semantic_blind_reads=0)
 else:
  old=json.loads((O/'HISTORY_BEFORE.json').read_text());b=old['files']
  changed=[n for n in b.keys()&s.keys() if b[n]!=s[n]]
  added=sorted(s.keys()-b.keys());missing=sorted(b.keys()-s.keys())
  out=dict(passed=not changed and not added and not missing and g==old['git'],
   files_checked=len(b),changed=changed,added=added,missing=missing,git_unchanged=g==old['git'],
   historical_baseline_passed=False,historical_baseline_counts=[1408,103,25],elapsed_s=time.time()-t)
 with (O/('HISTORY_'+x.action.upper()+'.json')).open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k not in ['files','git']}),flush=True)
 if x.action=='after' and not out['passed']:raise SystemExit(1)
if __name__=='__main__':main()
