"""Exact dtype/shape/C-order bytes, including signed zero. Never relax."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def main():
 a=argparse.ArgumentParser();a.add_argument('old');a.add_argument('new');a.add_argument('output');x=a.parse_args()
 old,new,out=map(Path,[x.old,x.new,x.output]);out.parent.mkdir(parents=True,exist_ok=True)
 results=[]
 assert json.loads((old/'RESULT.json').read_text())['passed'] and json.loads((new/'RESULT.json').read_text())['passed']
 files={p.name for p in old.glob('*.npz')};assert files=={p.name for p in new.glob('*.npz')}
 for name in sorted(files):
  with np.load(old/name,allow_pickle=False) as a,np.load(new/name,allow_pickle=False) as b:
   assert set(a.files)==set(b.files),(name,'keys')
   for k in sorted(a.files):
    u,v=a[k],b[k];same=u.shape==v.shape and u.dtype==v.dtype and u.tobytes()==v.tobytes()
    z=dict(file=name,field=k,shape=list(u.shape),dtype=str(u.dtype),bitwise_equal=same)
    if u.shape==v.shape and u.size:
     z['max_abs_error']=float(np.max(np.abs(u.astype(float)-v.astype(float))))
     if not same and u.dtype==v.dtype:
      d=np.flatnonzero(np.frombuffer(u.tobytes(),np.uint8)!=np.frombuffer(v.tobytes(),np.uint8))
      if len(d):z['first_different_index']=list(map(int,np.unravel_index(int(d[0])//u.dtype.itemsize,u.shape)))
    results.append(z)
 passed=all(z['bitwise_equal'] for z in results) and bool(results)
 report=dict(passed=passed,comparison_count=len(results),comparison_standard='bitwise; no tolerance',comparisons=results)
 with out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(dict(passed=passed,comparison_count=len(results),output=str(out))))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
