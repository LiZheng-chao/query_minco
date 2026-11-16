"""Exact native occupancy export for independent evaluation;never planner input."""
import json,hashlib
from pathlib import Path
import numpy as np
class Grid:
 def __init__(self,base):
  base=Path(base);self.meta_path=Path(str(base)+'.json');self.path=Path(str(base)+'.u8')
  m=json.loads(self.meta_path.read_text());self.shape=tuple(m['shape']);self.origin=np.asarray(m['origin'],float);self.h=float(m['resolution'])
  if len(self.shape)!=3 or any(type(x)is not int or x<2 for x in self.shape) or np.prod(self.shape)>128000000 or self.h!=np.float32(.1) and abs(self.h-.1)>1e-6:raise ValueError('GRID_CONTRACT')
  if self.path.stat().st_size!=int(np.prod(self.shape)) or self.origin.shape!=(3,) or not np.isfinite(self.origin).all():raise ValueError('GRID_FILE_CONTRACT')
  hasher=hashlib.sha256()
  with self.path.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):hasher.update(chunk)
  self.map_id=hasher.hexdigest()+':'+hashlib.sha256(self.meta_path.read_bytes()).hexdigest()
  self.data=np.memmap(self.path,dtype=np.uint8,mode='r',shape=self.shape);self.data_stat=self.path.stat();self.meta_stat=self.meta_path.stat()
 def clearance(self,p):
  for path,st in [(self.path,self.data_stat),(self.meta_path,self.meta_stat)]:
   q=path.stat()
   if q.st_size!=st.st_size or q.st_mtime_ns!=st.st_mtime_ns:raise ValueError('MAP_CHANGED')
  p=np.asarray(p,float)
  if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('POSITION')
  bound=np.minimum(p-self.origin,self.origin+np.array(self.shape)*self.h-p).min()
  if bound<=0:return -1.
  # native mapQuery treats z voxel<=0 as occupied regardless raw counter.
  bound=min(float(bound),float(p[2]-(self.origin[2]+self.h)))
  if bound<=0:return -1.
  radius=.8;low=np.maximum(0,np.floor((p-radius-self.origin)/self.h).astype(int));high=np.minimum(np.array(self.shape)-1,np.floor((p+radius-self.origin)/self.h).astype(int))
  crop=self.data[tuple(slice(a,b+1) for a,b in zip(low,high))]
  ij=np.argwhere(crop>0)+low
  if len(ij):
   lo=self.origin+ij*self.h;near=np.minimum(np.maximum(p,lo),lo+self.h)
   bound=min(bound,float(np.linalg.norm(near-p,axis=1).min()))
  # Any unqueried cell is at least radius away along an axis.
  return float(min(bound,radius)-1e-5)
