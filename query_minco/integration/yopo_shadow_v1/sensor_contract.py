"""Dynamic native YOPO observation and trajectory frame contract; no GT inputs."""
import numpy as np
from scipy.spatial.transform import Rotation
from query_minco.evaluation.validation import evaluate
COMPONENT_ID='yopo_dynamic_body_optical_state12_contract_v1'
OPTICAL_TO_BODY=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
def rotation(q_xyzw):
 q=np.asarray(q_xyzw,dtype=float)
 if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:raise ValueError('QUATERNION')
 return Rotation.from_quat(q).as_matrix()
def observation(depth,q,velocity_world,acceleration_world,position_world,goal_world,K):
 R=rotation(q);d=np.asarray(depth)
 if d.shape!=(90,160) or d.dtype!=np.float32 or list(K)!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:raise ValueError('DEPTH_CAMERA_CONTRACT')
 valid=np.isfinite(d)&(d>=.04)&(d<20)
 if valid.mean()<.01:raise ValueError('DEPTH_MISSING_OR_INVALID')
 x=np.zeros((1,2,128,160),np.float32);x[0,0,19:109]=np.where(valid,d/20,0);x[0,1,19:109]=valid
 vectors=[np.asarray(z,dtype=float) for z in [velocity_world,acceleration_world,position_world,goal_world]]
 if any(z.shape!=(3,) or not np.isfinite(z).all() for z in vectors):raise ValueError('STATE_NONFINITE')
 v,a,p,g=vectors;s=np.r_[R.T@v,R.T@a,R.T@(g-p),R.T@np.array([0,0,-1.])].astype(np.float32)[None]
 return x,s,R
def sample_world(coefficients,durations,origin,R,t):
 c=np.asarray(coefficients);d=np.asarray(durations);origin=np.asarray(origin);R=np.asarray(R)
 if origin.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(origin).all() or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(R),1,atol=1e-7):raise ValueError('TRAJECTORY_FRAME')
 if c.shape!=(3,3,6) or d.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(d).all() or (d<=0).any():raise ValueError('TRAJECTORY')
 if not np.isfinite(t) or t<0 or t>d.sum()+1e-7:raise ValueError('TRAJECTORY_TIME')
 k=min(int(np.searchsorted(np.cumsum(d),t,side='right')),2);u=t-float(d[:k].sum())
 values=[evaluate(c[k],u,order)@R.T for order in range(3)];values[0]+=origin
 return np.stack(values)
def replay_dynamic(counts,meta,position,q_xyzw):
 # Source-conformance diagnostic only,never network input or online selection.
 R=rotation(q_xyzw).astype(np.float32);pos=np.asarray(position,np.float32);h=np.float32(meta['resolution_m']);origin=np.asarray(meta['grid_origin'],np.float32)
 u,v=np.meshgrid(np.arange(160,dtype=np.float32),np.arange(90,dtype=np.float32))
 y=-(u-80)/80;z=-(v-45)/80;length=np.sqrt(1+y*y+z*z);x=1/length;y=y/length;z=z/length
 step=np.stack([np.full_like(x,.5*h),(y/x)*(.5*h),(z/x)*(.5*h)],-1)@R.T
 point=np.broadcast_to(pos,step.shape).copy();active=np.ones((90,160),bool);result=np.zeros((90,160),np.float32);distance=np.float32(0)
 for _ in range(402):
  distance=np.float32(distance+.5*h);point+=step
  ij=np.floor((point-origin)/h).astype(np.int64);fold=ij.copy()
  for axis,n in enumerate(counts.shape[:2]):
   fold[...,axis]%=2*n-2;fold[...,axis]=np.minimum(fold[...,axis],2*n-2-fold[...,axis])
  iz=ij[...,2];occupied=iz<=0;inside=(iz>0)&(iz<counts.shape[2])
  occupied[inside]=counts[fold[...,0][inside],fold[...,1][inside],iz[inside]]>0
  hit=active&occupied;centers=(ij.astype(np.float32)+.5)*h+origin
  result[hit]=((centers-pos)@R)[...,0][hit];active[hit]=False
  if distance>=20:result[active]=20;active[:]=False
  if not active.any():return result
 raise ValueError('RAY_REPLAY_BUDGET')
