"""Conservative instantaneous observed-space queries,not GT/free-space completion."""
import numpy as np,heapq
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation,OPTICAL_TO_BODY
class ObservedSpace:
 def __init__(self,depth,pose,stamp):
  self.depth=np.asarray(depth);self.p=np.asarray(pose['p']);self.R=rotation(pose['q']);self.stamp=stamp
  if self.depth.shape!=(90,160) or self.depth.dtype!=np.float32:raise ValueError('DEPTH_CONTRACT')
 def ball_free(self,world,radius):
  optical=(np.asarray(world)-self.p)@self.R@OPTICAL_TO_BODY
  x,y,z=optical;r=float(radius)
  if not np.isfinite(optical).all() or z-r<.04:return False
  # Conservative projected bounding rectangle of the entire ball,not center-pixel clearance.
  us=[80+80*a/b for a in [x-r,x+r] for b in [z-r,z+r]];vs=[45+80*a/b for a in [y-r,y+r] for b in [z-r,z+r]]
  lo=np.floor([min(us),min(vs)]).astype(int);hi=np.ceil([max(us),max(vs)]).astype(int)
  if (lo<0).any() or hi[0]>=160 or hi[1]>=90:return False
  d=self.depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
  return bool(np.isfinite(d).all() and (d>=.04).all() and (d<20).all() and (d>z+r+.1).all())
 def certify(self,plan):
  # Exact original polynomial sampler;speed Lipschitz bound covers time between samples.
  times=np.linspace(0,float(plan.d.sum()),129);padding=plan.maxima[0]*float(times[1]-times[0])/2
  for t in times:
   if not self.ball_free(plan.sample(t)[0],.15+.15+padding):raise ValueError('UNKNOWN_OR_BLOCKED_SWEPT_TUBE')
  return True
 def search_goal(self,start,goal):
  # Deterministic local A*,no predefined left/right candidate identities.
  start=np.asarray(start);goal=np.asarray(goal);h=.2
  directions=[(a,b) for a in [-1,0,1] for b in [-1,0,1] if a or b]
  frontier=[(0.,(0,0))];cost={(0,0):0.};parent={};end=None
  while frontier and len(cost)<2000:
   _,u=heapq.heappop(frontier);p=start+np.array([u[0]*h,u[1]*h,0])
   if np.linalg.norm(p-goal)<.25:end=u;break
   for dx,dy in directions:
    w=(u[0]+dx,u[1]+dy);q=start+np.array([w[0]*h,w[1]*h,0])
    if abs(w[0])>15 or abs(w[1])>10:continue
    n=max(2,int(np.linalg.norm(q-p)/.05)+1)
    clear=True
    for x in np.linspace(p,q,n):
     if not self.ball_free(x,.30+.03):
      if hasattr(self,'edge_rejected'):self.edge_rejected(p,q,x,.30+.03)
      clear=False;break
    if not clear:continue
    new=cost[u]+h*np.hypot(dx,dy)
    if new<cost.get(w,float('inf')):
     cost[w]=new;parent[w]=u;heapq.heappush(frontier,(new+float(np.linalg.norm(q-goal)),w))
  if end is None:raise ValueError('NO_SENSOR_CERTIFIED_LOCAL_ROUTE')
  route=[end]
  while route[-1]!=(0,0):route.append(parent[route[-1]])
  route.reverse();points=[start+np.array([u[0]*h,u[1]*h,0]) for u in route]
  return next((p for p in points[1:] if np.linalg.norm(p-start)>=.5),points[-1])
