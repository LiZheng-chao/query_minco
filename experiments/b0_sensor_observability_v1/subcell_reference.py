"""Offline exact-set-cover refinement. No production or FREE-map changes."""
from pathlib import Path
import sys,json,time,math,itertools,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from experiments.b0_sensor_observability_v1.reference import *
CORNERS=np.array(list(itertools.product([0.,1.],repeat=3)))
class RayCover:
 def __init__(self,depth,pose,max_depth=6):self.depth=depth;self.p=np.array(pose['p']);self.rot=rotation(pose['q'])@OPTICAL_TO_BODY;self.max_depth=max_depth;self.counts=collections.Counter();self.failure=None;self.cache={}
 def ray_free(self,lo,hi):
  key=(tuple(lo),tuple(hi))
  if key in self.cache:return self.cache[key]
  corners=(lo+CORNERS*(hi-lo)-self.p)@self.rot
  # Exact rotation displacement bound for ALL points in this cell. Norm is
  # convex, so corner maximum bounds every interior point. Same angle constant.
  e=CFG.position_error+2*math.sin(CFG.angle_error/2)*np.linalg.norm(corners,axis=1).max();low=corners.min(0)-e;high=corners.max(0)+e
  if low[2]<.04:result=(False,'NEAR_PLANE')
  else:
   u=[80+80*x/z for x in [low[0],high[0]] for z in [low[2],high[2]]];v=[45+80*y/z for y in [low[1],high[1]] for z in [low[2],high[2]]];pad=1.5+CFG.pixel_guard;left=math.floor(min(u)-pad);right=math.ceil(max(u)+pad);top=math.floor(min(v)-pad);bottom=math.ceil(max(v)+pad)
   if left<0 or right>=160 or top<0 or bottom>=90:result=(False,'FOV')
   else:
    w=self.depth[top:bottom+1,left:right+1];valid=np.isfinite(w)&(w>=.04)&(w<20)
    if not valid.all():result=(False,'NO_VALID_DEPTH_RETURN')
    elif w.min()<=high[2]+CFG.depth_error:result=(False,'OCCLUSION_OR_MARGIN')
    else:result=(True,'DEPTH_FULL_SUBCELL')
  self.cache[key]=result;return result
 def cover(self,lo,hi,p,r,birth,level=0):
  near=np.minimum(np.maximum(p,lo),hi)
  if np.linalg.norm(near-p)>r+1e-12:self.counts['OUTSIDE_QUERY']+=1;return True
  d=p-np.asarray(birth.center)
  # For x in query ball: |x-b|^2=|x-p|^2+2d.(x-p)+|d|^2.
  # Box support + r^2 upper-bound the ENTIRE intersection, including curved
  # boundaries that the original independent-axis AABB cannot exactly cover.
  support=np.where(d>=0,hi-p,lo-p);upper=r*r+2*float(np.dot(d,support))+float(np.dot(d,d))
  # Exact maximum for this experiment's forward collinear query/birth
  # centers: reduce perpendicular box coordinates to s in [smin,smax].
  # For x>=0,d>=0, min(x*x+smax,r*r+2*d*x-d*d) is monotone in x.
  axial_upper=float('inf')
  relative_lo=lo-np.asarray(birth.center);relative_hi=hi-np.asarray(birth.center)
  if d[0]>=0 and abs(d[1])+abs(d[2])<1e-12 and relative_lo[0]>=0:
   yzlo=relative_lo[1:];yzhi=relative_hi[1:];smin=float(np.maximum(np.maximum(yzlo,-yzhi),0).dot(np.maximum(np.maximum(yzlo,-yzhi),0)));smax=float(np.maximum(np.abs(yzlo),np.abs(yzhi)).dot(np.maximum(np.abs(yzlo),np.abs(yzhi))))
   if smin<=r*r:
    xmax=min(relative_hi[0],d[0]+math.sqrt(max(0,r*r-smin)))
    axial_upper=xmax*xmax+min(smax,r*r-(xmax-d[0])**2)
  if min(upper,axial_upper)<=birth.radius**2-1e-12 or birth_covers_intersection(lo[None],hi[None],p,r,birth)[0]:self.counts['BIRTH_INTERSECTION']+=1;return True
  ok,reason=self.ray_free(lo,hi)
  if ok:self.counts['DEPTH_FULL_SUBCELL']+=1;return True
  if level==self.max_depth:
   self.counts['UNRESOLVED_'+reason]+=1;self.failure=dict(lo=lo.tolist(),hi=hi.tolist(),reason=reason,level=level);return False
  mid=(lo+hi)/2
  for bits in CORNERS:
   a=np.where(bits==0,lo,mid);b=np.where(bits==0,mid,hi)
   if not self.cover(a,b,p,r,birth,level+1):return False
  return True

def main():
 torch.set_num_threads(1);item=json.loads((BASE/'capture_ordered/INPUTS.json').read_text())['records'][0];now=item['stamp'];dep=np.load(item['path'])['depth'];ledger=ObservedMap(CFG);ledger.add(dep,item['pose'],now,now,now);sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,BirthGrant(tuple(SPAWN),.6,now+.2,'OFFLINE_ONLY'),now,sensor.epoch,lambda:sensor.epoch);ray=RayCover(dep,item['pose']);results=[]
 def query(p,r):
  low=np.floor((p-r-snap.origin)/.1).astype(int);high=np.floor((p+r-snap.origin)/.1).astype(int)
  if (low<0).any() or (high>=CFG.shape).any():return dict(passed=False,reason='OUTSIDE_GRID')
  ids=np.array(list(itertools.product(*[range(a,b+1) for a,b in zip(low,high)])));lo=snap.origin+ids*.1;near=np.minimum(np.maximum(p,lo),lo+.1);intersect=np.linalg.norm(near-p,axis=1)<=r+1e-12
  for ij,a in zip(ids[intersect],lo[intersect]):
   val=int(snap.states[tuple(ij)])
   if val==2:return dict(passed=False,reason='OCCUPIED_ENVELOPE',cell=a.tolist())
   if val==1:continue
   if not ray.cover(a,a+.1,p,r,snap.birth_ball):return dict(passed=False,reason='UNRESOLVED',detail=ray.failure)
  return dict(passed=True)
 begin=time.perf_counter();q=query(np.array([.55,-.25,2.]),.33);results.append(dict(kind='FIRST_FAILED_QUERY',p=[.55,-.25,2.],r=.33,result=q,elapsed_s=time.perf_counter()-begin,counts=dict(ray.counts)))
 for distance in [.6,.8]:
  plan=build(SPAWN,np.zeros(3),np.zeros(3),[0,0,0,1],SPAWN+[distance,0,0],now);times=np.linspace(0,float(plan.d.sum()),129);radius=.30+plan.maxima[0]*(times[1]-times[0])/2;begin=time.perf_counter();failure=None;checked=0
  for t in times:
   p=plan.sample(t)[0];result=query(p,radius);checked+=1
   if not result['passed']:failure=dict(t=float(t),p=p.tolist(),r=float(radius),result=result);break
  results.append(dict(kind='MINCO_FULL_TUBE',distance=distance,plan_id=plan.id,passed=failure is None,checked_samples=checked,failure=failure,elapsed_s=time.perf_counter()-begin,counts=dict(ray.counts)))
 # Seek an ACTUAL uncovered query point in each failing subcell, not merely
 # an AABB corner that lies outside the queried tube.
 witnesses=[]
 for result in results:
  if result['kind']=='FIRST_FAILED_QUERY':p=np.array(result['p']);radius=result['r'];detail=result['result'].get('detail')
  else:
   fail=result.get('failure');p=np.array(fail['p']) if fail else None;radius=fail['r'] if fail else None;detail=fail['result'].get('detail') if fail else None
  if not detail:continue
  lo=np.array(detail['lo']);hi=np.array(detail['hi']);points=np.array(list(itertools.product(*[np.linspace(a,b,21) for a,b in zip(lo,hi)])));eligible=(np.linalg.norm(points-p,axis=1)<radius-1e-10)&(np.linalg.norm(points-SPAWN,axis=1)>.6+1e-10)
  witness=None
  for point in points[eligible]:
   optical=(point-SPAWN)@ray.rot;u=int(np.rint(80+80*optical[0]/optical[2]));v=int(np.rint(45+80*optical[1]/optical[2]));value=float(dep[v,u]);ok,why=ray.ray_free(point,point)
   if not ok:
    witness=dict(world=point.tolist(),distance_to_birth=float(np.linalg.norm(point-SPAWN)),distance_to_query=float(np.linalg.norm(point-p)),query_radius=float(radius),nominal_pixel=[u,v],nominal_depth=value,point_uncertainty_reference_reason=why,nominal_no_return=bool(not np.isfinite(value) or value<.04 or value>=20))
    if witness['nominal_no_return']:break
  witnesses.append(dict(kind=result['kind'],distance=result.get('distance'),actual_uncovered_point=witness,eligible_samples=int(eligible.sum())))
 bound=math.degrees(math.acos((.6**2+.3**2-.33**2)/(2*.6*.3)))
 write('SUBCELL_EXACT_SET_COVER_REFERENCE.json',dict(scope='OFFLINE_ONLY_NO_PRODUCTION_FREE_RELABEL_OR_EXECUTION_AUTHORITY',source_frames=69,unique_depth_contents=1,max_subdivision_depth=ray.max_depth,min_cell_edge=.1/2**ray.max_depth,query_outside_birth_nominal_cone_halfangle_deg=bound,original_vertical_half_fov_deg=math.degrees(math.atan(45/80)),results=results,actual_uncovered_point_witnesses=witnesses,proof='Each intersecting cell piece is either outside query ball, wholly/rigorously birth-intersection covered, or entirely enclosed by a conservative projected subcell with all finite valid ray returns and depth margin. All original occupied envelopes reject before subdivision. Unresolved subdivision is UNKNOWN. Same pose/depth/angle/pixel constants; exact rotation chord bound is at least as conservative as prior sin bound.',GT_used=False,exit_code=0));print('SUBCELL',results,flush=True)
if __name__=='__main__':main()
