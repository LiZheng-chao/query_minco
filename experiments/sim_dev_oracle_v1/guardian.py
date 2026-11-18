"""Independent simulation GT prefix + reaction + continuous brake monitor."""
import time,math,hashlib,json
import numpy as np
from query_minco.integration.native_grid_truth_v1.grid import Grid
from query_minco.integration.yopo_execution_v4.trajectory import Plan,digest
from query_minco.evaluation.validation import norm_max,evaluate
from query_minco.integration.observed_space_v2.core import sample_batch
MODE='SIM_DEV_ORACLE'
class Guardian:
 def __init__(self,grid,nonce):self.grid=Grid(grid) if isinstance(grid,(str,__import__('pathlib').Path)) else grid;self.nonce=nonce;self.serial=0;self.cache={};self.plans={}
 def _clearance(self,p):
  key=tuple(np.floor(np.asarray(p)/.02).astype(int));center=(np.array(key)+.5)*.02
  if key not in self.cache:
   if len(self.cache)>=8192:self.cache.clear()
   self.cache[key]=self.grid.clearance(center)
  return self.cache[key]-float(np.linalg.norm(np.asarray(p)-center))
 def _tube(self,sample,duration,max_speed,extra_margin=0.,batch=None):
  count=max(2,int(math.ceil(duration/.01))+1);ts=np.linspace(0,duration,count);pad=max_speed*(ts[1]-ts[0])/2;radius=.30+extra_margin+pad
  points=batch(ts) if batch is not None else [np.asarray(sample(float(t))[0]) for t in ts]
  clear=min(self._clearance(p) for p in points)
  if clear<radius:raise ValueError('ORACLE_PREFIX_OR_BRAKE_OCCUPIED')
  return dict(samples=count,clearance_lower_bound=clear,radius=radius,duration=duration)
 def check(self,record,state,now,source_stamp,revision,*,prefix=.10,delay=.35):
  if hasattr(self.grid,'data_stat'):
   for path,st in [(self.grid.path,self.grid.data_stat),(self.grid.meta_path,self.grid.meta_stat)]:
    current=path.stat()
    if current.st_size!=st.st_size or current.st_mtime_ns!=st.st_mtime_ns:raise ValueError('MAP_CHANGED')
  if not 0<prefix<=.10 or delay!=.35:raise ValueError('PREFIX_DELAY_CONTRACT')
  if not 0<=now-source_stamp<=.2:raise ValueError('SENSOR_STALE')
  pid=digest(record)
  if pid not in self.plans:
   if len(self.plans)>=64:self.plans.clear()
   self.plans[pid]=Plan(record)
  plan=self.plans[pid];t=now-plan.epoch
  if not 0<=t<=float(plan.d.sum())-prefix-delay:raise ValueError('PLAN_HORIZON')
  expected=plan.sample(t);actual=np.array([state['p'],state['v'],state['a']])
  if actual.shape!=(3,3) or not np.isfinite(actual).all():raise ValueError('STATE_CONTRACT')
  if np.linalg.norm(actual[0]-expected[0])>.15 or np.linalg.norm(actual[1]-expected[1])>.1 or np.linalg.norm(actual[2]-expected[2])>.5:raise ValueError('TRACKING_PVA')
  reaction=prefix+delay;ev=.1;ea=.5
  reaction_position_error=ev*reaction+.5*ea*reaction**2
  # Independent simulator freeze envelope: even an arbitrary admissible
  # direction at speed<=1 stays within this ball until lease+STOP deadline.
  if np.linalg.norm(actual[1])>1+1e-7 or np.linalg.norm(actual[2])>8+1e-7:raise ValueError('ACTUAL_DYNAMICS')
  fault_radius=.30+1.*reaction
  fault_clear=self._clearance(actual[0])
  if fault_clear<fault_radius:raise ValueError('ORACLE_FAULT_STOP_REACHABILITY')
  prefix_proof=self._tube(lambda s:plan.sample(t+s),reaction,plan.maxima[0],reaction_position_error,batch=lambda ss:sample_batch(plan,t+ss)[:,0])
  head=plan.sample(t+prefix+delay);p,v,a=head;T=.6
  # Quartic position: v(T)=a(T)=0, initial PVA unchanged. All derivatives
  # checked continuously by the existing polynomial norm extrema routine.
  coeff=np.zeros((3,6));coeff[:,0]=p;coeff[:,1]=v;coeff[:,2]=a/2;coeff[:,3]=-v/T**2-2*a/(3*T);coeff[:,4]=v/(2*T**3)+a/(4*T**2)
  extrema=[norm_max(coeff,T,k) for k in [1,2,3]]
  if any(x>limit+1e-7 for x,limit in zip(extrema,[1,8,100])):raise ValueError('BRAKE_CONTINUOUS_DYNAMICS')
  ev_end=ev+ea*reaction
  robust_extrema=[extrema[0]+ev_end+ea*T*4/27,extrema[1]+ev_end*1.5/T+ea,extrema[2]+ev_end*6/T**2+ea*4/T]
  if any(x>limit+1e-7 for x,limit in zip(robust_extrema,[1,8,100])):raise ValueError('BRAKE_UNCERTAINTY_DYNAMICS')
  brake_extra=reaction_position_error+.5*T*ev_end+T*T*ea/12
  brake=self._tube(lambda s:np.array([evaluate(coeff,s,k) for k in [0,1,2]]),T,robust_extrema[0],brake_extra,batch=lambda ss:evaluate(coeff,ss,0));self.serial+=1
  proof=dict(mode=MODE,type='SIM_DEV_ORACLE_PREFIX_BRAKE',plan_id=plan.id,source_stamp=source_stamp,revision=revision,issued=now,expires=min(now+prefix,source_stamp+.2),prefix=prefix,max_delay=delay,fault_stop_radius=fault_radius,fault_stop_clearance=fault_clear,fault_stop_scope='ADMISSIBLE_SPEED_LE_1_SIMULATOR_FREEZE_NOT_REAL_UAV_BRAKING',tracking_position_margin=.15,vehicle_radius=.15,nonce=self.nonce,serial=self.serial,map_id=self.grid.map_id,prefix_proof=prefix_proof,brake=brake,brake_coefficients=coeff.tolist(),brake_duration=T,brake_maxima=extrema,robust_brake_maxima=robust_extrema,brake_extra_margin=brake_extra,assumed_pva_error_bounds=[.15,.1,.5],runtime_tracking_and_stop_qualification_required=True,strict_depth_qualified=False)
  proof['map_files']=[dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns) for path,st in [(self.grid.path,self.grid.data_stat),(self.grid.meta_path,self.grid.meta_stat)]] if hasattr(self.grid,'data_stat') else []
  proof['identity']=digest(proof);return proof
 def valid(self,proof,plan_id,now,revision,nonce):
  d=dict(proof);ident=d.pop('identity',None)
  return bool(digest(d)==ident and proof['mode']==MODE and nonce==self.nonce==proof['nonce'] and plan_id==proof['plan_id'] and revision==proof['revision'] and proof['issued']<=now<=proof['expires'])
