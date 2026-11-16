"""Whole-event-window metrics. No final-heartbeat subsampling or reconstructed samples."""
import numpy as np
def summarize(events,begin=-float('inf'),end=float('inf'),anchor=None):
 es=[e for e in events if begin<=e.get('monotonic',-1)<=end]
 streams={};maxima=dict(speed=0.,acceleration=0.,position_drift=0.,tracking_error=0.);steps=[];faults=[]
 last_reference=None;imu_values=[]
 for e in es:
  k=e.get('kind');d=e.get('payload',{});stamp=e.get('stamp')
  if k=='fault':faults.append(e)
  if not e.get('accepted',False):continue
  if stamp is not None:streams.setdefault(k,[]).append((float(stamp),e['monotonic'],e.get('ros_now',stamp)))
  if k=='request':last_reference=d.get('pva')
  if k=='odom':
   maxima['speed']=max(maxima['speed'],float(np.linalg.norm(d['v'])))
   if anchor is not None:maxima['position_drift']=max(maxima['position_drift'],float(np.linalg.norm(np.array(d['p'])-anchor)))
   if last_reference is not None:maxima['tracking_error']=max(maxima['tracking_error'],float(np.linalg.norm(np.array(d['p'])-np.array(last_reference[0]))))
  if k=='imu':imu_values.append((float(stamp),np.asarray(d['a_world'])))
  if k=='imu':maxima['acceleration']=max(maxima['acceleration'],float(np.linalg.norm(d['a_world'])))
  if k=='plant':steps.append(d['integration_steps'])
 intervals={}
 for k,values in streams.items():
  x=np.array(values);ds=np.diff(x[:,0]);dr=np.diff(x[:,1]);intervals[k]=dict(n=len(values),unique_stamps=len(set(x[:,0])),duplicates=int((ds==0).sum()),rollbacks=int((ds<0).sum()),
   max_source_gap_s=float(ds.max()) if len(ds) else None,max_receipt_gap_s=float(dr.max()) if len(dr) else None,max_age_s=float((x[:,2]-x[:,0]).max()),
   end_receipt_age_s=float(end-x[-1,1]) if np.isfinite(end) else None,end_source_age_s=float(end+np.median(x[:,2]-x[:,1])-x[-1,0]) if np.isfinite(end) else None,
   first_stamp=float(x[0,0]),last_stamp=float(x[-1,0]))
 jerk=[float(np.linalg.norm((b[1]-a[1])/(b[0]-a[0]))) for a,b in zip(imu_values[:-1],imu_values[1:]) if b[0]>a[0]]
 maxima['actual_jerk_fd_diagnostic']=max(jerk or [0.])
 return dict(window_monotonic=[begin if np.isfinite(begin) else None,end if np.isfinite(end) else None],maxima=maxima,intervals=intervals,max_integration_steps=max(steps or [0]),faults=faults)
