"""Incremental exact-cell update in the candidate swept-tube ROI."""
import time
from dataclasses import replace
import numpy as np
from query_minco.integration.observed_space_v1.map import Frame,Snapshot,KNOWN_FREE,OCCUPIED,UNKNOWN
from query_minco.integration.observed_space_v2.core import Prepared
from .space import authorized_snapshot

def latest_proof(base,plan,frames,birth_loader,clock):
 now=clock();prepared=getattr(plan,'_spatial_prepared',None) or Prepared(plan);plan._spatial_prepared=prepared;h=base.cfg.resolution
 radius=.30+plan.maxima[0]*float(prepared.times[1]-prepared.times[0])/2
 points=prepared.values[:,0]
 lo=np.floor((points.min(0)-radius-base.origin)/h).astype(int)
 hi=np.floor((points.max(0)+radius-base.origin)/h).astype(int)+1
 if (lo<0).any() or (hi>base.states.shape).any():raise ValueError('TUBE_OUTSIDE_SNAPSHOT')
 shape=tuple((hi-lo).tolist());cfg=replace(base.cfg,shape=shape)
 states=base.states[tuple(slice(a,b) for a,b in zip(lo,hi))].copy()
 origin=base.origin+lo*h
 ij=np.indices(shape).transpose(1,2,3,0).reshape(-1,3)
 centers=origin+(ij+.5)*h;records=list(base.frames)
 retired=[]
 if clock()>min(f['stamp']+base.cfg.history_ttl for f in records):
  # Expired positive evidence cannot be renewed by a timestamp. Withdraw it.
  # Old occupied cells are retained as negative constraints, not free evidence.
  states[states!=OCCUPIED]=UNKNOWN
  retired=[f['id'] for f in records];records=[]
 deadline=time.perf_counter()+.05
 for depth,pose,stamp,rev in frames:
  # Source freshness was checked at the original receipt, not at a later
  # fusion attempt. Snapshot history TTL/current expiry still gate permission.
  frame=Frame(depth,pose,stamp,stamp,pose.get('_received_ros',clock()),'native_static_scene','native_camera')
  free,occ,visible=frame.carve(centers,origin,cfg,deadline)
  # New visible evidence can withdraw older free cells; old occupied evidence
  # remains occupied under the existing static-scene contract.
  old_occ=states==OCCUPIED
  states[visible&~free]=UNKNOWN;states[free]=KNOWN_FREE
  states[occ|old_occ]=OCCUPIED;records.append(frame.record)
 if time.perf_counter()>deadline:raise ValueError('LATEST_EVIDENCE_DEADLINE')
 rev=frames[-1][3] if frames else base.epoch
 sensor=Snapshot(states,origin,cfg,records,clock(),rev,lambda:rev)
 grant=birth_loader(clock())
 snap=authorized_snapshot(sensor,grant,clock(),rev,lambda:rev)
 proof=snap.certify(prepared,clock())
 proof.update(birth_id=snap.birth_id,sensor_snapshot=snap.sensor_snapshot,expired_positive_sources_retired=retired)
 return snap,proof,sensor
