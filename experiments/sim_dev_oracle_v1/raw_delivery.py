"""SIM-only one-message raw delivery; preserve native stamps and audit every receive."""
import math
import threading
from query_minco.integration.yopo_execution_v4.readiness import Readiness

def fields(m):
 return dict(stamp_ns=m.header.stamp.to_nsec(),audit_aux=[m.aux.current_yaw,m.aux.kf_correction,m.aux.angle_corrections[0],m.aux.angle_corrections[1],float(m.aux.enable_motors),float(m.aux.use_external_yaw),0.],frame=m.header.frame_id,force_world=[m.force.x,m.force.y,m.force.z],q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],kR=list(m.kR),kOm=list(m.kOm),motors_enabled=bool(m.aux.enable_motors))

class LatestRaw:
 def __init__(self,nonce):
  self.nonce=nonce;self.lock=threading.Lock();self.pending=None;self.receipts=[];self.error=None;self.last_ns=None;self.count=0;self.closed=False;self.validator=Readiness()
 def offer(self,m,receipt_ros,receipt_monotonic):
  # No Supervisor journal/Engine lock or file IO in the transport callback.
  meta=dict(nonce=self.nonce,receipt_ros=receipt_ros,receipt_monotonic=receipt_monotonic,publisher=(getattr(m,'_connection_header',None) or {}).get('callerid','UNKNOWN'))
  problem=None
  try:
   stamp=m.header.stamp.to_sec();ns=m.header.stamp.to_nsec();data=fields(m)
   meta.update(source_stamp=stamp,stamp_ns=ns,frame=data['frame'])
   if not all(math.isfinite(t) and t>0 for t in [stamp,receipt_ros]):raise ValueError('raw_STAMP')
   if receipt_ros-stamp>.2:raise ValueError('raw_STALE')
   if stamp-receipt_ros>.01:raise ValueError('raw_STAMP')
   if data['frame']!='/quadrotor':raise ValueError('raw_FRAME')
   self.validator.validate('raw',data)
  except (ValueError,TypeError,KeyError,IndexError,AttributeError) as ex:
   problem=str(ex) if str(ex).startswith('raw_') else 'raw_'+str(ex)
   meta['error']=problem
  with self.lock:
   if self.closed:return
   self.count+=1;meta['receive_sequence']=self.count
   if len(self.receipts)>=64:
    self.error=self.error or 'RAW_DELIVERY_AUDIT_OVERFLOW';return
   if problem is None and self.last_ns is not None and ns<self.last_ns:problem='raw_BACKWARD_STAMP'
   if problem:self.error=self.error or problem;meta['error']=problem
   if problem is None:self.last_ns=ns
   if self.pending is not None:self.pending[1]['superseded_by_receive_sequence']=self.count
   self.receipts.append(meta)
   if not self.error:self.pending=(m,meta)
 def service(self,now,handler,audit):
  with self.lock:
   pending=self.pending;receipts=self.receipts;error=self.error;self.pending=None;self.receipts=[]
  if not receipts and pending is None and error is None:return None
  if pending is not None:
   stamp=pending[0].header.stamp.to_sec()
   if now-stamp>.2:error=error or 'raw_STALE'
   elif stamp-now>.01:error=error or 'raw_STAMP'
  audit(dict(kind='RAW_LATEST_DELIVERY',nonce=self.nonce,processed_ros=now,receipts=receipts,error=error,selected_stamp_ns=pending[1]['stamp_ns'] if pending else None,policy='LATEST_NONEXECUTED_OUTPUT_ONLY_ORIGINAL_STAMP'))
  if error:
   with self.lock:self.error=self.error or error
   raise ValueError(error)
  if pending is not None:
   handler(pending[0]);return pending[1]['stamp_ns']
  return None
 def close(self,now,audit):
  with self.lock:
   self.closed=True;receipts=self.receipts;self.pending=None;self.receipts=[];error=self.error
  audit(dict(kind='RAW_LATEST_CLOSED',nonce=self.nonce,closed_ros=now,unexecuted_receipts=receipts,error=error))
