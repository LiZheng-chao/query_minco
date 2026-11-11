"""Fail-closed dry-run PositionCommand adapter; this module has no ROS Publisher."""
import numpy as np
from query_minco.integration.yopo_shadow_v1.sensor_contract import sample_world
from query_minco.evaluation.validation import norm_max
COMPONENT_ID='yopo_position_command_dryrun_safety_gate_v1'
class CommandGate:
 def __init__(self,max_age=.2,low_speed_cap=1.):
  self.max_age=max_age;self.low_speed_cap=low_speed_cap;self.aborted=False;self.reason=None
 def abort(self,reason):
  self.aborted=True;self.reason=reason;raise ValueError('ABORT_LATCHED '+reason)
 def sample(self,c,d,p,R,epoch,now,depth_stamp,odom_stamp,current_pva,previous_pva=None):
  if self.aborted:raise ValueError('ABORT_LATCHED '+str(self.reason))
  if any(not np.isfinite(t) or now<t-.01 or now-t>self.max_age for t in [epoch,depth_stamp,odom_stamp]):return self.abort('STALE_OR_FUTURE')
  try:
   target=sample_world(c,d,np.asarray(p),np.asarray(R),now-epoch)
   if any(norm_max(a,t,1)>self.low_speed_cap for a,t in zip(c,d)):return self.abort('LOW_SPEED_GATE')
   if any(max(norm_max(a,t,k) for a,t in zip(c,d))>limit for k,limit in [(2,8),(3,100)]):return self.abort('DYNAMICS')
  except (ValueError,TypeError,IndexError):return self.abort('TRAJECTORY_INVALID_OR_LIMIT')
  actual=np.asarray(current_pva)
  if actual.shape!=(3,3) or not np.isfinite(actual).all():return self.abort('STATE_INVALID')
  if any(np.linalg.norm(target[k]-actual[k])>tol for k,tol in enumerate([.05,.1,.5])):return self.abort('TRACKING_OR_START_PVA')
  if previous_pva is not None and any(np.linalg.norm(target[k]-previous_pva[k])>tol for k,tol in enumerate([.05,.1,.5])):return self.abort('SWITCH_DISCONTINUITY')
  return target
def position_command(message_class,pva,stamp,yaw,trajectory_id):
 if np.asarray(pva).shape!=(3,3) or not np.isfinite(pva).all() or not np.isfinite(yaw) or trajectory_id<1:raise ValueError('COMMAND_FIELDS')
 msg=message_class();msg.header.stamp=stamp;msg.header.frame_id='world'
 for field,vec in zip(['position','velocity','acceleration'],pva):
  dst=getattr(msg,field);dst.x,dst.y,dst.z=map(float,vec)
 msg.yaw=float(yaw);msg.yaw_dot=0.;msg.kx=[5.7,5.7,6.2];msg.kv=[3.4,3.4,4.]
 msg.trajectory_id=trajectory_id;msg.trajectory_flag=msg.TRAJECTORY_STATUS_READY
 return msg
