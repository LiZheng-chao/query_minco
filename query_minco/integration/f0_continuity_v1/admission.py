"""Original Admission mathematics with asynchronous independent Plan preparation."""
import copy,numpy as np
from query_minco.integration.yopo_execution_v4.trajectory import Admission,Plan,digest
from .worker import Preparation
from .authority import prepare_geometry
def prepare_plan(record):return prepare_geometry(Plan(record))
class ContinuousAdmission(Admission):
 def __init__(self):
  super().__init__();self.preparation=Preparation(prepare_plan);self.pending_id=None;self.hold=None
 def check(self,envelope,state,now,certify=None):
  # Reject malformed/nonfinite request immediately even during pending preparation.
  pva=np.asarray(envelope['pva'],float)
  if pva.shape!=(3,3) or not np.isfinite(pva).all() or np.linalg.norm(pva[1])>1+1e-6 or np.linalg.norm(pva[2])>8+1e-6:raise ValueError('REQUEST_PVA')
  if envelope['kind']=='hold':
   if np.linalg.norm(state['v'])>.1 or np.linalg.norm(state['a'])>.5:raise ValueError('INITIAL_PVA_HOLD')
   if self.plan is not None and any(np.linalg.norm(pva[k]-self.plan.at(now)[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('SWITCH_DISCONTINUITY')
   return super().check(envelope,state,now,certify)
  identity=digest(envelope['plan'])
  if self.plan is not None and self.plan.id==identity:return super().check(envelope,state,now,certify)
  if self.pending_id is None:
   self.preparation.submit(copy.deepcopy(envelope['plan']));self.pending_id=identity
  if identity!=self.pending_id:raise ValueError('PENDING_PLAN_IDENTITY_CHANGED')
  prepared=self.preparation.take()
  if prepared is not None:
   self.pending_id=None
   return super().check(envelope,state,now,certify,prepared=prepared)
  # No new authority while pending: retain an unexpired original plan, or the
  # fixed startup hold; caller must independently certify the returned reference.
  if self.plan is not None:
   retained=dict(kind='trajectory',plan=self.plan.record,stamp=envelope['stamp'],pva=self.plan.at(envelope['stamp']).tolist())
  else:
   if self.hold is None:
    if np.linalg.norm(state['v'])>.1 or np.linalg.norm(state['a'])>.5:raise ValueError('INITIAL_PVA_HOLD')
    self.hold=np.stack([state['p'],np.zeros(3),np.zeros(3)])
   retained=dict(kind='hold',stamp=envelope['stamp'],pva=self.hold.tolist())
  return super().check(retained,state,now,certify)
 def close(self):self.preparation.close()
