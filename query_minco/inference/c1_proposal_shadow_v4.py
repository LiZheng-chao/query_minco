"""CPU-prepared source-binding correction; NOT deployed in execution007.
An explicit minimal overlay preserves the exact v3 evidence/source binding.
The generated main uses trusted source-time100Hz state, never camera-frame0.
"""
import inspect
from query_minco.inference import c1_proposal_shadow_v3 as original
from query_minco.inference.causal_source_state import SourceBoundInbox
def corrected_main_source():
 s=inspect.getsource(original.main)
 old="""if previous_frame is None:previous_frame=(t,vel);continue
   dt=t-previous_frame[0];previous_frame=(t,vel.copy())
   if not .005<=dt<=.3:rejects['SOURCE_ACCEL_DT']+=1;continue
   acc=(vel-previous_frame[1])/dt;tilt=np.arccos(np.clip(R[2,2],-1,1))"""
 new="""try:
    source=inbox.source(t)
    source_pva=np.array(list(map(float,source['pva']))).reshape(3,3)
    source_rotation=np.array(list(map(float,source['rotation']))).reshape(3,3)
    if float(source['stamp'])!=t or float(source['recv_mono'])>time.monotonic():raise ValueError('SOURCE_CAUSAL_RECEIPT')
    if not np.allclose(source_pva[0],pos,atol=1e-7,rtol=0) or not np.allclose(source_rotation,R,atol=1e-7,rtol=0):raise ValueError('SOURCE_POSE_MISMATCH')
    vel=source_pva[1].copy();acc=source_pva[2].copy()
   except ValueError as exc:rejects[str(exc)]+=1;continue
   tilt=np.arccos(np.clip(R[2,2],-1,1))"""
 if s.count(old)!=1:raise ValueError('SOURCE_CORRECTION_BINDING_CHANGED')
 s=s.replace(old,new).replace('inbox=Inbox(','inbox=SourceBoundInbox(')
 old="envelope=metadata(bank,snapshot,index*6+qi+1,t,pos,R,activation,activation_mono,config)"
 new=old+"\n     envelope.update(source_state_version=2,source_pva=source_pva.reshape(-1).tolist(),network_state=state.tolist(),network_state_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest())"
 if s.count(old)!=1:raise ValueError('ENVELOPE_BINDING_CHANGED')
 return s.replace(old,new)
def main():
 env=dict(vars(original));env['SourceBoundInbox']=SourceBoundInbox
 exec(compile(corrected_main_source(),__file__+'[source-binding-overlay]','exec'),env);env['main']()
if __name__=='__main__':main()
