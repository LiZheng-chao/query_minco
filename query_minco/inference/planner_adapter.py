"""Experimental CPU wrapper. Original numerical components, serial batch semantics."""
import time
import numpy as np
import torch
from query_minco.registry import registration, METHOD_ID
from query_minco.inference.identity import state_sha
class InferenceAdapter:
 def __init__(self,checkpoint,method_id=METHOD_ID,expected_checkpoint_sha256=None):
  binding,self.config=registration(method_id,checkpoint,expected_checkpoint_sha256)
  from query_minco.models.planner import QueryMincoPlanner
  from query_minco.trajectory.postprocessor import TrajectoryPostprocessor
  torch.use_deterministic_algorithms(True)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.set_num_threads(1)
  blob=torch.load(str(checkpoint),map_location="cpu",weights_only=True)
  if blob["seed"]!=42 or blob["arm"]!="C1_orthogonal" :
   raise ValueError("CHECKPOINT_IDENTITY")
  if blob["step"]!=2880 or blob["optimizer_updates_completed"]!=2880:raise ValueError("CHECKPOINT_STEP")
  self.model=QueryMincoPlanner()
  loaded=self.model.load_state_dict(blob["state_dict"],strict=True)
  self.strict_load=dict(missing_keys=list(loaded.missing_keys),unexpected_keys=list(loaded.unexpected_keys))
  if state_sha(self.model)!=binding["weights_sha256"]:raise ValueError("STATE_DICT_SHA_MISMATCH")
  self.model.to(torch.device("cpu")).eval();self.post=TrajectoryPostprocessor();self.binding=binding;self.timings=[]
 @staticmethod
 def inputs(depth,state,times):
  if not isinstance(depth,np.ndarray) or not isinstance(state,np.ndarray):raise ValueError("numpy arrays required")
  if depth.ndim!=4 or depth.shape[1:]!=(2,128,160) or state.shape!=(len(depth),12) or len(depth)<1:raise ValueError("INPUT_SHAPE")
  if depth.dtype!=np.float32 or state.dtype!=np.float32 or not np.isfinite(depth).all() or not np.isfinite(state).all():raise ValueError("INPUT_FP32_FINITE")
  if not np.all((depth[:,1]==0)|(depth[:,1]==1)) or not np.all((depth[:,0]>=0)&(depth[:,0]<1)) or np.any(depth[:,0][depth[:,1]==0]!=0):raise ValueError("DEPTH_MASK")
  if times is not None:
   if not isinstance(times,np.ndarray) or times.dtype!=np.float32 or times.ndim!=3 or times.shape[:2]!=(len(depth),6) or times.shape[2]<1:raise ValueError("TIMES_SHAPE_DTYPE")
   if not np.isfinite(times).all() or np.any(times<0) or np.any(times>2.2):raise ValueError("TIMES_RANGE")
 @staticmethod
 def export(out,raw,sample):
  p=out.parameters;u=out.samples
  fields=dict(raw=raw,head_pva=p.head_pva,tail_pva=p.tail_pva,inner_pts=p.inner_pts,total_time=p.total_time,
   durations=p.durations,coefficients=out.trajectory.coeffs,solve_valid=out.trajectory.solve_valid,
   uniform_times=u.times,uniform_p=u.p,uniform_v=u.v,uniform_a=u.a,uniform_jerk=u.jerk,
   requested_times=sample.times,p=sample.p,v=sample.v,a=sample.a,jerk=sample.jerk,
   segment_indices=sample.segment_indices,local_times=sample.local_times)
  return {k:x.detach().cpu().numpy().copy() for k,x in fields.items()}
 def forward(self,depth_input,state,times=None):
  self.inputs(depth_input,state,times);rows=[]
  for i in range(len(depth_input)):
   d=torch.from_numpy(depth_input[i].copy())[None];s=torch.from_numpy(state[i].copy())[None]
   start=time.perf_counter()
   with torch.inference_mode(),torch.autocast(device_type="cpu",enabled=False):
    raw=self.model(d,s);net=time.perf_counter();out=self.post(raw,s,num_samples=65)
    sampled=out.samples if times is None else self.post.sampler(out.trajectory,times=torch.from_numpy(times[i:i+1].copy()))
    rows.append(self.export(out,raw,sampled))
   self.timings.append(dict(network_s=net-start,total_s=time.perf_counter()-start))
  return {k:np.concatenate([row[k] for row in rows],axis=0) for k in rows[0]}
 def sample(self,result,times):
  from query_minco.trajectory.types import MincoTrajectory
  with torch.inference_mode(),torch.autocast(device_type="cpu",enabled=False):
   tr=MincoTrajectory(torch.from_numpy(result["coefficients"].copy()),torch.from_numpy(result["durations"].copy()),
                      torch.from_numpy(result["solve_valid"].copy()),{})
   s=self.post.sampler(tr,times=torch.from_numpy(times.copy()))
   return {k:getattr(s,k).cpu().numpy().copy() for k in ["times","p","v","a","jerk","segment_indices","local_times"]}

 def plan(self,depth_input,state,times=None):
  return self.forward(depth_input,state,times)

def load_planner(method_id,checkpoint):
 return InferenceAdapter(checkpoint=checkpoint,method_id=method_id)
