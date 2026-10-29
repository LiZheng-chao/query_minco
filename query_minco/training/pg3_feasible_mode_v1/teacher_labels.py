"""Labels are built only from authenticated geometry and ORIGINAL-hard-certified Teachers."""
import numpy as np
import torch
from query_minco.training.io import require,digest
from query_minco.training.pg2_guided_v1.geometry import check_curve,tensor_digest,scene_from_field
from query_minco.evaluation.validation import evaluate
from query_minco.evaluation.signed_geometry import numpy_box_sdf
from query_minco.trajectory.parameter_decoder import PhysicalParameterConfig
from .corridor import build_corridor,validate_scene
from .bottleneck import crossing_events

def certificate_diagnostics(value,certified_empty):
 if isinstance(value,dict):return {k:certificate_diagnostics(v,certified_empty) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [certificate_diagnostics(v,certified_empty) for v in value]
 if isinstance(value,(float,np.floating)) and not np.isfinite(value):
  require(certified_empty and np.isinf(value),"PG3_NONFINITE_CERTIFICATE_METADATA")
  return None
 return value

def representability(inner,tail,durations):
 cfg=PhysicalParameterConfig();p=np.concatenate([np.asarray(inner),np.asarray(tail)[:1]])
 d=np.asarray(durations);tail=np.asarray(tail)
 valid=dict(positions=bool((p>np.asarray(cfg.position_min_m)).all() and (p<np.asarray(cfg.position_max_m)).all()),
  velocity=bool((abs(tail[1])<np.asarray(cfg.terminal_velocity_axis_max_mps)).all()),
  acceleration=bool((abs(tail[2])<np.asarray(cfg.terminal_acceleration_axis_max_mps2)).all()),
  duration=bool((d>cfg.min_segment_time_s).all() and cfg.min_total_time_s<sum(d)<cfg.max_total_time_s))
 require(np.isfinite(p).all() and np.isfinite(tail).all() and np.isfinite(d).all(),"PG3_TEACHER_NONFINITE")
 return valid

def inverse_raw(inner,tail,durations,scores=(0.,0.)):
 flags=representability(inner,tail,durations);require(all(flags.values()),"PG3_TEACHER_NOT_DECODER_REPRESENTABLE")
 cfg=PhysicalParameterConfig();p=np.concatenate([inner,np.asarray(tail)[:1]])
 raw=np.zeros(21);lower=np.asarray(cfg.position_min_m);upper=np.asarray(cfg.position_max_m)
 raw[:9]=np.arctanh(2*(p-lower)/(upper-lower)-1).reshape(-1)
 raw[9:12]=np.arctanh(np.asarray(tail)[1]/np.asarray(cfg.terminal_velocity_axis_max_mps))
 raw[12:15]=np.arctanh(np.asarray(tail)[2]/np.asarray(cfg.terminal_acceleration_axis_max_mps2))
 total=float(sum(durations));u=(total-cfg.min_total_time_s)/(cfg.max_total_time_s-cfg.min_total_time_s)
 raw[15]=np.log(u/(1-u));w=(np.asarray(durations)-cfg.min_segment_time_s)/(total-3*cfg.min_segment_time_s)
 raw[16:19]=np.log(w)-np.log(w).mean();raw[19:]=scores
 return raw

def landmarks(c,d,scene,near_range=.5):
 result=[];crossings=[]
 for bi,box in enumerate(scene["obstacles"]):
  # A geometry-derived obstacle-layer plane, never a historical fixed x.
  x=(box["min"][0]+box["max"][0])/2
  event=crossing_events(c,d,x);crossings.append(dict(box=bi,plane_x=x,status=event["status"],crossings=len(event["events"])))
  selected=event["selected"]
  if selected is None:continue
  z=np.asarray(selected["position"])
  distance=float(numpy_box_sdf(z[None],box)[0]-.12)
  if distance<.03-1e-6 or distance>near_range:continue
  progress=selected["time_s"]/float(sum(d))
  if not 1e-6<progress<1-1e-6:continue
  faces=[]
  for axis in (1,2):
   if z[axis]<=box["min"][axis]-.15:faces.append((box["min"][axis]-.15-z[axis],axis,-1))
   if z[axis]>=box["max"][axis]+.15:faces.append((z[axis]-box["max"][axis]-.15,axis,1))
  face=max(faces)[1:] if faces else None
  result.append(dict(valid=True,box=bi,plane_x=x,position=z.tolist(),normal=[1.,0.,0.],progress=progress,
   source="AUTHENTICATED_GEOMETRY_CERTIFIED_TEACHER_FORWARD_SECTION",event_status=event["status"],
   side=list(face) if face else None))
 return result,crossings

class TeacherBank:
 def __init__(self,cfg):self.cfg=cfg;self.cache={}
 def get(self,sample,field):
  scene=scene_from_field(field);empty=field.row["empty_certified"];validate_scene(scene,empty)
  key=digest(dict(tensors=tensor_digest(sample.teacher_coeffs,sample.teacher_durations,sample.state),scene=scene,cfg=self.cfg))
  if key not in self.cache:
   rows=[];head=np.stack([np.zeros(3),sample.state[:3].numpy(),sample.state[3:6].numpy()])
   for m,(cc,dd) in enumerate(zip(sample.teacher_coeffs.detach().double().numpy(),sample.teacher_durations.detach().double().numpy())):
    check=check_curve(cc,dd,scene,head);require(check["passed"],"PG3_TEACHER_NOT_HARD_CERTIFIED_"+sample.sample_id)
    inner=np.stack([evaluate(cc[i],dd[i]) for i in (0,1)])
    tail=np.stack([evaluate(cc[-1],dd[-1],k) for k in (0,1,2)])
    flags=representability(inner,tail,dd)
    labels,events=landmarks(cc,dd,scene,self.cfg["bottleneck_range_m"])
    corridor=build_corridor(cc,dd,scene,empty,self.cfg["corridor_max_depth"])
    rows.append(dict(teacher=m,coeffs=cc.tolist(),durations=dd.tolist(),inner=inner.tolist(),tail=tail.tolist(),
     certified=True,check=certificate_diagnostics(check,empty),geometry_bound_semantics="UNBOUNDED_CERTIFIED_EMPTY" if empty else "FINITE_AABB_BOUNDS",representability=flags,representable=flags["positions"] and flags["velocity"],
     bottlenecks=labels,crossing_diagnostics=events,corridor=corridor,
     mode_signature=[(x["box"],*x["side"]) for x in labels if x["side"] is not None],
     mode_label_source="CERTIFIED_TEACHER_AABB_FACE_SIGNATURE_NOT_HOMOTOPY" if any(x["side"] for x in labels) else "TEACHER_DIVERSITY_ONLY",
     cache_identity=key))
   self.cache[key]=dict(teachers=rows,scene=scene,empty_certified=empty,identity=key,status="CERTIFIED" if rows else "NO_TEACHER_UNKNOWN")
  return self.cache[key]
