"""Explicit training-only known-World boundary proxy; never used by Hard."""
import numpy as np
import torch
from .geometry import TypedBoundOccupancy,world_points
METHOD_ID="physics_only_v1_2_p0c_world_known_boundary_proxy_v1"
COMPONENT_ID="exact_known_world_trace_minus_roi_projection_distance_v1"

class RoiSafeTrainingGeometry:
    method_id=METHOD_ID
    component_id=COMPONENT_ID
    def __init__(self):self.records=[]
    def query(self,points,g):
        if type(g) is not TypedBoundOccupancy:raise ValueError("PRODUCTION_GEOMETRY_REQUIRED")
        g.validate()
        if (points.device.type!="cpu" or points.dtype not in (torch.float32,torch.float64) or
                points.shape[-1]!=3 or not bool(torch.isfinite(points).all())):
            raise ValueError("P0B_POINT_TYPE_OR_FINITE")
        shape=points.shape[:-1];flat=points.reshape(-1,3)
        w=world_points(flat,torch.as_tensor(g.origin,dtype=points.dtype),torch.as_tensor(g.rotation,dtype=points.dtype))
        x=w.detach().double().numpy()
        outside=torch.as_tensor(((x<g.roi[0])|(x>g.roi[1])).any(-1))
        low=torch.as_tensor(g.roi[0],dtype=torch.float64);high=torch.as_tensor(g.roi[1],dtype=torch.float64)
        if not bool(torch.isfinite(low).all() and torch.isfinite(high).all()) or bool((high<=low).any()):
            raise ValueError("INVALID_CERTIFIED_ROI")
        distance=None;delta=torch.zeros(len(flat),dtype=torch.float64)
        q=w.new_empty((0,3),dtype=torch.float64);trace=w.new_empty((0,),dtype=torch.float64);cast_error=0.
        if not bool(outside.any()):
            # Do not even mask/reorder this legacy fully known branch.
            distance=g.sdf(points)
        else:
            values=torch.zeros(len(flat),dtype=points.dtype)
            if bool((~outside).any()):values[~outside]=g.sdf(flat[~outside])
            wd=w[outside].double()
            q=torch.maximum(torch.minimum(wd,high),low)
            trace=g._sdf_world_known(q)
            d=(wd-q).norm(dim=-1)
            proxy=trace-d
            if bool((trace>0).any()) or bool((proxy>0).any()):raise ValueError("PROXY_SIGN")
            cast=proxy.to(points.dtype)
            cast_error=float((cast.double()-proxy).detach().abs().max())
            values[outside]=cast;delta[outside]=d
            distance=values.reshape(shape)
        # Diagnostic only: never replace original membership with a higher-precision transform.
        wd2=world_points(flat.double(),torch.as_tensor(g.origin),torch.as_tensor(g.rotation))
        mask64=((wd2<low)|(wd2>high)).any(-1)
        boundary=torch.minimum(w.double()-low,high-w.double()).amin(-1)
        result=dict(training_distance=distance,known_mask=(~outside).reshape(shape),outside_mask=outside.reshape(shape),
            outside_distance_m=delta.reshape(shape),inside_insufficient_margin=((~outside)&(boundary<.15)).reshape(shape),
            world_boundary_points=q,boundary_trace=trace,proxy_cast_max_abs_m=cast_error,
            FP32_vs_recomputed_FP64_mask_disagreement=int((outside!=mask64).sum()),
            max_world_recompute_difference_m=float((wd2-w.double()).detach().abs().max()),
            points_B0=points,label="training_proxy outside; original strict SDF inside; not a safety certificate")
        return result

class RoiSafeTrainingObjective:
    """Composition over original Objective; formulas and Hard geometry remain original."""
    method_id=METHOD_ID
    component_id=COMPONENT_ID
    def __init__(self,spec=None,*,num_candidates=6):
        from .objective import BoundPhysicsV12Objective
        self.geometry=RoiSafeTrainingGeometry()
        self.objective=BoundPhysicsV12Objective(spec,training_geometry=self.geometry,num_candidates=num_candidates)
    def __call__(self,raw,batch):
        result=self.objective(raw,batch)
        result.training_geometry_records=self.geometry.records
        return result
