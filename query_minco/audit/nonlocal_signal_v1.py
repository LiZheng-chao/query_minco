"""Diagnostic-only symmetric physical smoothing of unchanged K1 L0; no selector or trainer."""
from dataclasses import dataclass
import math
import torch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
COMPONENT_ID="diagnostic_symmetric_physical_inner_y_L0_mean_v1"
BASE_OBJECTIVE_ID="physics_v12_phase_a_L0_K_normalized_v1"
def physical_lateral_shift(raw,offset_m):
    if not isinstance(raw,torch.Tensor) or raw.shape!=(1,1,21) or raw.dtype!=torch.float32 or raw.device.type!="cpu" or not torch.isfinite(raw).all():
        raise ValueError("NONLOCAL_RAW_CPU_FP32_K1")
    if not math.isfinite(float(offset_m)):raise ValueError("NONLOCAL_OFFSET_FINITE")
    physical_y=4*torch.tanh(raw[..., [1,4]])
    moved=physical_y+float(offset_m)
    if (moved.abs()>=4).any():raise ValueError("NONLOCAL_OFFSET_OUTSIDE_A_SUPPORT")
    shifted=raw.clone()
    shifted[..., [1,4]]=torch.atanh(moved/4)
    return shifted
def lateral_tangent(raw):
    tangent=torch.zeros_like(raw)
    tangent[..., [1,4]]=1/(4*(1-torch.tanh(raw[..., [1,4]]).square()))
    return tangent
@dataclass
class NonlocalResult:
    loss: torch.Tensor
    candidates: list
class SymmetricNonlocalCost:
    component_id=COMPONENT_ID
    training_replacement_authorized=False
    def __init__(self,scales_m=(.1,.3,.5,.7),objective=None):
        self.scales=tuple(float(x) for x in scales_m)
        if not self.scales or len(set(self.scales))!=len(self.scales) or any(not math.isfinite(x) or x<=0 for x in self.scales):
            raise ValueError("NONLOCAL_SCALES")
        self.objective=objective if objective is not None else make_objective()
        if self.objective.component_id!=BASE_OBJECTIVE_ID:raise ValueError("ORIGINAL_OBJECTIVE_ID")
    def __call__(self,raw,batch):
        rows=[]
        for delta in self.scales:
            for sign in [-1,1]:
                shifted=physical_lateral_shift(raw,sign*delta)
                result=self.objective(shifted,batch)
                if not torch.isfinite(result.loss) or not result.trajectory.solve_valid.all():
                    raise ValueError("NONLOCAL_PHYSICS_INVALID")
                rows.append(dict(delta_m=delta,sign=sign,raw=shifted,result=result))
        return NonlocalResult(torch.stack([x["result"].loss for x in rows]).mean(),rows)
