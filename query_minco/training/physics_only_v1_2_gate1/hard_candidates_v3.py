"""Versioned candidate-axis helper; delegates all physics to frozen Hard v2."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from . import hard_validator as legacy
from query_minco.evaluation.validation import numerical_reference,evaluate
COMPONENT_ID="physics_v12_hard_candidate_axis_k1_k6_helper_v3"
LEGACY_COMPONENT_ID=legacy.COMPONENT_ID
LEGACY_SHA256="e49d7eee5bc45b1daafb6c575810dec4e49c13137250d0c3d803e96fafd974fa"

def validate_result_candidates(result,batch):
    if hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()!=LEGACY_SHA256:
        raise ValueError("LEGACY_HARD_SOURCE_SHA_DRIFT")
    tr=result.trajectory;p=result.parameters
    c=tr.coeffs;d=tr.durations
    if not isinstance(c,torch.Tensor) or c.ndim!=5 or c.shape[0]!=1 or c.shape[1] not in (1,6) or tuple(c.shape[2:])!=(3,3,6):
        raise ValueError("HARD_CANDIDATE_COEFFICIENT_SHAPE")
    k=c.shape[1]
    values=[(c,(1,k,3,3,6)),(d,(1,k,3)),(p.head_pva,(1,k,3,3)),(p.tail_pva,(1,k,3,3)),(p.inner_pts,(1,k,2,3)),(batch.state,(1,12))]
    for tensor,shape in values:
        if not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=shape or tensor.device.type!="cpu" or tensor.dtype not in (torch.float32,torch.float64):
            raise ValueError("HARD_CANDIDATE_INPUT_SHAPE_TYPE")
    if len(batch.geometry)!=1:raise ValueError("HARD_GEOMETRY_COUNT")
    if not all(torch.isfinite(x).all() for x in [p.head_pva,p.tail_pva,p.inner_pts,batch.state]):
        raise ValueError("HARD_NONFINITE_PVA_OR_STATE")
    rows=[]
    for i in range(k):
        coeff=c[0,i].detach().numpy();dt=d[0,i].detach().numpy()
        row=legacy.validate_polynomial(coeff,dt,batch.geometry[0],batch.state[0,6:9].detach().numpy())
        if not np.isfinite(coeff).all() or not np.isfinite(dt).all() or (dt<=0).any():
            rows.append(row);continue
        ref=numerical_reference(p.head_pva[0,i].detach().numpy(),p.tail_pva[0,i].detach().numpy(),p.inner_pts[0,i].detach().numpy(),dt)
        errors=[max(float(np.abs(evaluate(a,np.linspace(0,t,65),order)-evaluate(b,np.linspace(0,t,65),order)).max()) for a,b,t in zip(coeff,ref,dt)) for order in (0,1,2)]
        row["independent_minco_max_abs_pva"]=errors
        if any(e>tol for e,tol in zip(errors,(.002,.01,.05))):
            row["passed"]=False;row["reasons"].append("INDEPENDENT_MINCO")
        rows.append(row)
    return rows
