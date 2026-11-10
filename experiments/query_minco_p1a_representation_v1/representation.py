"""Offline P1-A prototypes only. No network, production decoder, or Loss replacement."""
from dataclasses import replace
import math
import numpy as np
import torch
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_result
from query_minco.training.pg1.losses import analytic_samples
from query_minco.evaluation.validation import numerical_reference,evaluate

ARMS=("A","B0","B1")
PILOT=("train_free_roi_1101_p0_near","train_free_roi_1101_p0_far","train_sparse_1102_p0_near","train_single_slit_1104_p0_near","train_two_portals_1103_p0_far")
D_VALUES=(2.,2.,2.,4.,4.,4.,6.,6.,6.,3.,3.,3.,8.,8.,8.)
SPEC=PhysicsV12Spec(phase="A",aggregation="foundation")

def reference(state,arm):
    """Only initial PVA, Goal, fixed time and frozen physical limits; never geometry."""
    s=state[0].double();v=s[:3];a=s[3:6];g=s[6:9];L=g.norm()
    if not bool(torch.isfinite(s).all()) or not L>0:raise ValueError("REFERENCE_STATE")
    if 3.2<float(L)<3.6:raise ValueError("UNREGISTERED_TRANSITION_TASK")
    near=float(L)<=3.2;n=g/L
    target=g if near else n*torch.minimum(L,L.new_tensor(2.4))
    vt=torch.zeros_like(g) if near else n*1.2
    at=torch.zeros_like(g)
    times=s.new_tensor([2/3,4/3,2.])
    if arm=="B0":
        points=times[:,None]/2*target
    elif arm=="B1":
        T=2.
        M=s.new_tensor([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
            [1,T,T*T,T**3,T**4,T**5],[0,1,2*T,3*T*T,4*T**3,5*T**4],
            [0,0,2,6*T,12*T*T,20*T**3]])
        rhs=torch.stack([torch.zeros_like(g),v,a,target,vt,at])
        coeff=torch.linalg.solve(M,rhs)
        powers=times[:,None]**torch.arange(6,dtype=s.dtype)[None]
        points=powers@coeff
    else:raise ValueError("REFERENCE_ARM")
    theta=torch.cat([points.reshape(-1),vt,at])
    D=s.new_tensor(D_VALUES)
    if bool((theta.abs()>=D).any()):raise ValueError("REFERENCE_OUTSIDE_SHARED_SUPPORT")
    return theta

def bias(state,arm):
    return torch.zeros(15,dtype=torch.float64) if arm=="A" else torch.atanh(reference(state,arm)/torch.tensor(D_VALUES,dtype=torch.float64))

def theta_math(u,state,arm):
    """FP64 mathematical oracle for AD/FD audit, not an alternative production MINCO."""
    if arm=="A":
        center=u.new_tensor([5,0,0]*3+[0]*6)
        scale=u.new_tensor([5,4,2]*3+[6,4,3,6,6,4])
        return center+scale*u.tanh()
    return u.new_tensor(D_VALUES)*(u+bias(state,arm).to(u.dtype)).tanh()

class PrototypeDecoder:
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError("ARM")
        self.arm=arm;self.original=FixedTimeDecoder(SPEC)
    def __call__(self,raw,state):
        base=self.original(raw,state)
        if self.arm=="A":return base
        theta=theta_math(raw[...,:15],state,self.arm)
        tail=theta[...,6:15].reshape(1,6,3,3)
        return replace(base,inner_pts=theta[...,:6].reshape(1,6,2,3),tail_pva=tail,
            diagnostics={**base.diagnostics,"terminal_speed_norm_mps":tail[...,1,:].norm(dim=-1),
                         "terminal_acceleration_norm_mps2":tail[...,2,:].norm(dim=-1)})

def raw_slots(u):
    if u.shape!=(15,) or u.dtype!=torch.float32 or u.device.type!="cpu":raise ValueError("DIRECT_VARIABLE_15D_CPU_FP32")
    return torch.cat([u,u.new_zeros(6)]).reshape(1,1,21).expand(1,6,21)

def objective(arm):
    obj=RoiSafeTrainingObjective(SPEC)
    # Explicit per-instance prototype dependency. No class/global monkeypatch or source edit.
    obj.objective.decoder=PrototypeDecoder(arm)
    return obj

def physical_theta(parameters):
    return torch.cat([parameters.inner_pts.flatten(-2),parameters.tail_pva.flatten(-2)],dim=-1)

def hard_one(result,batch):
    # All six duplicated slots are a single candidate; count only original validator row0.
    rows=validate_result(result,batch)
    if any(r!=rows[0] for r in rows[1:]):raise ValueError("DUPLICATE_SLOT_HARD_MISMATCH")
    return rows[0]

def samples_vector(u,state,arm):
    from query_minco.trajectory.minco_adapter import MincoAdapter
    p=PrototypeDecoder(arm)(raw_slots(u),state)
    tr=MincoAdapter()(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
    return analytic_samples(tr,64)["p"][0,0].reshape(-1)

def independent_samples(u,state,arm):
    theta=theta_math(torch.as_tensor(u,dtype=torch.float64),state,arm).detach().numpy()
    st=state[0].double().numpy()
    c=numerical_reference(np.stack([np.zeros(3),st[:3],st[3:6]]),theta[6:].reshape(3,3),theta[:6].reshape(2,3),np.full(3,2/3))
    return np.concatenate([evaluate(piece,np.linspace(0,2/3,64)) for piece in c],axis=0).reshape(-1)

def summarize(result,batch,u,gradient=None,arm="A"):
    ana=analytic_samples(result.trajectory,64);g=result.training_geometry_records[0]
    h=hard_one(result,batch);p=ana["p"][0,0].detach().double().numpy()
    v=ana["v"][0,0].detach().double().numpy()
    T=result.trajectory.durations[0,0].detach().double().numpy()
    speeds=np.linalg.norm(v,axis=-1).reshape(3,64)
    length=sum(np.trapz(a,np.linspace(0,t,64)) for a,t in zip(speeds,T))
    metrics={k:float(x[0,0]) for k,x in result.metrics.items() if x.ndim==2 and x.shape==(1,6)}
    metrics.pop("min_sdf_m",None)
    out=dict(channel_coverage=channel_coverage(result,batch),u=u.detach().tolist(),theta=physical_theta(result.parameters)[0,0].detach().tolist(),loss=float(result.loss),
        metrics=metrics,known_count=int(g["known_mask"][0].sum()),UNKNOWN_count=int(g["outside_mask"][0].sum()),
        roi_margin_insufficient=int(g["inside_insufficient_margin"][0].sum()),
        training_min_distance_m=float(g["training_distance"][0].min()),training_distance_is_not_actual_unknown_SDF=True,
        hard=h,path_length_m=float(length),horizon_s=2.,endpoint=ana["endpoint"][0,0].detach().tolist(),
        terminal_velocity=ana["terminal_velocity"][0,0].detach().tolist(),terminal_acceleration=result.parameters.tail_pva[0,0,2].detach().tolist(),
        coefficients=result.trajectory.coeffs[0,0].detach().tolist(),sampled_p_B0=p.tolist(),
        saturated_fraction=float(((u+bias(batch.state,arm).float()).tanh().abs()>.999).float().mean()))
    if int(g["outside_mask"][0].sum())==0:
        out["known_sample_clearance_m"]=float(batch.geometry[0].sdf(ana["p"][0,0]).min())-.15
    else:out["known_sample_clearance_m"]=None
    bounds=[c.get("surface_clearance_lower_bound_m") for c in h["certificates"]]
    out["certified_surface_clearance_lower_bound_m"]=min(bounds) if all(x is not None for x in bounds) else None
    if gradient is not None:out["gradient"]=gradient.detach().tolist();out["gradient_norm_FP64"]=float(gradient.detach().double().norm())
    return out
def channel_coverage(result,batch):
    """Evaluation only: actual native geometry and continuous gate crossings, not init input."""
    import json
    from pathlib import Path
    from query_minco.training.physics_only_v1_2_gate1.hard_validator import transform_coefficients
    sid=batch.manifest["scene_id"]
    if not any(k in sid for k in ("two_portals","single_slit")):return dict(applicable=False)
    specs=[Path(p) for p in batch.manifest["files_sha256"] if p.endswith(sid+"_SCENE_SPEC.json")]
    if not specs:
        # Continuation manifests bind scene.json -> source_spec_sha256, not a direct spec path.
        candidate=Path(batch.manifest["scene_dir"]).parents[2]/"runs/qualification_001"/(sid+"_SCENE_SPEC.json")
        scene=json.loads((Path(batch.manifest["scene_dir"])/"scene.json").read_text())
        if __import__("hashlib").sha256(candidate.read_bytes()).hexdigest()!=scene["source_spec_sha256"]:
            raise ValueError("CHANNEL_SPEC_SHA")
        specs=[candidate]
    if len(specs)!=1:raise ValueError("CHANNEL_SPEC_IDENTITY")
    layout=json.loads(specs[0].read_text())["layout"];gate=layout["gate_x_m"];g=batch.geometry[0]
    c=transform_coefficients(result.trajectory.coeffs[0,0].detach().numpy(),g.origin,g.rotation)
    durations=result.trajectory.durations[0,0].detach().numpy();cross=[];offset=0.
    for piece,T in zip(c,durations):
        poly=piece[0].copy();poly[0]-=gate
        for z in np.polynomial.polynomial.polyroots(poly):
            if abs(z.imag)<1e-8 and -1e-9<=z.real<=float(T)+1e-9:
                t=float(np.clip(z.real,0,T));p=evaluate(piece,t);kind="UNCLASSIFIED"
                if "two_portals" in sid:
                    boxes=g.boxes[(g.boxes[:,0,0]<=gate)&(g.boxes[:,1,0]>=gate)&
                        (g.boxes[:,0,2]-.15<=p[2])&(g.boxes[:,1,2]+.15>=p[2])&
                        (np.abs(g.boxes.mean(1)[:,1])<1)]
                    if len(boxes):
                        low=float(boxes[:,0,1].min());high=float(boxes[:,1,1].max())
                        kind="negative_y" if p[1]<low-.15 else "positive_y" if p[1]>high+.15 else "central_or_insufficient_margin"
                else:
                    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import slit_gap
                    gap=slit_gap(g.meta,g.counts,layout)
                    bounds=gap.get("y_bounds_world_m")
                    if bounds:kind="slit_center_corridor" if bounds[0]+.15<=p[1]<=bounds[1]-.15 else "outside_slit_corridor"
                cross.append(dict(time_s=offset+t,world_m=p.tolist(),channel=kind))
        offset+=float(T)
    return dict(applicable=True,gate_x_world_m=gate,crossings=cross,
        crossed=bool(cross),note="Only original Hard-passed curves with actual crossing count as covered channels; pre-gate stopping is not coverage.")