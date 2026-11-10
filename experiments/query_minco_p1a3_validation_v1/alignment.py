"""Read-only offline objective evaluation. Witnesses never reach an optimizer."""
from dataclasses import replace
from pathlib import Path
import json,itertools
import numpy as np
import torch
from experiments.query_minco_p1a2_falsification_v1 import falsification as f
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2_gate1.hard_validator import transform_coefficients
BASE=Path(__file__).resolve().parents[2]
OLD=BASE/"research_outputs/query_minco_p1a2_falsification_v1"
TERMS=("G_task_train","P","C_viol","C_shape","F","S")
# Contributions to actual frozen Phase A scene objective, not the diagnostic per-query J.
FACTORS=dict(G_task_train=1.,P=.025,C_viol=1.,C_shape=.1,F=1.,S=.1)

class EvaluationOnlyThetaDecoder:
    def __init__(self,theta):self.theta=theta
    def __call__(self,raw,state):
        base=f.p1.PrototypeDecoder("B0")(raw,state)
        t=self.theta.reshape(1,1,15).expand(1,6,15)
        return replace(base,inner_pts=t[...,:6].reshape(1,6,2,3),tail_pva=t[...,6:].reshape(1,6,3,3))
def evaluate_theta(theta,batch):
    if theta.shape!=(15,) or theta.dtype!=torch.float32 or not theta.requires_grad:raise ValueError("EVALUATION_LEAF")
    obj=RoiSafeTrainingObjective(f.p1.SPEC);obj.objective.decoder=EvaluationOnlyThetaDecoder(theta)
    return obj(f.p1.raw_slots(torch.zeros(15)),batch)
def gradient_or_zero(value,x):
    if not value.requires_grad:return torch.zeros_like(x)
    g=torch.autograd.grad(value,x,retain_graph=True,allow_unused=True)[0]
    return torch.zeros_like(x) if g is None else g
def cosine(a,b):
    den=float(a.double().norm()*b.double().norm())
    return None if den==0 else float(torch.dot(a.double(),b.double())/den)

def diagnose(batch,contract,theta,*,kind,identity,u=None,which=None):
    theta=theta.detach().clone().float().requires_grad_()
    r=evaluate_theta(theta,batch)
    contributions={k:FACTORS[k]*r.metrics[k].mean() for k in TERMS}
    reconstructed=sum(contributions.values())
    if not torch.allclose(reconstructed,r.loss,rtol=1e-6,atol=1e-7):raise ValueError("PHASE_A_CONTRIBUTION_MISMATCH")
    total=gradient_or_zero(r.loss,theta)
    gs={k:gradient_or_zero(x,theta) for k,x in contributions.items()}
    if not torch.allclose(sum(gs.values()),total,rtol=2e-4,atol=2e-5):raise ValueError("GRADIENT_DECOMPOSITION")
    if not torch.isfinite(total).all():raise ValueError("NONFINITE")
    raw_info=None
    if u is not None:
        u=u.detach().clone().requires_grad_()
        prod=f.p1.objective(which)(f.p1.raw_slots(u),batch)
        err=float((f.p1.physical_theta(prod.parameters)[0,0]-theta).abs().max())
        gp=gradient_or_zero(prod.loss,u)
        J=torch.autograd.functional.jacobian(lambda z:f.actual_theta(z,batch.state,which),u)
        predicted=J.T@total
        if err>3e-6 or not torch.allclose(prod.loss,r.loss,rtol=.003,atol=.003) or not torch.allclose(predicted,gp,rtol=.003,atol=.003):
            raise ValueError("RAW_CHAIN_RULE_MISMATCH")
        singular=torch.linalg.svdvals(J.double())
        raw_info=dict(arm=which,u=u.detach().tolist(),gradient=gp.tolist(),gradient_norm_FP64=float(gp.double().norm()),theta_Jacobian_singular_values=singular.tolist(),chain_rule_max_abs=float((predicted-gp).abs().max()),gradient_end_x_under_raw_descent=float(-(J@gp)[6]))
    h=f.p1.hard_one(r,batch);task=f.crossing_success(r,batch,contract,h)
    ana=f.p1.summarize(r,batch,torch.zeros(15),total,arm="B0")
    # Common physical-coordinate gradient, with units stated; these are local diagnostics only.
    progress_grad=gradient_or_zero(r.metrics["P"].mean(),theta)
    end_world=task["endpoint_world"]
    per={k:dict(raw_term=float(r.metrics[k].mean()),phase_A_factor=FACTORS[k],contribution=float(v),theta_gradient=gs[k].tolist(),theta_gradient_norm_FP64=float(gs[k].double().norm()),theta_D_scaled_gradient_norm_FP64=float((gs[k].double()*torch.tensor(f.p1.D_VALUES,dtype=torch.float64)).norm()),end_x_under_component_descent=float(-gs[k][6])) for k,v in contributions.items()}
    pairs={a+"__"+b:cosine(gs[a],gs[b]) for a,b in itertools.combinations(TERMS,2)}
    record=dict(identity=identity,kind=kind,witness_read_only_evaluation=kind=="frozen_qualification_evaluation",used_as_optimizer_initialization=False,optimizer_updates=0,source_sample=batch.manifest["sample_id"] if "sample_id" in batch.manifest else contract["source_scene_id"],theta=theta.detach().tolist(),actual_Phase_A_loss=float(r.loss),per_query_J=float(r.metrics["J"][0,0]),V=float(r.metrics["V"][0,0]),Q=float(r.metrics["Q"][0,0]),components=per,gradient_theta=total.tolist(),gradient_norm_theta_FP64=float(total.double().norm()),gradient_cosines=pairs,raw_coordinate_check=raw_info,local_progress_derivative_end_x=float(progress_grad[6]),end_x_under_total_physical_gradient_descent=float(-total[6]),OriginalHardAll=h["passed"],TaskSuccess=task["TaskSuccess"],task=task,hard=h,endpoint_B0=ana["endpoint"],endpoint_world=end_world,min_known_sample_clearance_m=ana["known_sample_clearance_m"],certified_surface_clearance_lower_bound_m=ana["certified_surface_clearance_lower_bound_m"],UNKNOWN_count=ana["UNKNOWN_count"],coefficients=ana["coefficients"],source_geometry_identity=batch.geometry[0].identity)
    return record,r

def frozen_sources():
    """No optimizer imported/called here; data are already frozen P1-A2 evaluation records."""
    result=[]
    for case,sample in f.CASE_SOURCES.items():
        if not case.startswith("native"):continue
        batch,source=f.load_source(sample);contract=f.native_contract(batch,case)
        controls,aliases=f.canonical_controls(batch.state)
        for control in controls:
            u=f.initial(control,batch.state);which=f.arm(control)
            result.append((case,batch,contract,f.actual_theta(u,batch.state,which),dict(kind="analytic_nominal_reference",identity=case+"__"+control+"__step0",u=u,which=which,source=source)))
        for control in controls:
            path=OLD/"runs/acceptance_smoke_001"/(case+"__None__"+control)/"STEP_4.json"
            old=json.loads(path.read_text());u=torch.tensor(old["u"],dtype=torch.float32);which=f.arm(control)
            result.append((case,batch,contract,f.actual_theta(u,batch.state,which),dict(kind="frozen_smoke_final",identity=str(path),u=u,which=which,old_record=old,source=source)))
        path=OLD/"runs/acceptance_gate_b_001"/(case+".json")
        old=json.loads(path.read_text())
        for i,row in enumerate(old["candidates"]):
            result.append((case,batch,contract,torch.tensor(row["theta"],dtype=torch.float32),dict(kind="frozen_qualification_evaluation",identity=str(path)+"#"+str(i),old_record=row,source=source)))
    return result
