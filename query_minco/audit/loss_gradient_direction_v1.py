"""Zero-update diagnostics of the unchanged Phase-A/P0c objective."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingGeometry
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.evaluation.mirror_holdout_v1 import load_scene,write,sha
NAMES=("C_viol","F","G_task_train","C_shape","P","S")
WEIGHTS=(1.,1.,1.,.1,.025,.1)
COMPONENT_ID="phase_a_p0c_zero_update_gradient_direction_audit_v1"
def lateral_raw(raw,delta):
    r=raw.detach().clone()
    y=4*torch.tanh(r[..., [1,4]])
    if ((y+delta).abs()>=4).any():raise ValueError("LATERAL_RANGE")
    r[..., [1,4]]=torch.atanh((y+delta)/4)
    return r
def tangent(raw):
    v=torch.zeros_like(raw)
    v[..., [1,4]]=1/(4*(1-torch.tanh(raw[..., [1,4]]).square()))
    return v
def bound_batch(depth,state,g,manifest):
    b=BoundBatch(depth,state,(g,),dict(diagnostic_only=True,source_manifest=manifest),g.validate)
    b._depth=depth.numpy().tobytes();b._state=state.numpy().tobytes();b.validate()
    return b
def cost(objective,raw,batch,with_hard=False):
    out=objective(raw,batch)
    if not out.trajectory.solve_valid.all() or not torch.isfinite(out.loss):raise ValueError("INVALID_OBJECTIVE")
    terms={n:float(out.metrics[n].mean().detach()) for n in NAMES}
    reconstructed=sum(w*terms[n] for n,w in zip(NAMES,WEIGHTS))
    if abs(float(out.loss.detach())-reconstructed)>2e-5:raise ValueError("L0_FORMULA")
    value=dict(loss=float(out.loss.detach()),terms=terms,outside_proxy_points=int(out.training_geometry_records[0]["outside_mask"].sum()))
    if with_hard:value["hard"]=validate_result_candidates(out,batch)[0]
    return out,value
def run_gradient_audit(g3root,preregistration,output):
    root=Path(g3root);plan=json.loads(Path(preregistration).read_text());outdir=Path(output);outdir.mkdir()
    old=json.loads((root/"FILES_AND_SHA.json").read_text());source=root/"FROZEN_MIRROR_ARRAYS.npz"
    if sha(source)!=old["artifacts"]["FROZEN_MIRROR_ARRAYS.npz"]["sha256"]:raise ValueError("FROZEN_RAW_DRIFT")
    z=np.load(source)
    starts={"fixed_forward":z["offset_passage_A_left_fixed_forward_raw21"],
            "common_step128_negative_bias":z["offset_passage_A_left_A_reasonable_step128_raw21"]}
    scenes=json.loads((root/"scenes/QUALIFICATION_RESULT.json").read_text())["scenes"]
    rows=[];scans=[];points=[];firststate=None
    for scene in scenes:
        depth,state,g,m,cert=load_scene(scene["manifest"],scene["manifest_sha256"])
        if firststate is None:firststate=state.clone()
        if not torch.equal(firststate,state):raise ValueError("STATE_NOT_IDENTICAL")
        batch=bound_batch(depth,state,g,m);objective=make_objective()
        for start,base_np in starts.items():
            base=torch.from_numpy(base_np.copy()).float();raw=base.clone().requires_grad_(True)
            result,values=cost(objective,raw,batch)
            all_grads={}
            for name in NAMES:
                all_grads[name]=torch.autograd.grad(result.metrics[name].mean(),raw,retain_graph=True)[0]
            total=torch.autograd.grad(result.loss,raw,retain_graph=True)[0]
            reconstructed=sum(w*all_grads[n] for n,w in zip(NAMES,WEIGHTS))
            if not torch.allclose(total,reconstructed,atol=3e-5,rtol=3e-5):raise ValueError("GRADIENT_SUM")
            if not torch.isfinite(total).all() or total[...,15:].count_nonzero():raise ValueError("RAW_GRADIENT")
            v=tangent(base);collision=all_grads["C_viol"]+.1*all_grads["C_shape"];other=total-collision
            cn=float(collision[...,:15].norm());on=float(other[...,:15].norm())
            cosine=float((collision[...,:15]*other[...,:15]).sum())/(cn*on) if cn*on>1e-20 else None
            direction=float((total*v).sum());collision_direction=float((collision*v).sum())
            fd=[]
            for eps in plan["gradient"]["finite_difference_steps_m"]:
                _,plus=cost(objective,lateral_raw(base,eps),batch)
                _,minus=cost(objective,lateral_raw(base,-eps),batch)
                finite=(plus["loss"]-minus["loss"])/(2*eps)
                fd.append(dict(epsilon_m=eps,loss_minus=minus["loss"],loss_plus=plus["loss"],
                    finite_difference_total=finite,analytic_total=direction,absolute_discrepancy=abs(finite-direction),
                    term_finite_differences={n:(plus["terms"][n]-minus["terms"][n])/(2*eps) for n in NAMES}))
            grads={n:dict(raw0_15=all_grads[n][0,0,:15].tolist(),norm=float(all_grads[n][...,:15].norm()),
                        weighted_norm=float(all_grads[n][...,:15].norm())*w,lateral_directional_derivative=float((all_grads[n]*v).sum()),
                        weighted_lateral_derivative=float((all_grads[n]*v).sum())*w) for n,w in zip(NAMES,WEIGHTS)}
            rows.append(dict(scene_id=scene["scene_id"],pair_id=m["pair"]["pair_id"],side=scene["side"],start=start,
                common_raw_sha256=hashlib.sha256(base.numpy().tobytes()).hexdigest(),state_sha256=hashlib.sha256(state.numpy().tobytes()).hexdigest(),
                values=values,gradients=grads,total_raw0_15=total[0,0,:15].tolist(),total_norm=float(total[...,:15].norm()),
                total_lateral_derivative=direction,collision_lateral_derivative=collision_direction,
                descent_prefers_registered_side=bool(-direction*scene["side"]>1e-7),
                collision_descent_prefers_registered_side=bool(-collision_direction*scene["side"]>1e-7),
                collision_other_gradient_cosine=cosine,finite_difference=fd,raw15_21_nonzero=0,raw_leaf_updates=0))
            for delta in plan["gradient"]["delta_inner_y_m"]:
                shifted=lateral_raw(base,delta)
                _,value=cost(objective,shifted,batch,with_hard=True)
                scans.append(dict(scene_id=scene["scene_id"],side=scene["side"],start=start,physical_inner_y_shift_m=delta,**value))
            if not np.array_equal(base.numpy(),base_np):raise ValueError("RAW_LEAF_UPDATED")
        # Independent local-cost probes. At voxel-face switches FD need not equal one subgradient.
        proxy=RoiSafeTrainingGeometry();center_x=cert["planes_B0"][1]
        for y in [-.8,-.4,0.,.4,.8]:
            point=torch.tensor([[center_x,y,0.]],dtype=torch.float32,requires_grad=True)
            record=proxy.query(point,g);distance=record["training_distance"]
            local=((.15-distance).clamp_min(0)/.15).square()
            gradient=torch.autograd.grad(local.sum(),point)[0]
            if not torch.isfinite(gradient).all():raise ValueError("NONFINITE_POINT_GRADIENT")
            eps=.001
            with torch.no_grad():
                shift=torch.tensor([[0.,eps,0.]])
                pp=proxy.query(point+shift,g)["training_distance"];mm=proxy.query(point-shift,g)["training_distance"]
                fp=((.15-pp).clamp_min(0)/.15).square();fm=((.15-mm).clamp_min(0)/.15).square()
            points.append(dict(scene_id=scene["scene_id"],point_B0=point.detach().tolist(),sdf_m=float(distance[0]),
                local_collision_cost=float(local[0]),gradient_xyz=gradient[0].tolist(),finite_difference_y=float((fp-fm)[0]/(2*eps)),
                outside_proxy=bool(record["outside_mask"].any()),note="Local original collision formula; not a witness or optimization target."))
    write(outdir/"LOSS_GRADIENT_RESULTS.json",dict(component_id=COMPONENT_ID,records=rows,weights=dict(zip(NAMES,WEIGHTS)),
        same_state_and_common_raw=True,model_instantiated=False,optimizer_constructed=False,raw_leaf_updates=0,new_network_updates=0))
    write(outdir/"LATERAL_COST_HARD_SCAN.json",dict(records=scans,selected_for_online_route=False,network_updates=0))
    write(outdir/"LOCAL_P0C_POINT_PROBES.json",dict(records=points,network_updates=0))
    return dict(base_cases=len(rows),cost_hard_scans=len(scans),local_probes=len(points),new_network_updates=0)
