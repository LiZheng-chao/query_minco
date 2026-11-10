"""P1-A2 independent fairness and crossing evaluation; production and P1A are read-only."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json,hashlib
import numpy as np
import torch
from experiments.query_minco_p1a_representation_v1 import representation as p1
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1_2_gate1.hard_validator import transform_coefficients
from query_minco.evaluation.validation import evaluate
BASE=Path(__file__).resolve().parents[2]
INDEX=BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json"
CONTROLS=("A_nominal","A_matched_B0_ref","A_matched_B1_ref","B0_nominal","B1_nominal")
CASE_SOURCES={"free_near":"train_free_roi_1101_p0_near","sparse_near":"train_sparse_1102_p0_near",
 "native_slit_cross_far_v1":"train_single_slit_1104_p0_far",
 "native_two_portals_cross_far_v1":"train_two_portals_1103_p0_far"}
MARGIN=.150001

def index():return json.loads(INDEX.read_text())
def load_source(sample_id):
    row=next(r for r in index() if r["sample_id"]==sample_id)
    if not row["data_eligible"] or row["reachability"]!="FEASIBLE_CERTIFIED":raise ValueError("SOURCE_NOT_QUALIFIED")
    return load_e1_bound(row["p0c_manifest_path"],row["p0c_manifest_sha256"]),row
def arm(control):return "A" if control.startswith("A_") else "B0" if control.startswith("B0_") else "B1"
def actual_theta(u,state,which):
    return p1.physical_theta(p1.PrototypeDecoder(which)(p1.raw_slots(u),state))[0,0]
def inverse_A(theta):
    c=theta.new_tensor([5,0,0]*3+[0]*6);d=theta.new_tensor([5,4,2]*3+[6,4,3,6,6,4])
    ratio=(theta-c)/d
    if not bool(torch.isfinite(theta).all()) or bool((ratio.abs()>=1).any()):raise ValueError("A_OPEN_SUPPORT")
    return torch.atanh(ratio).float()
def inverse_B(theta,state,which):
    ratio=theta/theta.new_tensor(p1.D_VALUES)
    if bool((ratio.abs()>=1).any()):raise ValueError("B_OPEN_SUPPORT")
    return (torch.atanh(ratio)-p1.bias(state,which)).float()
def initial(control,state,seed=None):
    if control=="A_nominal":
        return torch.zeros(15) if seed is None else torch.randn(15,generator=torch.Generator().manual_seed(seed))*.05
    ref_arm="B0" if "B0" in control else "B1"
    ref=actual_theta(torch.zeros(15),state,ref_arm).double()
    if seed is not None:ref=ref+torch.tensor(p1.D_VALUES,dtype=torch.float64)*.01*torch.randn(15,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    if control.startswith("A_"):return inverse_A(ref)
    return torch.zeros(15) if seed is None else inverse_B(ref,state,ref_arm)

def canonical_controls(state):
    b0=actual_theta(torch.zeros(15),state,"B0");b1=actual_theta(torch.zeros(15),state,"B1")
    if float((b0-b1).abs().max())<=3e-6:
        return ("A_nominal","A_matched_B0_ref","B0_nominal"),{"A_matched_B1_ref":"A_matched_B0_ref","B1_nominal":"B0_nominal"}
    return CONTROLS,{}
def scene_spec(batch):
    sid=batch.manifest["scene_id"];p=Path(batch.manifest["scene_dir"]).parents[2]/"runs/qualification_001"/(sid+"_SCENE_SPEC.json")
    scene=json.loads((Path(batch.manifest["scene_dir"])/"scene.json").read_text())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=scene["source_spec_sha256"]:raise ValueError("SCENE_SPEC_SHA")
    return json.loads(p.read_text()),str(p),scene["source_spec_sha256"]

def native_contract(batch,case_id):
    g=batch.geometry[0];g.validate()
    if not np.array_equal(g.rotation,np.eye(3)):raise ValueError("UNREGISTERED_ROTATED_GATE")
    s,path,h=scene_spec(batch);layout=s["layout"];gate=float(layout["gate_x_m"])
    box=np.asarray(layout.get("slab_box_m",layout.get("block_box_m")),float)
    centers=g.boxes.mean(1)
    # Layout identifies the wall; all safety extents come from ACTUAL certified native boxes.
    use=(centers[:,0]>=box[0,0]-.11)&(centers[:,0]<=box[1,0]+.11)&(centers[:,1]>=box[0,1]-.11)&(centers[:,1]<=box[1,1]+.11)
    wall=g.boxes[use]
    if not len(wall):raise ValueError("NATIVE_WALL_MISSING")
    lo=wall[:,0].min(0);hi=wall[:,1].max(0)
    post=float(hi[0]+MARGIN+.10)
    if post>=g.roi[1,0]-MARGIN:raise ValueError("POST_REGION_OUTSIDE_KNOWN_ROI")
    source_before=bool(g.origin[0]<=lo[0]-MARGIN)
    if not source_before:raise ValueError("INITIAL_NOT_BEFORE_WALL")
    return dict(case_id=case_id,source_scene_id=batch.manifest["scene_id"],kind="slit" if "single_slit" in s["scene_id"] else "two_portals",
        gate_x_world_m=gate,native_wall_bounds_world=[lo.tolist(),hi.tolist()],post_checkpoint_x_world_m=post,
        initial_world=g.origin.tolist(),initial_before_wall=source_before,reference_ball_margin_m=MARGIN,
        layout=layout,source_spec_path=path,source_spec_sha256=h,goal_unchanged=True,Loss_unchanged=True,
        geometric_progress_to_checkpoint_m=post-g.origin[0],max_speed_time_bound_m=3*2,
        source_qualification="old source geometry only; new crossing TaskSuccess requires its own witness")

def value_at(c,d,t):
    offset=0.
    for i,T in enumerate(d):
        if t<=offset+T or i==len(d)-1:return evaluate(c[i],np.clip(t-offset,0,T))
        offset+=T
    raise ValueError("TIME")
def point_distance(g,p):
    if ((p<g.roi[0])|(p>g.roi[1])).any():return None
    q=np.abs(p-g.centers)-g.halves
    a=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
    return float(min(a.min(),p[2]-g.floor,np.minimum(p-g.roi[0],g.roi[1]-p).min()))
def opening_interval(g,contract,zlow,zhigh):
    gate=contract["gate_x_world_m"];b=g.boxes
    use=(b[:,0,0]<=gate)&(b[:,1,0]>=gate)&(b[:,0,2]<=zhigh+MARGIN)&(b[:,1,2]>=zlow-MARGIN)
    selected=b[use]
    if not len(selected):return None
    if contract["kind"]=="slit":
        neg=selected[selected[:,1,1]<0];pos=selected[selected[:,0,1]>0]
        if not len(neg) or not len(pos):return None
        low=max(float(neg[:,1,1].max()),float(contract["layout"]["remove_y_interval_m"][0]))
        high=min(float(pos[:,0,1].min()),float(contract["layout"]["remove_y_interval_m"][1]))
        return dict(low=low+MARGIN,high=high-MARGIN,kind="slit")
    block=selected[np.abs(selected.mean(1)[:,1])<1.]
    if not len(block):return None
    return dict(low=float(block[:,0,1].min())-MARGIN,high=float(block[:,1,1].max())+MARGIN,kind="sides")

def crossing_success(result,batch,contract,hard=None):
    h=p1.hard_one(result,batch) if hard is None else hard
    g=batch.geometry[0];c=transform_coefficients(result.trajectory.coeffs[0,0].detach().double().numpy(),g.origin,g.rotation)
    d=result.trajectory.durations[0,0].detach().double().numpy();T=float(d.sum());gate=contract["gate_x_world_m"]
    roots=[];offset=0.
    for piece,dt in zip(c,d):
        poly=piece[0].copy();poly[0]-=gate
        if np.abs(poly[1:]).max()==0:offset+=dt;continue
        for r in np.polynomial.polynomial.polyroots(poly):
            if abs(r.imag)<=1e-9 and 0<=r.real<=dt:
                roots.append(float(offset+r.real))
        offset+=dt
    roots=sorted(set(round(t,12) for t in roots));events=[]
    for t in roots:
        left=max(0.,t-1e-6);right=min(T,t+1e-6)
        a=value_at(c,d,left);b=value_at(c,d,right);p=value_at(c,d,t)
        residual=abs(float(p[0]-gate));forward=bool(a[0]<gate-1e-8 and b[0]>gate+1e-8)
        # Original speed bound ensures displacement <=3*dt IF Hard passed; use conservative bracket.
        ys=[a[1],b[1],p[1]];zs=[a[2],b[2],p[2]]
        interval=opening_interval(g,contract,min(zs)-3e-6,max(zs)+3e-6)
        channel=None
        if interval:
            if interval["kind"]=="slit" and min(ys)-3e-6>=interval["low"] and max(ys)+3e-6<=interval["high"]:channel="slit"
            elif interval["kind"]=="sides":
                if max(ys)+3e-6<=interval["low"]:channel="negative_y"
                elif min(ys)-3e-6>=interval["high"]:channel="positive_y"
        distance=point_distance(g,p)
        certified=bool(h["passed"] and forward and residual<=1e-7 and channel is not None and distance is not None and distance>=MARGIN)
        events.append(dict(time_s=t,bracket_s=[left,right],world=p.tolist(),root_residual_m=residual,forward_sign_bracket=forward,
           native_opening=interval,channel=channel,all_box_known_surface_distance_m=distance,certified=certified))
    endpoint=value_at(c,d,T);post=bool(endpoint[0]>=contract["post_checkpoint_x_world_m"])
    certified=any(e["certified"] for e in events)
    return dict(OriginalHardAll=bool(h["passed"]),certified_gate_crossing=certified,endpoint_post_wall=post,
        TaskSuccess=bool(h["passed"] and certified and post),events=events,endpoint_world=endpoint.tolist(),
        successful_channels=sorted({e["channel"] for e in events if e["certified"]}),
        UNKNOWN_accepted_as_free=False,numeric_root_certification="sign-bracket + original global Hard; not interval arithmetic")

def result_for_theta(theta,state):
    from query_minco.trajectory.minco_adapter import MincoAdapter
    base=p1.PrototypeDecoder("B0")(p1.raw_slots(torch.zeros(15)),state)
    t=torch.as_tensor(theta,dtype=torch.float32).reshape(1,1,15).expand(1,6,15)
    params=replace(base,inner_pts=t[...,:6].reshape(1,6,2,3),tail_pva=t[...,6:].reshape(1,6,3,3))
    tr=MincoAdapter()(params.head_pva,params.tail_pva,params.inner_pts,params.durations)
    return SimpleNamespace(parameters=params,trajectory=tr)
