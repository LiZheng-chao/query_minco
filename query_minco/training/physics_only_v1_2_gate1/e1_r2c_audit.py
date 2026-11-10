"""Read-only R2b geometry/ledger audit and new CPU-only reachability protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .e1_reachability_v2 import protocol,search_task,evaluate_witness,signed_box_distance,digest
from .e1_dataset_qualification import scene_data,geometry_identity,check_split,choose_poses
from .geometry import TypedBoundOccupancy,_SEAL
from .e1_dataset_binding import load_e1_bound
from .hard_validator import validate_polynomial
from query_minco.evaluation.validation import evaluate
BASE=Path(__file__).resolve().parents[3]
OLD=BASE/"research_outputs/physics_only_v1_2_e1_dataset_v2"
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r2c_audit_v1"
PREP=OUT/"_preparation/review_001"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def protect():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("HISTORICAL_SOURCE_DRIFT "+p)
    for p,h in read(PREP/"SOURCE_FREEZE.json").items():
        if sha(BASE/p)!=h:raise ValueError("NEW_SOURCE_DRIFT "+p)
    if time.time()-e["started_unix"]>=1800:raise ValueError("TOTAL_WALL_BUDGET")
    return e
def ledger_audit():
    ledger=OLD/"_preparation/qualification_001/GPU_CALLS.jsonl"
    rows=[json.loads(line) for line in ledger.read_text().splitlines()]
    if [r["index"] for r in rows]!=list(range(1,11)):raise ValueError("LEDGER_SEQUENCE")
    poses=[];receipts=[]
    for row in rows:
        if row["kind"]=="counts":
            p=OLD/"runs/qualification_001"/(row["name"]+"_EXIT.json")
        elif row["kind"]=="depth":
            folder=Path(row["argv"][-1]).parent;p=folder/"render_EXIT.json"
            poses.append(folder.name.removeprefix("render_").removesuffix("_001") if hasattr(str,"removeprefix") else folder.name[7:-4])
        else:raise ValueError("UNKNOWN_GPU_LEDGER_KIND")
        exit=read(p)
        if exit["exit_code"]!=0 or exit["status"]!="PASS":raise ValueError("HISTORICAL_GPU_EXIT")
        receipts.append(dict(path=str(p),sha256=sha(p),exit_code=exit["exit_code"]))
    if sum(r["kind"]=="counts" for r in rows)!=7 or len(poses)!=3 or len(set(poses))!=3:raise ValueError("LEDGER_COUNT")
    return dict(ledger_path=str(ledger),ledger_sha256=sha(ledger),counts=7,depth=3,completed_poses=poses,
        total_GPU_processes=10,receipts=receipts,new_counts_allowed=0,remaining_depth_budget=11)
def make_audit_geometry(data,counts,meta,cert):
    data=Path(data);tracked={str(data/n):sha(data/n) for n in ("cloud_xyz.f32","occupancy_counts.i32","grid.json","scene.json")}
    def verify():
        for p,h in tracked.items():
            if sha(p)!=h:raise ValueError("AUDIT_NATIVE_SOURCE_DRIFT")
    origin=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    roi=np.stack([origin+1e-5,origin+np.asarray(counts.shape)*h-1e-5])
    identity=read(data/"scene.json")["identity"]
    return TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,pose=cert,roi=roi,domain=[[0,-4,-2],[10,4,2]],
        identity=dict(scene=data.name,geometry_scene_id=identity["spatial_occupancy_sha256"],audit_only=True,no_sensor_source_claim=True),revalidate=verify)
def actual_gate_geometry(meta,counts,layout,geometry):
    """All native x slices of actual obstacle and full vertical column checks."""
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    is_slit="slab_box_m" in layout;box=np.asarray(layout["slab_box_m" if is_slit else "block_box_m"],float)
    xs=o[0]+(np.arange(counts.shape[0])+.5)*h
    ids=np.where((xs>=box[0,0]-.06)&(xs<=box[1,0]+.06))[0]
    gate_x=layout["gate_x_m"];columns=[]
    zlo=max(1,int(np.floor((1.8-.150011-o[2])/h)));zhi=min(counts.shape[2],int(np.ceil((1.8+.150011-o[2])/h)))
    occupied=(counts[ids,:,zlo:zhi]>0).any(axis=(0,2))
    ys=o[1]+(np.arange(counts.shape[1])+.5)*h
    known_y=[max(geometry.roi[0,1]+.150001,geometry.origin[1]-4),min(geometry.roi[1,1]-.150001,geometry.origin[1]+4)]
    zcenter=[max(geometry.floor+.150001,geometry.origin[2]-2),min(geometry.roi[1,2]-.150001,geometry.origin[2]+2)]
    for ix in ids:
        relevant=(ys>=box[0,1]-.06)&(ys<=box[1,1]+.06)
        ii=np.argwhere((counts[ix]>0)&relevant[:,None]&(np.arange(counts.shape[2])[None,:]>0))
        if not len(ii):continue
        z0=int(ii[:,1].min());z1=int(ii[:,1].max())
        yactive=np.unique(ii[:,0])
        # Slit has disconnected y support; validate all present vertical columns.
        solid=bool(all((counts[ix,y,z0:z1+1]>0).all() for y in yactive))
        low=o[2]+z0*h-1e-5;high=o[2]+(z1+1)*h+1e-5
        columns.append(dict(x_index=int(ix),z_range_m=[float(low),float(high)],y_occupied_indices=yactive.tolist(),
            continuous_vertical_columns=solid,no_over_under=bool(solid and low-.15<=zcenter[0] and high+.15>=zcenter[1])))
    common=dict(actual_x_slices=columns,body_center_altitude_m=zcenter,known_center_y_m=known_y,
                resolution_m=h,outward_reserve_m=1e-5,reference_sphere_clearance_m=.15,
                no_over_under=bool(columns and all(c["no_over_under"] for c in columns)),
                geometry_only_not_dynamic_certificate=True)
    if is_slit:
        center=sum(layout["remove_y_interval_m"])/2;j=int(np.floor((center-o[1])/h))
        if occupied[j]:return dict(common,kind="slit",gap_m=0.,center_corridor_width_m=0.,geometric_corridor="BLOCKED_AT_DECLARED_OPENING")
        lo=hi=j
        while lo>0 and not occupied[lo-1]:lo-=1
        while hi+1<len(occupied) and not occupied[hi+1]:hi+=1
        gap=(hi-lo+1)*h-2e-5
        return dict(common,kind="slit",gap_m=float(gap),center_corridor_width_m=float(gap-.3),
            opening_world_y_m=[float(o[1]+lo*h+1e-5),float(o[1]+(hi+1)*h-1e-5)],
            geometric_corridor="POSITIVE_CLEARANCE_INTERVAL" if gap>.300002 else "NO_CERTIFIED_CENTER_INTERVAL")
    active=np.where(occupied)[0];lo=int(active.min());hi=int(active.max())
    ylow=o[1]+lo*h-1e-5;yhigh=o[1]+(hi+1)*h+1e-5
    negative=[known_y[0],float(ylow-.150001)];positive=[float(yhigh+.150001),known_y[1]]
    xactive=[c["x_index"] for c in columns]
    gate=dict(x=gate_x,block_y_low=float(ylow),block_y_high=float(yhigh),block_x_high=float(o[0]+(max(xactive)+1)*h+1e-5))
    return dict(common,kind="two_portals",negative_center_interval_m=negative,positive_center_interval_m=positive,
        both_geometric_intervals_positive=bool(negative[1]>negative[0] and positive[1]>positive[0]),
        gate=gate)
def original_reproduction(run):
    manifest=OLD/"data/train/train_sparse_1102/p0/near/SAMPLE_MANIFEST.json"
    batch=load_e1_bound(manifest,sha(manifest))
    old=read(OLD/"runs/qualification_001/train_sparse_1102_p0_near_TASK.json")["nominal"]
    c=np.asarray(old["coefficients_B0"]);d=np.asarray(old["durations"]);g=batch.geometry[0];s=batch.state[0].numpy()
    result=evaluate_witness(c,d,s,g)
    expected={k:v for k,v in old["hard"].items() if k!="boundary_PVA_max_abs"}
    if result["hard"]!=expected or result["qualified_witness"]:raise ValueError("ORIGINAL_HARD_REPRODUCTION_MISMATCH")
    collisions=[]
    for i,(piece,T) in enumerate(zip(c,d)):
        for t in (0.,float(T/2),float(T)):
            b0=evaluate(piece,t);world=g.rotation@b0+g.origin
            distances=np.linalg.norm(np.maximum(np.maximum(g.boxes[:,0]-world,world-g.boxes[:,1]),0),axis=1)
            j=int(np.argmin(distances))
            if distances[j]<.15:
                collisions.append(dict(segment=i,t_segment_s=t,point_B0=b0.tolist(),point_world=world.tolist(),
                    independent_closed_box_distance_m=float(distances[j]),box_index=j,box_world=g.boxes[j].tolist()))
    if not collisions:raise ValueError("NO_INDEPENDENT_COLLISION_WITNESS")
    record=dict(status="REPRODUCED",old_task_state="UNKNOWN preserved",original_manifest_sha256=sha(manifest),
        actual_Hard=result,independent_point_collision_witnesses=collisions,
        implication="one witness invalid; task not thereby impossible",old_task_sha256=sha(OLD/"runs/qualification_001/train_sparse_1102_p0_near_TASK.json"))
    write(run/"ORIGINAL_NEAR_REPRODUCTION.json",record)
def audit():
    torch.set_num_threads(1);protect();run=OUT/"runs/audit_001";run.mkdir(parents=True)
    config=read(PREP/"PROTOCOL.json");oldcfg=read(OLD/"_preparation/qualification_001/CONFIG.json")
    ledger=ledger_audit();write(run/"LEDGER_AUDIT.json",ledger);original_reproduction(run)
    identities=[];tasks=[];geometry_rows=[];poses=[];hard_calls=0
    for scene in oldcfg["scenes"]:
        protect();sid=scene["scene_id"];data=OLD/"data"/scene["split"]/sid
        cloud,counts,meta=scene_data(data);identity=geometry_identity(cloud,counts,meta)
        if read(data/"scene.json")["identity"]!=identity:raise ValueError("SCENE_IDENTITY_DRIFT")
        identities.append(dict(scene_id=sid,split=scene["split"],**identity));check_split(identities)
        certs=choose_poses(meta,counts)
        for pi,cert in enumerate(certs):
            pose_id=sid+"_p"+str(pi);oldcert=data/("p"+str(pi))/"POSE_CERTIFICATE.json"
            if oldcert.exists():
                previous=read(oldcert)
                if cert["position_world"]!=previous["position_world"] or not previous["passed"]:raise ValueError("FROZEN_POSE_DRIFT")
            g=make_audit_geometry(data,counts,meta,cert)
            pose=dict(pose_id=pose_id,scene_id=sid,split=scene["split"],certificate=cert,
                already_rendered=pose_id in ledger["completed_poses"],counts_sha256=sha(data/"occupancy_counts.i32"),
                raw_cloud_sha256=sha(data/"cloud_xyz.f32"))
            poses.append(pose)
            layout=oldcfg["layout_precommit"].get(sid)
            geom=actual_gate_geometry(meta,counts,layout,g) if layout else dict(kind="not_portal_or_slit")
            if pi==0:geometry_rows.append(dict(scene_id=sid,split=scene["split"],geometry=geom))
            gate=geom.get("gate") if geom.get("no_over_under") and geom.get("both_geometric_intervals_positive") else None
            for statecase in oldcfg["state_cases"]:
                s=np.array(statecase["v0_B0_mps"]+statecase["a0_B0_mps2"]+statecase["g_B0_m"]+statecase["gravity_B0"],np.float32)
                sample_id=pose_id+"_"+statecase["case"];manifest=data/("p"+str(pi))/statecase["case"]/"SAMPLE_MANIFEST.json"
                source_qualified=manifest.exists()
                if source_qualified:load_e1_bound(manifest,sha(manifest)).validate()
                result=search_task(s,g,source_qualified,config,gate=gate if statecase["case"]=="far" else None)
                if geom.get("kind")=="two_portals" and statecase["case"]=="far" and gate is None:
                    result["portal_coverage"]="UNKNOWN"
                hard_calls+=result["attempts_executed"]
                if hard_calls>config["maximum_28_task_calls"]:raise ValueError("PREREGISTERED_CALL_BUDGET")
                result.update(sample_id=sample_id,scene_id=sid,split=scene["split"],state_case=statecase["case"],
                    full_sensor_contract="existing byte-bound sample" if source_qualified else "PENDING; read-only counts/pose task audit only",
                    actual_gate_geometry=geom,existing_manifest=str(manifest) if source_qualified else None)
                write(run/(sample_id+"_REACHABILITY.json"),result);tasks.append(result)
    pending=[p for p in poses if not p["already_rendered"]]
    if len(pending)!=11:raise ValueError("REMAINING_POSE_COUNT")
    write(run/"POSE_PLAN.json",poses)
    write(run/"GEOMETRY_PREFLIGHT.json",geometry_rows)
    write(run/"TASK_STATE_INDEX.json",[dict(sample_id=t["sample_id"],source=t["source"],reachability=t["reachability"],data_eligible=t["data_eligible"],
        attempts=t["attempts_executed"],portal_coverage=t["portal_coverage"],file=str(run/(t["sample_id"]+"_REACHABILITY.json"))) for t in tasks])
    from collections import Counter
    write(run/"RESULT.json",dict(contract_audit="PASS",tasks=28,reachability_counts=dict(Counter(t["reachability"] for t in tasks)),
        source_qualified=sum(t["source"]=="SOURCE_QUALIFIED" for t in tasks),new_CUDA_calls=0,remaining_poses=11,Hard_calls=hard_calls,
        continuation_proposal_allowed=True,training_or_calibration_allowed=False,
        note="contract PASS includes honest UNKNOWN/INFEASIBLE task outcomes; old dataset labels unchanged"))
    protect()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",choices=["audit"],required=True);parser.parse_args();audit()
if __name__=="__main__":main()
