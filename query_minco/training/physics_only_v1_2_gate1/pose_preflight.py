"""Independent finite-base initial sphere clearance proof, before any rendering."""
import hashlib
import json
from pathlib import Path
import numpy as np

SAFE = .15
RESERVE = 1e-5


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_source(output):
    root=Path(output)
    data=root/"data/train_sparse_1001_001"
    bindings=json.loads((root/"runs/render_sparse_1001_002/EXPORTED_SHA256.json").read_text())
    raw={}
    for name,sha in bindings.items():
        raw[name]=(data/name).read_bytes()
        if sha_bytes(raw[name])!=sha:raise ValueError("SOURCE_SHA_DRIFT "+name)
    meta=json.loads(raw["renderer.json"])
    counts=np.frombuffer(raw["occupancy_counts.i32"],"<i4").reshape(meta["grid_shape"]).copy()
    if (counts<0).any() or counts.sum()!=meta["point_count"]:raise ValueError("SOURCE_COUNTS")
    if meta["seed"]!=1001 or meta["occupy_threshold"]!=0:raise ValueError("SOURCE_PROFILE")
    return meta,counts,bindings


def grid_boxes(meta,counts,reserve=RESERVE):
    index=np.argwhere(counts>meta["occupy_threshold"])
    index=index[index[:,2]>0]  # index<=0 is the separately represented solid floor.
    origin=np.array(meta["grid_origin"],np.float32).astype(np.float64)
    h=float(np.float32(meta["resolution_m"]))
    lo=origin+index*h-reserve
    return np.stack([lo,origin+(index+1)*h+reserve],axis=1),index


def mirrored_cell(position,meta,counts):
    p=np.asarray(position,np.float32);o=np.asarray(meta["grid_origin"],np.float32)
    i=np.floor((p-o)/np.float32(meta["resolution_m"])).astype(np.int64)
    folded=i.copy()
    for axis,n in enumerate(counts.shape[:2]):
        folded[axis]%=2*n-2
        if folded[axis]>=n:folded[axis]=2*n-2-folded[axis]
    if folded[2]<=0:return i,folded,True,None
    if folded[2]>=counts.shape[2]:return i,folded,False,None
    count=int(counts[tuple(folded)])
    return i,folded,count>meta["occupy_threshold"],count


def closed_box_distances(position,boxes):
    q=np.abs(np.asarray(position)-boxes.mean(axis=1))-(boxes[:,1]-boxes[:,0])/2
    return np.linalg.norm(q.clip(min=0),axis=1)


def certify_position(position,meta,counts,boxes=None):
    p=np.asarray(position,np.float32).astype(np.float64)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError("INVALID_POSITION")
    if boxes is None:boxes,_=grid_boxes(meta,counts)
    origin=np.asarray(meta["grid_origin"],np.float32).astype(np.float64)
    h=float(np.float32(meta["resolution_m"]))
    upper=origin+np.array(counts.shape)*h
    floor_top=origin[2]+h+RESERVE
    boundary=float(np.min(np.r_[p-origin,upper-p]))-RESERVE
    distances=closed_box_distances(p,boxes)
    native=float(distances.min()) if len(boxes) else float("inf")
    # Independent coordinate-clipping check, with no SDF formula reuse.
    nearest=np.minimum(np.maximum(p,boxes[:,0]),boxes[:,1])
    independent=float(np.linalg.norm(p-nearest,axis=1).min()) if len(boxes) else native
    if abs(native-independent)>1e-10:raise ValueError("INDEPENDENT_DISTANCE_DISAGREEMENT")
    floor=float(p[2]-floor_top)
    world,folded,occupied,count=mirrored_cell(p,meta,counts)
    lower=min(native,floor,boundary)
    passed=bool(lower>=SAFE+1e-6 and not occupied)
    return dict(passed=passed,position_world=p.tolist(),R_WB0=np.eye(3).tolist(),
                clearance_lower_bound_m=max(0.,float(lower)),
                native_closed_voxel_distance_m=native,
                independently_clipped_voxel_distance_m=independent,
                floor_distance_m=floor,known_base_boundary_distance_m=boundary,
                global_mirror_lower_bound_basis="outside-base obstacles cannot be closer than base boundary",
                world_voxel=world.tolist(),mirrored_voxel=folded.tolist(),
                start_occupied=occupied,start_count=count,
                reference_radius_m=.12,margin_m=.03,required_clearance_m=SAFE,
                voxel_outward_reserve_m=RESERVE,
                known_base_bounds_m=[origin.tolist(),upper.tolist()])


def select_pose(output):
    meta,counts,bindings=load_source(output)
    boxes,_=grid_boxes(meta,counts)
    origin=np.asarray(meta["grid_origin"],np.float32).astype(np.float64)
    h=float(np.float32(meta["resolution_m"]))
    # Enumerate within the finite source base; no cloud_min-minus-offset heuristic.
    target_y=float(meta["position_world"][1])
    ys=sorted(range(2,counts.shape[1]-2),key=lambda j:(abs(origin[1]+(j+.5)*h-target_y),j))
    visited=0;selected=None
    for i in range(2,counts.shape[0]-2):
        for j in ys:
            p=np.array([origin[0]+(i+.5)*h,origin[1]+(j+.5)*h,1.8],dtype=np.float32)
            visited+=1
            candidate=certify_position(p,meta,counts,boxes)
            if candidate["passed"]:
                selected=candidate;break
        if selected:break
    old=certify_position(meta["position_world"],meta,counts,boxes)
    if old["passed"] or not old["start_occupied"] or old["start_count"]!=28:
        raise ValueError("HISTORICAL_REJECTION_REGRESSION")
    result=dict(schema="gate1_pose_certificate_v1",passed=selected is not None,
                scene_seed=1001,case="sparse",sample_id="train_sparse_1001_posecert_001",
                source_scene_id="319ec0566f09763268e8b8c5c3099d519c3d200d622b538c8b2ec7486996e7b7",
                source_occupancy_sha256=bindings["occupancy_counts.i32"],
                source_cloud_sha256=bindings["cloud_xyz.f32"],candidates_visited=visited,
                history_rejected_pose=old,external_certificate_authority=False)
    if selected:result.update(selected)
    return result
