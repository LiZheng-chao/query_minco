"""Read exact exported bytes; prove occupancy mapping and reject unsafe initial states."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .camera import camera_input,replay_depth


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def qualify_export(data, expected_sha256, binary_binding, pose_certificate=None):
    data=Path(data)
    buffers={}
    for name,h in expected_sha256.items():
        raw=(data/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError("EXPORT_SHA_DRIFT "+name)
        buffers[name]=raw
    if set(buffers)!={"renderer.json","cloud_xyz.f32","occupancy_counts.i32","depth_z.f32"}:
        raise ValueError("export file schema mismatch")
    if hashlib.sha256(Path(binary_binding["path"]).read_bytes()).hexdigest()!=binary_binding["sha256"]:
        raise ValueError("BINARY_SHA_DRIFT")
    m=json.loads(buffers["renderer.json"])
    if m["schema"]!="yopo_original_renderer_export_v1" or m["case"]!="sparse" or m["seed"]!=1001:
        raise ValueError("scene identity mismatch")
    c=m["camera"]
    if [c[k] for k in ("fx","fy","cx","cy")]!=[80,80,80,45]:
        raise ValueError("CAMERA_K_MISMATCH")
    if m["occupy_threshold"]!=0 or np.float32(m["resolution_m"])!=np.float32(.1):
        raise ValueError("OCCUPANCY_PROFILE_MISMATCH")
    shape=tuple(m["grid_shape"])
    if len(shape)!=3 or min(shape[:2])<2 or shape[2]<1:raise ValueError("invalid grid dimensions")
    cloud=np.frombuffer(buffers["cloud_xyz.f32"],dtype="<f4").reshape(-1,3)
    counts=np.frombuffer(buffers["occupancy_counts.i32"],dtype="<i4").reshape(shape)
    depth=np.frombuffer(buffers["depth_z.f32"],dtype="<f4").reshape(90,160)
    if len(cloud)!=m["point_count"] or not np.isfinite(cloud).all() or (counts<0).any():
        raise ValueError("invalid raw cloud/counts")
    origin=np.asarray(m["grid_origin"],np.float32)
    resolution=np.float32(m["resolution_m"])
    if not np.array_equal(cloud.min(0),origin):raise ValueError("grid/cloud origin mismatch")
    expected_shape=np.ceil((cloud.max(0)-cloud.min(0)+np.float32(.002))/resolution).astype(np.int64)
    if tuple(expected_shape)!=shape:raise ValueError("grid shape differs from original constructor")
    shifted=cloud+np.float32(.001)
    indices=np.floor((shifted-origin)/resolution).astype(np.int64)
    if (indices<0).any() or (indices>=np.array(shape)).any():raise ValueError("point indexing out of bounds")
    flat=(indices[:,0]*shape[1]+indices[:,1])*shape[2]+indices[:,2]
    reconstructed=np.bincount(flat,minlength=int(np.prod(shape))).reshape(shape)
    if not np.array_equal(counts,reconstructed):raise ValueError("exported CUDA occupancy differs from independent reconstruction")
    input_tensor,camera=camera_input(depth,m)
    replay=replay_depth(counts,m)
    error=np.abs(replay.astype(np.float64)-depth)
    # Fixed floating-point conformance allowance, not a safety/geometry relaxation.
    tolerance=2e-5
    mismatches=int((error>tolerance).sum())
    p=np.asarray(m["position_world"],np.float32)
    world_voxel=np.floor((p-origin)/resolution).astype(np.int64)
    folded=world_voxel.copy()
    for axis in (0,1):
        folded[axis]%=2*shape[axis]-2
        if folded[axis]>=shape[axis]:folded[axis]=2*shape[axis]-2-folded[axis]
    if folded[2]>=shape[2]:start_occupied=False;start_count=None
    elif folded[2]<=0:start_occupied=True;start_count=None
    else:start_count=int(counts[tuple(folded)]);start_occupied=start_count>m["occupy_threshold"]
    reasons=[]
    if mismatches:reasons.append("DEPTH_REPLAY_MISMATCH")
    if start_occupied:reasons.append("INITIAL_BODY_CENTER_IN_OCCUPIED_VOXEL")
    # A free center alone would still require a 0.12+0.03m independent clearance proof.
    if not start_occupied:
        if pose_certificate is None:
            reasons.append("REFERENCE_BODY_CLEARANCE_NOT_YET_CERTIFIED")
        else:
            from .pose_preflight import certify_position
            proof=certify_position(p,m,counts)
            if (not proof["passed"] or not pose_certificate["passed"] or
                not np.array_equal(p,np.asarray(pose_certificate["position_world"],np.float32)) or
                expected_sha256["occupancy_counts.i32"]!=pose_certificate["source_occupancy_sha256"] or
                expected_sha256["cloud_xyz.f32"]!=pose_certificate["source_cloud_sha256"] or
                pose_certificate["schema"]!="gate1_pose_certificate_v1" or
                abs(proof["clearance_lower_bound_m"]-pose_certificate["clearance_lower_bound_m"])>1e-9):
                reasons.append("POSE_CERTIFICATE_MISMATCH")
    scene=dict(cloud_sha256=expected_sha256["cloud_xyz.f32"],
               occupancy_sha256=expected_sha256["occupancy_counts.i32"],shape=list(shape),
               origin=m["grid_origin"],resolution_m=m["resolution_m"],point_shift_m=.001,
               occupied_rule="count>threshold",threshold=m["occupy_threshold"],
               xy_semantics="symmetricIndex period 2*n-2",z_semantics="index<=0 occupied; index>=nz free",
               case=m["case"],seed=m["seed"],binary_sha256=binary_binding["sha256"])
    result=dict(pose_certificate=pose_certificate,schema="gate1_single_export_qualification_v1",status="REJECTED" if reasons else "QUALIFIED",
                sample_id=data.name,split_candidate="train",training_qualified=not reasons,
                rejection_reasons=reasons,raw_exports_sha256=expected_sha256,
                scene_sha256=digest(scene),scene_contract=scene,
                point_count=int(len(cloud)),grid_shape=list(shape),grid_cells=int(counts.size),
                occupied_base_cells=int((counts>m["occupy_threshold"]).sum()),
                occupancy_reconstruction_exact=True,occupancy_count_sum=int(counts.sum()),
                depth_shape=list(depth.shape),depth_finite=bool(np.isfinite(depth).all()),
                depth_range_m=[float(depth.min()),float(depth.max())],
                independent_depth_replay_max_error_m=float(error.max()),
                independent_depth_replay_mismatch_pixels=mismatches,replay_tolerance_m=tolerance,
                camera=camera,start_world_voxel=world_voxel.tolist(),
                start_mirrored_voxel=folded.tolist(),start_occupancy_count=start_count,
                start_occupied=start_occupied,reference_body_radius_m=.12,required_margin_m=.03,
                body_clearance_contract="occupied center/boundary cannot certify positive body clearance",
                unknown_space_not_certified_free=True,production_objective_ready=False,
                model_forward_backward="NOT_RUN",optimizer_updates=0,episodes=0)
    return result,input_tensor
