"""Versioned real-source E1-R2 factory. No synthetic tag or old-loader changes."""
from pathlib import Path
import json
import numpy as np
import torch
from .binding import BoundBatch
from .geometry import TypedBoundOccupancy,_SEAL
from .e1_dataset_qualification import sha,read,digest,scene_data,camera_qualification,geometry_identity,check_split,DOMAIN
from .pose_preflight import certify_position

def load_e1_bound(path,expected_sha):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError("MANIFEST_SHA_DRIFT")
    raw=path.read_bytes();m=json.loads(raw)
    if (m["schema"]!="e1_r2_bound_sample_v1" or m["frame"]!="B0-FLU" or
        m["teacher_loaded"] is not False or m["split"] not in ("train","dev")):raise ValueError("BOUND_SCHEMA")
    def verify():
        if path.read_bytes()!=raw:raise ValueError("MANIFEST_SHA_DRIFT")
        for p,h in m["files_sha256"].items():
            if sha(p)!=h:raise ValueError("SOURCE_BYTE_DRIFT "+p)
    verify()
    scene=Path(m["scene_dir"]);pose=Path(m["pose_dir"])
    required=[scene/"cloud_xyz.f32",scene/"occupancy_counts.i32",scene/"grid.json",scene/"scene.json",
              pose/"renderer.json",pose/"depth_z.f32",pose/"POSE_CERTIFICATE.json",pose/"CAMERA_BINDING.json",
              pose/"depth_input_2x128x160.f32",path.parent/"STATE.json",Path(m["binary_path"]),Path(m["scene_registry_path"])]
    if any(str(p.resolve()) not in m["files_sha256"] for p in required):raise ValueError("UNBOUND_PATH")
    info=read(scene/"scene.json")
    if info["split"]!=m["split"] or info["scene_id"]!=m["scene_id"] or info["identity"]!=m["scene_identity"]:raise ValueError("CROSS_SPLIT_SCENE")
    cloud,counts,grid=scene_data(scene)
    actual_identity=geometry_identity(cloud,counts,grid)
    if actual_identity!=m["scene_identity"]:raise ValueError("SCENE_IDENTITY")
    registry=read(m["scene_registry_path"]);check_split(registry)
    if sum(r["scene_id"]==m["scene_id"] and r["split"]==m["split"] and all(r[k]==v for k,v in actual_identity.items()) for r in registry)!=1:raise ValueError("SCENE_REGISTRY_BINDING")
    meta=read(pose/"renderer.json");cert=read(pose/"POSE_CERTIFICATE.json")
    if meta["mode"]!="depth" or meta["schema"]!="e1_native_cuda_two_stage_v1":raise ValueError("RENDERER_SCHEMA")
    for key in ("grid_origin","grid_shape","resolution_m","occupy_threshold","point_count"):
        if meta[key]!=grid[key]:raise ValueError("RENDER_COUNTS_GRID_DRIFT")
    proof=certify_position(meta["position_world"],grid,counts)
    if (cert["schema"]!="e1_r2_pose_v1" or not cert["passed"] or not proof["passed"] or
        not np.array_equal(np.asarray(cert["position_world"],np.float32),np.asarray(meta["position_world"],np.float32)) or cert["R_WB0"]!=np.eye(3).tolist() or
        abs(proof["clearance_lower_bound_m"]-cert["clearance_lower_bound_m"])>1e-9 or
        cert["source_cloud_sha256"]!=sha(scene/"cloud_xyz.f32") or
        cert["source_occupancy_sha256"]!=sha(scene/"occupancy_counts.i32")):raise ValueError("POSE_BINDING")
    depth=np.fromfile(pose/"depth_z.f32","<f4").reshape(90,160)
    x,camera=camera_qualification(depth,counts,meta)
    if camera!=read(pose/"CAMERA_BINDING.json") or x.tobytes()!=(pose/"depth_input_2x128x160.f32").read_bytes():raise ValueError("CAMERA_BINDING")
    state=read(path.parent/"STATE.json")
    if state["snapshot_time_s"]!=0 or state["origin"]!="explicit offline initial condition, not measured odometry":raise ValueError("STATE_PROVENANCE")
    R=np.asarray(cert["R_WB0"]);v=np.r_[R.T@state["v0_world_mps"],R.T@state["a0_world_mps2"],state["g_B0_m"],R.T@state["gravity_world_unit"]]
    if v.shape!=(12,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v[-3:]),1):raise ValueError("STATE_12D")
    origin=np.asarray(grid["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    roi=np.stack([origin+1e-5,origin+np.asarray(counts.shape)*h-1e-5])
    identity=dict(sample_id=m["sample_id"],source_scene_id=m["scene_id"],geometry_scene_id=m["scene_identity"]["spatial_occupancy_sha256"],
                  split=m["split"],frame="B0-FLU",renderer_binary_sha256=sha(m["binary_path"]),qualification_sha256=digest(cert))
    geometry=TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=grid,pose=cert,roi=roi,
                                domain=DOMAIN,identity=identity,revalidate=verify)
    batch=BoundBatch(torch.from_numpy(x[None].copy()),torch.tensor(v[None],dtype=torch.float32),(geometry,),m,verify)
    batch._depth=batch.depth.numpy().tobytes();batch._state=batch.state.numpy().tobytes();batch.validate()
    return batch
