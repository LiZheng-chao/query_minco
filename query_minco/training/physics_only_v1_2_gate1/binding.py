"""Byte-bound real CUDA sample loader. No teacher feature, file, or filter."""
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from .geometry import TypedBoundOccupancy,_SEAL
from .qualification import qualify_export,digest
from .camera import camera_input


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class BoundBatch:
    depth: torch.Tensor
    state: torch.Tensor
    geometry: tuple
    manifest: dict
    verify: object

    def validate(self):
        self.verify()
        if (self.depth.shape!=(1,2,128,160) or self.state.shape!=(1,12) or
            self.depth.dtype!=torch.float32 or self.state.dtype!=torch.float32 or
            self.depth.device.type!="cpu" or self.state.device.type!="cpu"):
            raise ValueError("BOUND_BATCH_SHAPE_TYPE")
        if self.depth.numpy().tobytes()!=self._depth or self.state.numpy().tobytes()!=self._state:
            raise ValueError("BOUND_INPUT_MEMORY_DRIFT")
        if len(self.geometry)!=1 or type(self.geometry[0]) is not TypedBoundOccupancy:
            raise ValueError("PRODUCTION_TYPE_REQUIRED")
        self.geometry[0].validate()


def write_manifest(output):
    root=Path(output)
    run=root/"runs/render_posecert_001"
    raw=run/"sample/train_sparse_1001_posecert_001"
    prep=root/"_preparation/pose_geometry_gradient_001"
    config=json.loads((prep/"CONFIG.json").read_text())
    certpath=root/"runs/pose_preflight_001/POSE_CERTIFICATE.json"
    files={str(p.resolve()):sha(p) for p in list(raw.iterdir())+[
        run/"BINARY_BINDING.json",run/"EXPORTED_SHA256.json",certpath,prep/"CONFIG.json",
        root/"runs/bind_posecert_001/EXPORT_QUALIFICATION.json"]}
    binary=json.loads((run/"BINARY_BINDING.json").read_text())
    files[binary["path"]]=binary["sha256"]
    pose=json.loads(certpath.read_text())
    counts_sha=pose["source_occupancy_sha256"]
    result=dict(schema="gate1_typed_bound_sample_v1",sample_id=raw.name,split="train",
        source_scene_id=pose["source_scene_id"],geometry_scene_id=counts_sha,
        split_registry={counts_sha:"train"},files_sha256=files,
        raw_dir=str(raw.resolve()),pose_certificate_path=str(certpath.resolve()),
        binary_binding_path=str((run/"BINARY_BINDING.json").resolve()),
        config_path=str((prep/"CONFIG.json").resolve()),
        export_hashes_path=str((run/"EXPORTED_SHA256.json").resolve()),
        state=config["state"],pose=pose,units=dict(position="m",velocity="m/s",acceleration="m/s2"),
        frame="B0-FLU",camera_profile="Z/20+valid; pad19+19; raw K(80,80,80,45); processed cy64",
        geometry_authority="complete generated source occupancy with audited exact XY mirror extension; not sensor reconstruction",
        unknown_policy="reject outside finite ROI and above source top; no-hit depth is invalid, never free authority",
        external_certificate_authority=False,teacher_loaded=False)
    result["contract_sha256"]=digest(result)
    path=root/"runs/bind_posecert_001/BOUND_MANIFEST.json"
    with path.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return path,sha(path)


def load_bound(path,expected_sha256):
    path=Path(path)
    if sha(path)!=expected_sha256:raise ValueError("MANIFEST_SHA_DRIFT")
    raw=path.read_bytes();m=json.loads(raw)
    contract=m.pop("contract_sha256")
    if digest(m)!=contract:raise ValueError("MANIFEST_CONTRACT_DRIFT")
    m["contract_sha256"]=contract
    if (m["schema"]!="gate1_typed_bound_sample_v1" or m["frame"]!="B0-FLU" or m["teacher_loaded"] is not False or
        m["units"]!=dict(position="m",velocity="m/s",acceleration="m/s2")):raise ValueError("BOUND_SCHEMA")
    def verify():
        if path.read_bytes()!=raw:raise ValueError("MANIFEST_SHA_DRIFT")
        for p,h in m["files_sha256"].items():
            if sha(p)!=h:raise ValueError("SOURCE_BYTE_DRIFT "+p)
    verify()
    def bound_json(key):
        p=m[key]
        if p not in m["files_sha256"]:raise ValueError("UNBOUND_PATH")
        return json.loads(Path(p).read_text())
    hashes=bound_json("export_hashes_path");binary=bound_json("binary_binding_path")
    cert=bound_json("pose_certificate_path");config=bound_json("config_path")
    rawdir=Path(m["raw_dir"])
    if any(str(rawdir/name) not in m["files_sha256"] or m["files_sha256"][str(rawdir/name)]!=h for name,h in hashes.items()):
        raise ValueError("UNBOUND_EXPORT")
    if binary["path"] not in m["files_sha256"] or m["files_sha256"][binary["path"]]!=binary["sha256"]:
        raise ValueError("UNBOUND_BINARY")
    qualification,depth=qualify_export(rawdir,hashes,binary,cert)
    if not qualification["training_qualified"]:raise ValueError("EXPORT_REJECTED")
    if (m["pose"]!=cert or m["state"]!=config["state"] or m["source_scene_id"]!=cert["source_scene_id"] or
        m["geometry_scene_id"]!=hashes["occupancy_counts.i32"] or
        m["sample_id"]!=rawdir.name or m["split"]!="train" or
        m["split_registry"].get(m["geometry_scene_id"])!=m["split"]):
        raise ValueError("POSE_STATE_SCENE_SPLIT_BINDING")
    # The fixed protocol has no Dev authority for any pose of this source scene.
    if config["accepted_scene_split_registry"].get(m["source_scene_id"])!=m["split"]:raise ValueError("CROSS_SPLIT_SCENE")
    state=m["state"];R=np.asarray(cert["R_WB0"],np.float64)
    vector=np.r_[R.T@state["v0_world_mps"],R.T@state["a0_world_mps2"],
                 state["g_B0_m"],R.T@state["gravity_world_unit"]]
    if vector.shape!=(12,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector[-3:]),1):
        raise ValueError("STATE_12D_GRAVITY")
    if state["snapshot_time_s"]!=0 or "not measured odometry" not in state["origin"]:
        raise ValueError("STATE_TIME_PROVENANCE")
    meta=json.loads((rawdir/"renderer.json").read_text())
    counts=np.frombuffer((rawdir/"occupancy_counts.i32").read_bytes(),"<i4").reshape(meta["grid_shape"])
    p0=np.asarray(cert["position_world"])
    roi=p0+np.asarray(config["roi_B0_offsets_m"])
    top=np.float32(meta["grid_origin"][2])+meta["grid_shape"][2]*float(np.float32(meta["resolution_m"]))
    roi[1,2]=min(roi[1,2],top-1e-5)
    identity={k:m[k] for k in ("sample_id","source_scene_id","geometry_scene_id","split","frame")}
    identity.update(renderer_binary_sha256=binary["sha256"],camera_sha256=hashes["renderer.json"],
                    qualification_sha256=digest(qualification))
    geometry=TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,pose=cert,roi=roi,
                               domain=config["center_domain_B0_m"],identity=identity,revalidate=verify)
    batch=BoundBatch(torch.from_numpy(depth[None].copy()),torch.tensor(vector[None],dtype=torch.float32),
                     (geometry,),m,verify)
    batch._depth=batch.depth.numpy().tobytes();batch._state=batch.state.numpy().tobytes()
    batch.validate()
    return batch
