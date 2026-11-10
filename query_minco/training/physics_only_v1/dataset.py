"""Independent pinned depth/state/native-AABB loader. No trainer or legacy dataset.

SHA pins establish byte identity, not truth of certificates or forest equivalence.
Only native AABB geometry is supported. No implicit forest-to-box conversion.
All returned tensors are CPU FP32; certificates attest provenance, not flight safety.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from query_minco.training.io import exact, require

FRAME = "fixed B0 FLU"
CAMERA = dict(frame="optical RDF", optical_to_B0_rotation=[[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
              origin_B0=[0., 0., 0.], depth_encoding="normalized_linear_z_invalid_zero",
              depth_scale_m=20.0)
STATE_LAYOUT = "velocity_mps,acceleration_mps2,goal_m,gravity_unit"


def _pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, "DUPLICATE_JSON_KEY")
        result[k] = v
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


class PhysicsOnlyDataset:
    """Caller pins manifest SHA and allowed roots independently of manifest contents.

    qualification must be explicitly SYNTHETIC_TEST_ONLY for test fixtures.
    load revalidates the entire binding chain, parsing exactly the bytes hashed.
    FrozenDataset/GeometrySidecar and teacher arrays are never used.
    """
    def __init__(self, binding, *, roots, qualification="TRAIN"):
        require(qualification in ("TRAIN", "SYNTHETIC_TEST_ONLY"), "QUALIFICATION")
        self._binding = copy.deepcopy(binding)
        self._roots = tuple(Path(p).resolve(strict=True) for p in roots)
        require(self._roots and all(p.is_dir() for p in self._roots), "ROOTS")
        self.qualification = qualification
        self._manifest = self._read_manifest()
        self._rows = {}
        artifacts = set()
        for row in self._manifest["rows"]:
            exact(row, ["sample_id", "split", "sample", "geometry", "certificate"], "physics_row")
            sid = row["sample_id"]
            require(isinstance(sid, str) and sid and sid not in self._rows, "SAMPLE_ID")
            require(row["split"] == "train", "NONTRAIN")
            sample_path = str(Path(row["sample"]["path"]).resolve())
            require(sample_path not in artifacts, "DUPLICATE_SAMPLE_ARTIFACT")
            artifacts.add(sample_path)
            self._rows[sid] = copy.deepcopy(row)
            self._load_row(row)  # Fail before exposing a partially validated dataset.
        require(self._rows, "EMPTY_DATASET")

    @property
    def sample_ids(self):
        return tuple(self._rows)

    def _bytes(self, binding):
        exact(binding, ["path", "sha256"], "binding")
        digest = binding["sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), "SHA256_FORMAT")
        path = Path(binding["path"])
        require(path.is_absolute(), "ABSOLUTE_PATH")
        resolved = path.resolve(strict=True)
        require(any(root in resolved.parents for root in self._roots), "PATH_ESCAPE")
        require(not any("blind" in part.lower() for part in resolved.parts), "BLIND_PATH")
        data = resolved.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "SHA256_MISMATCH")
        return data

    def _read_manifest(self):
        m = _json(self._bytes(self._binding))
        exact(m, ["schema", "qualification", "selection_provenance", "rows"], "physics_manifest")
        require(m["schema"] == "physics_only_manifest_v1", "SCHEMA_VERSION")
        require(m["qualification"] == self.qualification, "QUALIFICATION_MISMATCH")
        require(m["selection_provenance"] in ("independent", "teacher_selected", "unknown"), "SELECTION_PROVENANCE")
        require(isinstance(m["rows"], list), "ROWS")
        return m

    def _load_row(self, row):
        sample_bytes = self._bytes(row["sample"])
        geometry = _json(self._bytes(row["geometry"]))
        cert = _json(self._bytes(row["certificate"]))
        exact(cert, ["schema", "sample_id", "split", "sample", "geometry", "qualification",
                     "frame", "units", "camera", "state", "empty_certified"], "certificate")
        require(cert["schema"] == "physics_binding_certificate_v1" and
                cert["qualification"] == self.qualification, "CERTIFICATION")
        require(cert["sample_id"] == row["sample_id"] and cert["split"] == row["split"] == "train" and
                cert["sample"] == row["sample"] and cert["geometry"] == row["geometry"], "SAMPLE_GEOMETRY_BINDING")
        exact(geometry, ["schema", "sample_id", "split", "representation", "frame", "units",
                         "domain_semantics", "boxes", "domain", "empty_certified"], "geometry")
        require(geometry["schema"] == "physics_native_aabb_v1" and
                geometry["representation"] == "native_aabb", "UNVERIFIED_GEOMETRY_REPRESENTATION")
        require(geometry["sample_id"] == row["sample_id"] and geometry["split"] == "train", "GEOMETRY_SAMPLE_SPLIT")
        require(geometry["frame"] == cert["frame"] == FRAME and geometry["units"] == cert["units"] == "m_s", "FRAME_UNITS")
        require(geometry["domain_semantics"] == "center_bounds_not_occupied_solids", "DOMAIN_SEMANTICS")
        require(cert["camera"] == CAMERA, "CAMERA_CONTRACT")
        empty = geometry["empty_certified"]
        require(type(empty) is bool and type(cert["empty_certified"]) is bool and
                empty == cert["empty_certified"], "EMPTY_CERTIFICATION")
        boxes = np.asarray(geometry["boxes"], dtype=np.float32)
        if geometry["boxes"] == []:
            boxes = boxes.reshape(0, 2, 3)
        domain = np.asarray(geometry["domain"], dtype=np.float32)
        require(boxes.ndim == 3 and boxes.shape[1:] == (2, 3) and np.isfinite(boxes).all(), "BOX_SHAPE_FINITE")
        require(bool(len(boxes) == 0) == empty and (boxes[:, 0] < boxes[:, 1]).all(), "BOX_EMPTY_EXTENTS")
        require(domain.shape == (2, 3) and np.isfinite(domain).all() and (domain[0] < domain[1]).all() and
                (domain[0] <= 0).all() and (domain[1] >= 0).all(), "DOMAIN")
        with np.load(io.BytesIO(sample_bytes), allow_pickle=False) as z:
            require(set(z.files) == {"sample_id", "depth_input", "state"}, "SAMPLE_ARRAYS")
            require(z["sample_id"].shape == () and z["sample_id"].dtype.kind == "U" and
                    z["sample_id"].item() == row["sample_id"], "NPZ_SAMPLE_ID")
            depth, state = z["depth_input"].copy(), z["state"].copy()
        for arr, shape in ((depth, (2, 128, 160)), (state, (12,))):
            require(arr.dtype == np.float32 and arr.shape == shape and np.isfinite(arr).all(), "SAMPLE_SHAPE_DTYPE_FINITE")
        require(np.isin(depth[1], [0., 1.]).all() and ((depth[0] >= 0) & (depth[0] <= 1)).all() and
                (depth[0][depth[1] == 0] == 0).all(), "DEPTH_RANGE_MASK")
        expected = np.asarray(cert["state"], dtype=np.float32)
        require(expected.shape == (12,) and np.isfinite(expected).all() and np.array_equal(state, expected), "STATE_BINDING")
        require(abs(float(np.linalg.norm(state[9:12])) - 1.) < 1e-4, "GRAVITY_UNIT")
        return SimpleNamespace(sample_id=row["sample_id"], depth_input=torch.from_numpy(depth),
                               state=torch.from_numpy(state), boxes=torch.from_numpy(boxes),
                               domain=torch.from_numpy(domain), empty_certified=empty,
                               qualification=self.qualification, camera=copy.deepcopy(CAMERA),
                               state_layout=STATE_LAYOUT, binding=copy.deepcopy(row),
                               selection_provenance=self._manifest["selection_provenance"])

    def load(self, ids):
        require(self._read_manifest() == self._manifest, "MANIFEST_CHANGED")
        ids = tuple(ids)
        require(ids and len(set(ids)) == len(ids) and all(sid in self._rows for sid in ids), "UNKNOWN_OR_DUPLICATE_ID")
        return [self._load_row(self._rows[sid]) for sid in ids]


def optical_points_to_b0(points):
    """RDF [right,down,forward] -> FLU [forward,left,up], co-located camera."""
    require(isinstance(points, torch.Tensor) and points.shape[-1:] == (3,) and
            points.is_floating_point() and bool(torch.isfinite(points).all()), "CAMERA_POINTS")
    return points @ points.new_tensor(CAMERA["optical_to_B0_rotation"]).T
