"""Explicit frozen cache rows. No directory discovery and no Teacher generation."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ..io import artifact, read, require, exact

class FrozenDataset:
    def __init__(self, binding, profile):
        self.path = artifact(binding)
        self.manifest = read(self.path)
        m = self.manifest
        exact(m, ["schema", "qualification", "roots", "rows", "evidence"], "cache_manifest")
        require(m["schema"] == 1 and m["qualification"] in ("TRAIN", "SYNTHETIC_TEST_ONLY"), "DATA_QUALIFICATION")
        require((profile == "disposable_single_step_v1") == (m["qualification"] == "SYNTHETIC_TEST_ONLY"),
                "PROFILE_DATA_QUALIFICATION")
        self.roots = [Path(p).resolve() for p in m["roots"]]
        require(self.roots and all(p.is_absolute() and p.is_dir() for p in self.roots), "CACHE_ROOTS")
        self.rows = {}
        self.files = [self.path]
        for row in m["rows"]:
            exact(row, ["id", "artifact", "split", "regime", "M", "roles", "teacher_status", "certificates"], "cache_row")
            require(row["id"] not in self.rows and row["split"] == "train" and row["regime"] in ("near", "far"), "SPLIT_OR_DUPLICATE")
            require(row["M"] in (1, 4) and row["teacher_status"] == "verified_positive", "TEACHER_QUALIFICATION")
            p = self.check_artifact(row["artifact"])
            # Resolve and root-check before opening NPZ. Only explicit rows enter the loader.
            self.files.append(p)
            require(row["roles"] and len(set(row["roles"])) == len(row["roles"]), "TASK_ROLES")
            for cert in row["certificates"]:
                self.files.append(self.check_artifact(cert))
            require(profile == "disposable_single_step_v1" or row["certificates"], "MISSING_TEACHER_CERTIFICATE")
            self.rows[row["id"]] = row
        require(self.rows, "EMPTY_DATASET")
        for b in m["evidence"]:
            self.files.append(self.check_artifact(b))

    def check_artifact(self, binding):
        p = Path(binding["path"]).resolve()
        require(any(p == root or root in p.parents for root in self.roots), "CACHE_PATH_ESCAPE")
        # No path-based blacklist substitutes for the explicit train row/evidence whitelist.
        require(not any("blind" in part.lower() for part in p.parts), "BLIND_PATH")
        return artifact(binding)

    def load(self, ids):
        samples = []
        for sid in ids:
            require(sid in self.rows, "UNKNOWN_TRAIN_TASK")
            row = self.rows[sid]
            p = self.check_artifact(row["artifact"])
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in ("depth_input", "state", "teacher_coeffs", "teacher_durations")}
            shapes = {"depth_input": (2, 128, 160), "state": (12,),
                      "teacher_coeffs": (row["M"], 3, 3, 6), "teacher_durations": (row["M"], 3)}
            for k, a in arrays.items():
                require(a.dtype == np.float32 and a.shape == shapes[k] and np.isfinite(a).all(), "CACHE_SHAPE_DTYPE_FINITE " + k)
            dep = arrays["depth_input"]
            require(np.isin(dep[1], [0., 1.]).all() and ((dep[0] >= 0) & (dep[0] <= 1)).all(), "DEPTH_MASK_RANGE")
            require((dep[0][dep[1] == 0] == 0).all(), "INVALID_DEPTH_NOT_ZERO")
            duration = arrays["teacher_durations"]
            require((duration >= .12 - 1e-6).all() and (duration.sum(1) >= .8 - 1e-6).all() and
                    (duration.sum(1) <= 2.2 + 1e-6).all(), "TEACHER_TIME_CONTRACT")
            state = arrays["state"]
            require(abs(float(np.linalg.norm(state[9:12])) - 1.) < 1e-4, "GRAVITY_UNIT")
            # Frozen Near/Far semantics remain the original <=3.4 m boundary.
            require(("near" if np.linalg.norm(state[6:9].astype(float)) <= 3.4 else "far") == row["regime"], "NEAR_FAR_CONTRACT")
            samples.append(SimpleNamespace(sample_id=sid, **{k: torch.from_numpy(a) for k, a in arrays.items()}))
        return samples
