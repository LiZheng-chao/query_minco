"""Fixed order validation; never shuffle, resample, or sort cache directories."""
from collections import Counter
import numpy as np
from ..io import artifact, require
from ..run_spec import PROFILES

ROLES = ("backbone_near", "backbone_far", "supplement_near", "supplement_far")
class FrozenOrder:
    def __init__(self, binding, dataset, profile):
        self.path = artifact(binding)
        with np.load(self.path, allow_pickle=False) as z:
            require(set(z.files) == {"task_ids", "roles"}, "ORDER_FIELDS")
            self.ids = z["task_ids"].copy()
            self.roles = z["roles"].copy()
        cfg = PROFILES[profile]
        require(self.ids.dtype.kind == "U" and self.roles.dtype.kind == "U" and
                self.ids.shape == self.roles.shape == (cfg["updates"], cfg["batch_size"]), "ORDER_SHAPE")
        teacher_count = 0
        used = set()
        for ids, roles in zip(self.ids.tolist(), self.roles.tolist()):
            for sid, role in zip(ids, roles):
                require(sid in dataset.rows, "ORDER_UNAUTHORIZED_TASK")
                row = dataset.rows[sid]
                require(role in row["roles"], "ORDER_TASK_ROLE")
                require(role.endswith(row["regime"]), "ORDER_REGIME")
                used.add(sid)
                teacher_count += row["M"]
            if profile == "b2c1_reference_v1":
                require(Counter(roles) == {role: 2 for role in ROLES}, "REFERENCE_BATCH_ROLES")
        require(used == set(dataset.rows), "UNUSED_OR_MISSING_MANIFEST_TASK")
        if profile == "b2c1_reference_v1":
            require(teacher_count == 40320 and self.ids.size == 23040, "REFERENCE_EXPOSURE")
        self.summary = dict(updates=len(self.ids), task_exposures=int(self.ids.size),
                            teacher_exposures=teacher_count, unique_tasks=len(used),
                            task_counts=dict(Counter(self.ids.ravel().tolist())),
                            role_counts=dict(Counter(self.roles.ravel().tolist())))
    def batch(self, index):
        return self.ids[index].tolist(), self.roles[index].tolist()
