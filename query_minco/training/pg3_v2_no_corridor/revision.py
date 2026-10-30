"""Opt-in PG3 revision: no convex flight corridor; training-only soft constraints.

This module does NOT generate a flight certificate or grant execution permission.
The old PG3 methods remain unchanged for paired comparisons.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from query_minco.training.io import require, digest, sha
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v1.geometry import tensor_digest, scene_from_field
from query_minco.training.run_spec import ROOT

import torch
import torch.nn.functional as F

from query_minco.training.pg3_feasible_mode_v1.assignment import assign
from query_minco.training.pg3_feasible_mode_v1.bottleneck import at_progress, crossing_events


@dataclass(frozen=True)
class RevisionWeights:
    # Prespecified PILOT values only; calibrate or ablate on training split before full runs.
    section: float = 0.10
    safe_keep: float = 0.05
    full_stop: float = 0.01
    # No convex corridor, and no unconditional pairwise repulsion.


def _mode_key(t):
    """Certified AABB-face signature; NEVER a homotopy/topology assertion."""
    signature = t.get("mode_signature") or []
    return tuple(tuple(x) for x in signature) if signature else None


class PerSampleFairAssignment:
    """One transaction per optimizer update; duplicates count as separate exposures."""
    def __init__(self):
        self.history = {}
        self.pending = []
        self.last_step = -1

    def choose(self, sid, inner, tail, matches, hard_pass, teachers,
               coeffs, durations, signed, step, capacity):
        require(capacity == 2 and step == self.last_step + 1, "V2_ASSIGN_STEP_CAPACITY")
        require(not self.pending or self.pending[0][0] == step, "V2_PENDING_STEP")
        state = copy.deepcopy(self.history.get(sid, [dict(eligible=0, direct=0, wait=0) for _ in range(6)]))
        for _, psid, selected, bad in self.pending:
            if psid == sid:
                self._advance(state, selected, bad)
        matched = {int(q) for q in matches["queries"]}
        bad = [q for q in range(6) if q not in matched and not bool(hard_pass[q].detach())]
        priority = sorted(bad, key=lambda q:(state[q]["direct"], -state[q]["wait"], q))
        result = assign(inner, tail, matches, hard_pass, teachers, step, capacity, fixed_priority=priority)
        result.update(fair_priority=priority, exposure_before=copy.deepcopy(state),
                      scheduler="PER_SAMPLE_COMMITTED_OCCURRENCE_V2")
        self.pending.append((step, sid, list(result["selected"]), bad))
        return result

    @staticmethod
    def _advance(state, selected, bad):
        for q in range(6):
            state[q]["eligible"] += int(q in bad)
            state[q]["direct"] += int(q in selected)
            state[q]["wait"] = state[q]["wait"] + 1 if q in bad and q not in selected else 0

    def commit(self, step):
        require(step == self.last_step + 1 and all(p[0] == step for p in self.pending), "V2_COMMIT_SEQUENCE")
        receipt = copy.deepcopy(self.pending)
        for _, sid, selected, bad in self.pending:
            state = self.history.setdefault(sid, [dict(eligible=0, direct=0, wait=0) for _ in range(6)])
            self._advance(state, selected, bad)
        self.last_step = step
        self.pending.clear()
        return dict(step=step, occurrences=receipt, state_sha256=digest(self.state_dict()))

    def discard(self):
        self.pending.clear()

    def state_dict(self):
        require(not self.pending, "V2_PENDING_NOT_CHECKPOINTABLE")
        return copy.deepcopy(dict(schema="pg3_v2_fair_v1", last_step=self.last_step, history=self.history))

    def load_state_dict(self, value):
        require(not self.pending and value.get("schema") == "pg3_v2_fair_v1", "V2_FAIR_SCHEMA")
        require(type(value["last_step"]) is int and value["last_step"] >= -1, "V2_FAIR_STEP")
        for sid, rows in value["history"].items():
            require(isinstance(sid,str) and bool(sid) and len(rows)==6, "V2_FAIR_ROWS")
            for r in rows:
                require(set(r)=={"eligible","direct","wait"} and all(type(v) is int and v>=0 for v in r.values())
                        and r["direct"]<=r["eligible"] and r["wait"]<=r["eligible"], "V2_FAIR_VALUES")
        self.history=copy.deepcopy(value["history"]); self.last_step=value["last_step"]

    def summary(self):
        return self.state_dict()


def _at_progress_with_velocity(c, d, s):
    """Evaluation of position/velocity at detached normalized progress."""
    s = torch.as_tensor(s, dtype=d.dtype, device=d.device).detach().reshape(1)
    t = s * d.sum()
    ends = d.cumsum(0)
    seg = torch.searchsorted(ends.detach().contiguous(), t.detach().contiguous(), right=True).clamp_max(2)
    starts = torch.cat([d.new_zeros(1), ends[:-1]])
    local = t - starts[seg]
    exponents = torch.arange(6, dtype=c.dtype, device=c.device)
    pp = local[:, None] ** exponents
    vv = local[:, None] ** exponents[:-1]
    pos = (c[seg] * pp[:, None, :]).sum(-1)[0]
    vel = (c[seg, :, 1:] * exponents[None, None, 1:] * vv[:, None, :]).sum(-1)[0]
    return pos, vel


def section_side_proxy(c, d, target, scene, *, margin=0.03, scale=0.15):
    """Prefer correct *certified Teacher face*, without fixing entire Teacher tube.

    A detached student polynomial root chooses a local crossing branch. When
    unique, the first-order implicit-root correction keeps transverse gradients
    aligned with the spatial x-section; otherwise use Teacher progress as an
    explicitly labeled proxy. Neither option proves continuous collision safety.
    """
    losses, statuses = [], []
    for label in target.get("bottlenecks", []):
        side = label.get("side")
        if not label.get("valid") or side is None or label.get("event_status") != "UNIQUE_FORWARD_CROSSING":
            continue
        axis, sign = int(side[0]), int(side[1])
        if axis not in (1, 2) or sign not in (-1, 1):
            raise ValueError("PG3_V2_BAD_CERTIFIED_FACE")
        box = scene["obstacles"][int(label["box"])]
        plane_x = float(label["plane_x"])
        with torch.no_grad():
            crossing = crossing_events(c.detach().double().cpu().numpy(),
                                       d.detach().double().cpu().numpy(), plane_x)
        unique = (crossing["status"] == "UNIQUE_FORWARD_CROSSING" and len(crossing["events"]) == 1
                  and crossing["selected"]["vx"] > 1e-4
                  and 1e-6 < crossing["selected"]["time_s"] < float(d.detach().sum()) - 1e-6
                  and min(abs(crossing["selected"]["time_s"]-float(t)) for t in d.detach().cumsum(0)[:-1]) > 1e-6)
        s = (crossing["selected"]["time_s"] / float(d.detach().sum())) if unique else float(label["progress"])
        pos, vel = _at_progress_with_velocity(c, d, s)
        if unique and float(vel[0].detach()) > 1e-4:
            # Newton correction: implicit spatial-section gradient, local branch.
            dt = -(pos[0] - plane_x) / vel[0].detach()
            at_plane = pos + vel * dt
            statuses.append("LOCAL_IMPLICIT_SECTION_PROXY")
        else:
            at_plane = pos
            statuses.append("TEACHER_PROGRESS_FALLBACK")
        bound = float(box["max"][axis]) + .15 if sign > 0 else float(box["min"][axis]) - .15
        # Student should pass outside the certified box face by the extra margin.
        signed_gap = sign * (at_plane[axis] - bound)
        side_violation = F.relu((margin - signed_gap) / scale).square()
        # Fallback also needs to drive longitudinal progress to the correct x plane.
        axial = ((pos[0] - plane_x) / 1.0).square() if not unique else pos.sum() * 0
        losses.append(side_violation + .1 * axial)
    zero = (c.sum() + d.sum()) * 0
    return (torch.stack(losses).mean() if losses else zero,
            dict(count=len(losses), statuses=statuses, meaning="SECTION_FACE_PROXY_NOT_CERTIFICATE"))


def full_stopping_risk(positions, velocities, weights, fields, *,
                       braking_acceleration=4., clearance=.15, temperature=.05):
    """Uncapped stopping-distance training proxy, NOT execution permission."""
    from query_minco.training.losses.set_physical_risk import box_sdf
    require(braking_acceleration > 0 and temperature > 0 and clearance >= 0, "V2_STOP_CONSTANTS")
    require(positions.shape == velocities.shape and weights.shape == positions.shape[:-1]
            and len(fields)==len(positions) and positions.ndim==4 and positions.shape[1]==6, "V2_STOP_SHAPE")
    require(all(bool(torch.isfinite(x).all()) for x in (positions,velocities,weights))
            and bool((weights>=0).all()) and bool((weights.sum(-1)>0).all()), "V2_STOP_MASS")
    values = []
    for b, field in enumerate(fields):
        boxes = field.boxes.to(positions)
        if not len(boxes):
            if not field.row["empty_certified"]:
                raise ValueError("PG3_V2_UNKNOWN_EMPTY_GEOMETRY")
            values.append(positions[b].sum() * 0)
            continue
        require(not field.row["empty_certified"], "V2_STOP_EMPTY_CONTRADICTION")
        distance = box_sdf(positions[b], boxes)
        # Analytic normal on the active AABB branch, with corner derivatives.
        center=(boxes[:,0]+boxes[:,1])/2
        q=(positions[b].unsqueeze(-2)-center).abs()-(boxes[:,1]-boxes[:,0])/2
        delta=positions[b].unsqueeze(-2)-center
        outside=q.clamp_min(0)
        length=outside.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
        outer=outside/length*delta.sign()
        face=F.one_hot(q.argmax(-1),3).to(q)*delta.sign()
        normals=torch.where((q.max(-1).values>0)[...,None],outer,face)
        all_d=outside.norm(dim=-1)+q.max(-1).values.clamp_max(0)
        index=all_d.argmin(-1)
        normal=normals.gather(-2,index[...,None,None].expand(*index.shape,1,3)).squeeze(-2)
        closing = (-(velocities[b] * normal).sum(-1)).relu()
        required = closing.square() / (2. * braking_acceleration)
        violation = F.softplus((clearance + required - distance) / temperature)
        alpha = weights[b] / weights[b].sum(-1, keepdim=True)
        values.append((alpha * violation.square()).sum(-1).mean())
    return torch.stack(values).mean()


def query_collapse_diagnostic(trajectory, *, threshold_m=.30):
    """Trajectory-sample similarity; a proxy, NOT number of homotopy classes."""
    rows = []
    with torch.no_grad():
        for b in range(len(trajectory.coeffs)):
            paths = torch.stack([
                at_progress(trajectory.coeffs[b, q].detach(),
                            trajectory.durations[b, q].detach(), [.25, .50, .75])
                for q in range(6)
            ]).reshape(6, -1)
            pairwise = torch.pdist(paths, p=2) / (3. ** .5)
            rows.append(dict(min_pair_distance_m=float(pairwise.min()),
                             near_duplicate_pairs=int((pairwise < threshold_m).sum()),
                             pairs=15,
                             label="GEOMETRIC_PATH_SAMPLE_SIMILARITY_NOT_TOPOLOGY"))
    return rows


class FrozenSafeReference:
    """Preserve initially hard-safe candidate coordinates using an immutable model.

    Teacher and geometry are train/offline only. This is a trust-region penalty,
    not a safety guarantee; it must be rechecked with the original hard verifier.
    """

    def __init__(self, model, waypoint_tolerance=.08, velocity_tolerance=.30,
                 duration_tolerance=.08, *, expected_sha):
        require(state_sha(model)==expected_sha, "V2_REFERENCE_NOT_INITIAL_C1")
        self.identity=expected_sha
        self.inputs={}
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.cache = {}
        self.wtol = waypoint_tolerance
        self.vtol = velocity_tolerance
        self.ttol = duration_tolerance

    def _reference(self, sample, field):
        sid = sample.sample_id
        require(state_sha(self.model)==self.identity and not self.model.training
                and not any(p.requires_grad for p in self.model.parameters()), "V2_REFERENCE_MUTATED")
        key=digest(dict(model=self.identity, inputs=tensor_digest(sample.depth_input,sample.state),
                        scene=scene_from_field(field), empty=field.row["empty_certified"],
                        validator=sha(ROOT/"query_minco/evaluation/validation.py")))
        require(sid not in self.inputs or self.inputs[sid]==key, "V2_REFERENCE_SAMPLE_DRIFT")
        self.inputs[sid]=key
        if sid not in self.cache:
            from query_minco.trajectory import TrajectoryPostprocessor
            from query_minco.training.pg2_guided_v1.geometry import hard_labels
            with torch.no_grad():
                dp = sample.depth_input[None].to(next(self.model.parameters()))
                st = sample.state[None].to(dp)
                raw = self.model(depth_input=dp, state=st)
                pred = TrajectoryPostprocessor()(raw, st)
                valid = hard_labels(pred, [sample], [field])["passed"][0]
                p = pred.parameters
                self.cache[sid] = dict(
                    safe=valid.detach().cpu().clone(),
                    inner=p.inner_pts[0].detach().cpu().clone(),
                    tail=p.tail_pva[0].detach().cpu().clone(),
                    durations=p.durations[0].detach().cpu().clone())
        return self.cache[sid]

    def loss(self, z, samples, fields):
        param = z["output"].parameters
        per_sample, counts = [], []
        for b, (sample, field) in enumerate(zip(samples, fields)):
            ref = self._reference(sample, field)
            mask = ref["safe"].to(device=param.inner_pts.device)
            counts.append(int(mask.sum()))
            if not bool(mask.any()):
                continue
            way = (param.inner_pts[b] - ref["inner"].to(param.inner_pts)).norm(dim=-1)
            end = (param.tail_pva[b, :, 0] - ref["tail"].to(param.tail_pva)[..., 0, :]).norm(dim=-1)
            vel = (param.tail_pva[b, :, 1] - ref["tail"].to(param.tail_pva)[..., 1, :]).norm(dim=-1)
            dur = (param.durations[b] - ref["durations"].to(param.durations)).abs().sum(-1)
            per = (F.relu(way - self.wtol).square().mean(-1)
                   + F.relu(end - self.wtol).square()
                   + .1 * F.relu(vel - self.vtol).square()
                   + F.relu(dur - self.ttol).square())
            per_sample.append(per[mask])
        return (torch.cat(per_sample).mean() if per_sample else param.inner_pts.sum()*0), dict(initial_safe_count=sum(counts),
            reference_sha256=self.identity,
            meaning="FROZEN_HARD_SAFE_PARAMETER_TRUST_REGION_NOT_SAFETY_CERTIFICATE")
