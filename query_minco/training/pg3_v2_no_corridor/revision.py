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


# V2.1 APIs are explicitly separate: the historical V2 functions above are frozen.
def section_events_v21(c, d, target, scene):
    """Train only on valid spatial crossings; no teacher-progress gradient fallback."""
    zero=(c.sum()+d.sum())*0
    require(c.shape==(3,3,6) and d.shape==(3,) and bool((d>0).all())
            and bool(torch.isfinite(c).all()) and bool(torch.isfinite(d).all()),"V21_SECTION_CURVE")
    values=[];events=[];skipped=[]
    trusted=target.get("certified",False) and target.get("representable",False)
    for i,label in enumerate(target.get("bottlenecks",[])):
        reason=None;side=label.get("side")
        if not trusted:reason="UNTRUSTED_TEACHER"
        elif not label.get("valid"):reason="INVALID_LABEL"
        elif side is None or len(side)!=2 or side[0] not in (1,2) or side[1] not in (-1,1):reason="NO_CERTIFIED_SIDE"
        elif label.get("event_status")!="UNIQUE_FORWARD_CROSSING":reason="TEACHER_CROSSING_AMBIGUOUS"
        if reason:
            skipped.append(dict(event=i,box=label.get("box"),reason=reason));continue
        box=scene["obstacles"][int(label["box"])];axis,sign=side
        x=float(label["plane_x"])
        crossing=crossing_events(c.detach().double().cpu().numpy(),d.detach().double().cpu().numpy(),x)
        e=crossing["selected"]
        if crossing["status"]!="UNIQUE_FORWARD_CROSSING" or len(crossing["events"])!=1:reason="STUDENT_"+crossing["status"]
        elif e["vx"]<=1e-4:reason="SMALL_VX"
        elif not 1e-6<e["time_s"]<float(d.detach().sum())-1e-6:reason="ENDPOINT_CROSSING"
        elif any(abs(e["time_s"]-float(t))<=1e-6 for t in d.detach().cumsum(0)[:-1]):reason="SEGMENT_JOIN"
        if reason:
            skipped.append(dict(event=i,box=label["box"],reason=reason));continue
        pos,vel=_at_progress_with_velocity(c,d,e["time_s"]/float(d.detach().sum()))
        if float(vel[0].detach())<=1e-4:
            skipped.append(dict(event=i,box=label["box"],reason="SMALL_EVALUATED_VX"));continue
        point=pos+vel*(-(pos[0]-x)/vel[0].detach())
        bound=float(box["max"][axis])+.15 if sign>0 else float(box["min"][axis])-.15
        gap=sign*(point[axis]-bound)
        values.append(F.relu(-gap/.15).square())
        events.append(dict(event=i,box=label["box"],axis=axis,sign=sign,plane_x=x,
                           time_s=e["time_s"],vx=e["vx"],gap_m=float(gap.detach()),
                           deficit_m=float((-gap).relu().detach()),source="LOCAL_IMPLICIT_SECTION_PROXY"))
    return (torch.stack(values).mean() if values else zero),dict(valid=bool(values),count=len(values),
        event_denominator=len(values),discarded_count=len(skipped),events=events,skipped=skipped,
        delta_extra_m=0.,sigma_section_m=.15,d_safe_m=.15)


def full_stopping_risk_v21(positions,velocities,weights,fields):
    """[tau*softplus((.15+vc^2/8-SDF)/tau)/d_ref]^2; tau=.05m,d_ref=.15m."""
    return (.05/.15)**2*full_stopping_risk(positions,velocities,weights,fields,
        braking_acceleration=4.,clearance=.15,temperature=.05)


class UniqueSampleFairAssignment:
    """One committed exposure per successful update and unique sample ID."""
    SCHEMA="pg3_v21_unique_exposure_v1"
    def __init__(self):
        self.history={};self.last_step=-1;self.pending={};self.pending_step=None
        self.batch_ids=[];self.seen={}

    def begin_step(self,step,batch_sample_ids):
        require(type(step) is int and step==self.last_step+1 and self.pending_step is None,"V21_BEGIN_SEQUENCE")
        require(bool(batch_sample_ids) and all(isinstance(s,str) and s for s in batch_sample_ids),"V21_BATCH_IDS")
        self.pending_step=step;self.batch_ids=list(batch_sample_ids);self.seen={}

    def validate_duplicate(self,sid,matching_digest,assignment_digest):
        require(sid in self.pending and self.pending[sid]["matching_digest"]==matching_digest
                and self.pending[sid]["assignment_digest"]==assignment_digest,"DUPLICATE_SAMPLE_CONFLICT")

    def choose_once(self,sid,inner,tail,matches,hard_pass,teachers,*,authenticated_digest,capacity=2):
        require(self.pending_step is not None and sid in self.batch_ids and capacity==2,"V21_CHOOSE_TRANSACTION")
        require(isinstance(authenticated_digest,str) and len(authenticated_digest)==64,"V21_AUTHENTICATED_INPUT_DIGEST")
        fingerprint=digest(dict(authenticated=authenticated_digest,inner=tensor_digest(inner,tail),
            matches={k:[int(x) for x in matches[k]] for k in ("queries","teachers")},
            hard=hard_pass.detach().tolist(),teachers=teachers))
        count=self.seen.get(sid,0)+1
        require(count<=self.batch_ids.count(sid),"V21_TOO_MANY_OCCURRENCES")
        if sid in self.pending:
            p=self.pending[sid];self.validate_duplicate(sid,fingerprint,digest(p["decision"]))
            self.seen[sid]=count;return copy.deepcopy(p["decision"])
        state=copy.deepcopy(self.history.get(sid,[dict(eligible=0,direct=0,wait=0) for _ in range(6)]))
        matched={int(q) for q in matches["queries"]}
        bad=[q for q in range(6) if q not in matched and not bool(hard_pass[q].detach())]
        priority=sorted(bad,key=lambda q:(state[q]["direct"],-state[q]["wait"],q))
        result=assign(inner,tail,matches,hard_pass,teachers,self.pending_step,2,fixed_priority=priority)
        result.update(fair_priority=priority,exposure_before=state,scheduler=self.SCHEMA)
        self.pending[sid]=dict(matching_digest=fingerprint,assignment_digest=digest(result),decision=copy.deepcopy(result),eligible=bad)
        self.seen[sid]=count
        return result

    def commit_after_optimizer_step(self,step,*,optimizer_succeeded=False,finite=False,scores_unchanged=False):
        require(optimizer_succeeded is True and finite is True and scores_unchanged is True,"V21_OPTIMIZER_NOT_CONFIRMED")
        require(self.pending_step==step==self.last_step+1,"V21_COMMIT_SEQUENCE")
        require(all(self.seen.get(sid,0)==self.batch_ids.count(sid) for sid in set(self.batch_ids)),"V21_INCOMPLETE_BATCH")
        decisions=copy.deepcopy(self.pending)
        for sid,p in self.pending.items():
            state=self.history.setdefault(sid,[dict(eligible=0,direct=0,wait=0) for _ in range(6)])
            PerSampleFairAssignment._advance(state,p["decision"]["selected"],p["eligible"])
        receipt=dict(step=step,batch_occurrences=len(self.batch_ids),unique_samples=len(self.pending),decisions=decisions)
        self.last_step=step;self.pending={};self.pending_step=None;self.batch_ids=[];self.seen={}
        receipt["state_sha256"]=digest(self.state_dict());return receipt

    def rollback(self,step):
        require(self.pending_step is None or self.pending_step==step,"V21_ROLLBACK_STEP")
        self.pending={};self.pending_step=None;self.batch_ids=[];self.seen={}

    def state_dict(self):
        require(self.pending_step is None and not self.pending,"V21_PENDING_NOT_CHECKPOINTABLE")
        return copy.deepcopy(dict(schema=self.SCHEMA,last_step=self.last_step,history=self.history))

    def load_state_dict(self,state):
        require(self.pending_step is None and state.get("schema")==self.SCHEMA,"V21_STATE_SCHEMA")
        # Reuse the historical value validation without accepting its semantics/schema.
        validator=PerSampleFairAssignment()
        validator.load_state_dict(dict(state,schema="pg3_v2_fair_v1"))
        require(all(r["eligible"]<=state["last_step"]+1 for rows in state["history"].values() for r in rows),"V21_STATE_EXPOSURE_BOUND")
        self.history=copy.deepcopy(state["history"]);self.last_step=state["last_step"]


def save_v21_checkpoint(path,model,optimizer,assignment,spec):
    """Exclusive atomic publication binds model, optimizer, exposure, RNG and spec."""
    import os,uuid
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=assignment.state_dict()  # rejects pending before creating any file
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex)
    with temporary.open("xb") as f:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),exposure=state,
            model_sha256=state_sha(model),spec=spec,spec_sha256=digest(spec),torch_rng=torch.get_rng_state()),f)
        f.flush();os.fsync(f.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()


def load_v21_checkpoint(path,model,optimizer,assignment,spec):
    blob=torch.load(path,map_location="cpu",weights_only=True)
    require(blob["spec"]==spec and blob["spec_sha256"]==digest(spec),"V21_RESUME_IDENTITY")
    model.load_state_dict(blob["model"],strict=True)
    require(state_sha(model)==blob["model_sha256"],"V21_RESUME_MODEL")
    optimizer.load_state_dict(blob["optimizer"]);assignment.load_state_dict(blob["exposure"])
    torch.set_rng_state(blob["torch_rng"])
    return blob


def optimizer_step_v21(model,optimizer,assignment,step,score_reference):
    """Commit only after all checks and successful step; exceptions reject the run."""
    from query_minco.training.initialization import rows_unchanged
    try:
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"V21_NONFINITE_GRADIENT")
        require(rows_unchanged(model,score_reference),"V21_PRESTEP_SCORE_DRIFT")
        for name,p in model.named_parameters():
            if name.startswith("trajectory_head.") and p.shape[0]==21 and p.grad is not None:
                require(not torch.count_nonzero(p.grad[19:21]),"V21_SCORE_GRADIENT")
        optimizer.step()
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        scores=rows_unchanged(model,score_reference)
        require(finite and scores,"V21_POSTSTEP_PARAMETER_GUARD")
        return assignment.commit_after_optimizer_step(step,optimizer_succeeded=True,finite=finite,scores_unchanged=scores)
    except BaseException:
        assignment.rollback(step)
        # Parameters/optimizer are NOT claimed rolled back. Resume requires a validated checkpoint.
        raise


class FrozenSafeReferenceV21(FrozenSafeReference):
    """Same Keep mathematics and masks; adds per-query drift/activation monitoring."""
    def loss(self,z,samples,fields):
        value,info=super().loss(z,samples,fields);p=z["output"].parameters;rows=[]
        with torch.no_grad():
            for b,(sample,field) in enumerate(zip(samples,fields)):
                ref=self._reference(sample,field)
                way=(p.inner_pts[b]-ref["inner"].to(p.inner_pts)).norm(dim=-1)
                end=(p.tail_pva[b,:,0]-ref["tail"].to(p.tail_pva)[:,0]).norm(dim=-1)
                vel=(p.tail_pva[b,:,1]-ref["tail"].to(p.tail_pva)[:,1]).norm(dim=-1)
                times=(p.durations[b]-ref["durations"].to(p.durations)).abs().sum(-1)
                for q in range(6):
                    safe=bool(ref["safe"][q]);active=safe and (bool((way[q]>self.wtol).any()) or float(end[q])>self.wtol or float(vel[q])>self.vtol or float(times[q])>self.ttol)
                    rows.append(dict(sample_id=sample.sample_id,query=q,initial_safe=safe,active=active,
                        waypoint_m=way[q].tolist(),endpoint_m=float(end[q]),velocity_mps=float(vel[q]),duration_l1_s=float(times[q])))
        return value,dict(info,drift=rows,active_initial_safe_queries=sum(r["active"] for r in rows))
