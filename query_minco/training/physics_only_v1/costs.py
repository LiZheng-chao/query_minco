"""Four differentiable physical proxies, split into violation and quality signals.

All tensors B0-FLU / SI. These discrete costs are NOT flight certificates.
Geometry provided here is only for controlled S0 synthetic tests. Production
training must use a provenance-validated new loader, not this S0 adapter.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.training.pg1.losses import analytic_samples
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss


@dataclass(frozen=True)
class SyntheticGeometry:
    """S0 test-only geometry, intentionally NOT a production attestation."""
    boxes: torch.Tensor  # [N,2,3]
    domain: torch.Tensor  # [2,3] center-space limits
    empty_certified: bool  # explicit certification only for synthetic scenes
    tag: str = "SYNTHETIC_TEST_ONLY"
    frame: str = "fixed B0 FLU"
    units: str = "m_s"


def _finite(x, name):
    if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be finite floating tensor")


def check_synthetic_geometry(geometry, batch, *, dtype, device):
    if not isinstance(geometry, (tuple, list)) or len(geometry) != batch:
        raise ValueError("geometry must be a scene-bound list, one per batch item")
    for record in geometry:
        if not isinstance(record, SyntheticGeometry) or record.tag != "SYNTHETIC_TEST_ONLY" or record.frame != "fixed B0 FLU" or record.units != "m_s":
            raise ValueError("S0 only accepts explicit synthetic test geometry; production geometry is NOT implemented")
        for t in (record.boxes, record.domain):
            _finite(t, "geometry")
            if t.device != device or t.dtype != dtype or t.requires_grad:
                raise ValueError("geometry tensor type/device/gradient mismatch")
        if record.boxes.ndim != 3 or record.boxes.shape[1:] != (2, 3):
            raise ValueError("boxes must be [N,2,3]")
        if record.domain.shape != (2, 3) or not bool((record.domain[1] > record.domain[0]).all()):
            raise ValueError("bad center domain")
        if type(record.empty_certified) is not bool:
            raise ValueError("empty_certified requires bool")
        if len(record.boxes) == 0 and not record.empty_certified:
            raise ValueError("unproven empty geometry: FAIL CLOSED")
        if len(record.boxes) and record.empty_certified:
            raise ValueError("boxes/empty certification contradiction")
        if len(record.boxes) and not bool((record.boxes[:, 1] > record.boxes[:, 0]).all()):
            raise ValueError("invalid box extents")


def collision_cost(ana, geometry, spec):
    p, w = ana["p"], ana["weights"]
    check_synthetic_geometry(geometry, len(p), dtype=p.dtype, device=p.device)
    if w.shape != p.shape[:-1] or not bool((w >= 0).all()) or not bool((w.sum(-1) > 0).all()):
        raise ValueError("bad physical quadrature weights")
    alpha = w / w.sum(-1, keepdim=True)
    out, minimum_sdf = [], []
    for b, g in enumerate(geometry):
        if not len(g.boxes):
            out.append(p[b, :, 0, 0] * 0)
            minimum_sdf.append(p[b, :, 0, 0] * 0)  # undefined for certified-empty; see has_obstacles
            continue
        d = box_sdf(p[b], g.boxes)
        risk = (spec.collision_epsilon_m / (spec.radius_m + spec.clearance_m)) * F.softplus(
            (spec.radius_m + spec.clearance_m - d) / spec.collision_epsilon_m)
        log_a = torch.where(alpha[b] > 0, alpha[b].clamp_min(torch.finfo(alpha.dtype).tiny).log(),
                            torch.full_like(alpha[b], -float('inf')))
        out.append(spec.collision_tau * torch.logsumexp(risk / spec.collision_tau + log_a, dim=-1))
        minimum_sdf.append(d.min(-1).values.detach())
    return torch.stack(out), torch.stack(minimum_sdf)


def goal_cost(endpoint, terminal_velocity, goal, enabled, spec):
    """Smooth training surrogate. Official hard near/far pass remains UNCHANGED.

    In contrast to old loss, far angle is removed (redundant at zero violation)
    and the near distance penalty is scaled by target distance rather than 0.15m.
    Compact cubic transition avoids a hard 3.4m step. This is a TRAINING CHANGE.
    """
    for x, name in ((endpoint, "endpoint"), (terminal_velocity, "velocity"), (goal, "goal")):
        _finite(x, name)
    if endpoint.ndim != 3 or endpoint.shape[1:] != (6, 3) or terminal_velocity.shape != endpoint.shape or goal.shape != (len(endpoint), 3):
        raise ValueError("goal tensor shape mismatch")
    if goal.dtype != endpoint.dtype or goal.device != endpoint.device or terminal_velocity.dtype != endpoint.dtype or terminal_velocity.device != endpoint.device:
        raise ValueError("goal tensor type mismatch")
    if enabled.shape != (len(goal),) or enabled.dtype != torch.bool or enabled.device != goal.device:
        raise ValueError("goal enable mask [B] bool required")
    distance = goal.norm(dim=-1)
    unit = goal / distance[:, None].clamp_min(1.0)
    err = endpoint - goal[:, None]
    remaining = err.norm(dim=-1)
    speed = terminal_velocity.norm(dim=-1)
    scale = distance.clamp_min(1.0)
    near = ((remaining - 0.15).clamp_min(0) / scale[:, None]).square() + ((speed - 0.25).clamp_min(0) / 0.25).square()
    h = (endpoint * unit[:, None]).sum(-1)
    reduction = distance[:, None] - remaining
    lateral = (endpoint - h[..., None] * unit[:, None]).norm(dim=-1)
    far = (1.0 - reduction).clamp_min(0).square() + (1.0 - h).clamp_min(0).square() + ((lateral - 0.8).clamp_min(0) / 0.8).square()
    low = spec.near_threshold_m - spec.transition_half_width_m
    high = spec.near_threshold_m + spec.transition_half_width_m
    u = ((distance - low) / (high - low)).clamp(0.0, 1.0)
    blend = u.square() * (3 - 2 * u)
    task = (1 - blend[:, None]) * near + blend[:, None] * far
    progress = (torch.sqrt(err.square().sum(-1) + .05**2) - .05) / scale[:, None]
    mask = enabled.to(endpoint.dtype)[:, None]
    return task * mask, progress * mask, {"near_task": near.detach(), "far_task": far.detach(), "blend": blend.detach()}


def feasibility_cost(ana, geometry, spec):
    p, w = ana["p"], ana["weights"]
    alpha = w / w.sum(-1, keepdim=True)
    terms = {}
    for field, limit in (("v", spec.v_max_mps), ("a", spec.a_max_mps2), ("jerk", spec.j_max_mps3)):
        terms[field] = (alpha * ((ana[field].norm(dim=-1) / limit - 1).clamp_min(0).square())).sum(-1)
    out_domain = []
    for b, g in enumerate(geometry):
        lo = g.domain[0].reshape(1, 1, 3)
        hi = g.domain[1].reshape(1, 1, 3)
        excursion = ((lo - p[b]).clamp_min(0)/(hi-lo)).square() + ((p[b] - hi).clamp_min(0)/(hi-lo)).square()
        out_domain.append((alpha[b] * excursion.sum(-1)).sum(-1))
    domain = torch.stack(out_domain)
    peaks = {name + "_peak_ratio": ana[field].norm(dim=-1).amax(-1) / limit
             for name, field, limit in (("speed", "v", spec.v_max_mps), ("acceleration", "a", spec.a_max_mps2), ("jerk", "jerk", spec.j_max_mps3))}
    peaks["domain_peak_m"] = torch.stack([
        torch.maximum((g.domain[0] - p[b]).clamp_min(0), (p[b] - g.domain[1]).clamp_min(0)).amax((-1, -2))
        for b, g in enumerate(geometry)])
    return terms["v"] + terms["a"] + terms["jerk"] + spec.eta_domain * domain, {**terms, "domain": domain, **peaks}


def four_costs(trajectory, geometry, goal, enabled, spec):
    ana = analytic_samples(trajectory, spec.points_per_segment)
    c, min_sdf = collision_cost(ana, geometry, spec)
    gt, prog, gdiag = goal_cost(ana["endpoint"], ana["terminal_velocity"], goal, enabled, spec)
    f, fdiag = feasibility_cost(ana, geometry, spec)
    quality = dynamic_quality_loss(trajectory, j_ref=spec.jerk_ref_mps3, time_ref=spec.time_ref_s,
                                   enabled=("jerk",))
    smooth = quality["terms"]["jerk"]
    has_obstacles = torch.tensor([bool(len(x.boxes)) for x in geometry], device=goal.device, dtype=torch.bool)
    return dict(C=c, F=f, G_task=gt, Progress=prog, S=smooth, min_sdf=min_sdf, has_obstacles=has_obstacles,
                **{"F_" + k: v for k, v in fdiag.items()}, **{"G_" + k: v for k, v in gdiag.items()})
