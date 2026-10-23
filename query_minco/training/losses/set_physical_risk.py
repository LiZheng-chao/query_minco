"""Isolated RSPS prototype: SI units, B0 FLU, no trainer registration.
Hard min chooses first box/face at ties. abs(0) has Torch zero subgradient.
Discrete quadrature is not a continuous collision certificate.
"""
import math
import torch
import torch.nn.functional as F
from query_minco.trajectory.sampler_e1 import evaluate_piece


def _finite(x, name):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError(name + " must be a finite floating tensor")


def box_sdf(points, boxes):
    """points [...,3], constant boxes [N,2,3] -> occupied union SDF."""
    _finite(points, 'points'); _finite(boxes, 'boxes')
    if points.shape[-1:] != (3,) or boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not len(boxes):
        raise ValueError('requires points [...,3] and nonempty boxes [N,2,3]')
    if boxes.dtype != points.dtype or boxes.device != points.device or not bool((boxes[:, 1] > boxes[:, 0]).all()) or boxes.requires_grad:
        raise ValueError('box extent/dtype/device/constant contract')
    q = (points.unsqueeze(-2) - (boxes[:, 0] + boxes[:, 1]) / 2).abs() - (boxes[:, 1] - boxes[:, 0]) / 2
    d = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.max(-1).values.clamp_max(0)
    return d.min(-1).values


def segment_samples(trajectory, points_per_segment=64):
    """Endpoint-inclusive segment trapezoids, normalized by each query T."""
    c, dt = trajectory.coeffs, trajectory.durations
    _finite(c, 'coeffs'); _finite(dt, 'durations')
    if c.ndim != 5 or c.shape[1:] != (6, 3, 3, 6) or dt.shape != c.shape[:3]:
        raise ValueError('expected [B,6,3,3,6] and [B,6,3]')
    if c.dtype != dt.dtype or c.device != dt.device or not bool((dt > 0).all()):
        raise ValueError('invalid durations/dtype/device')
    if trajectory.solve_valid.shape != c.shape[:2] or not bool(trajectory.solve_valid.all()):
        raise ValueError('invalid MINCO solve')
    if type(points_per_segment) is not int or points_per_segment < 2:
        raise ValueError('points_per_segment >=2 required')
    u = torch.linspace(0, 1, points_per_segment, dtype=c.dtype, device=c.device)
    local = dt[..., None] * u
    p, v = (evaluate_piece(c, local, k).flatten(2, 3) for k in (0, 1))
    trap = torch.ones_like(u); trap[0] = .5; trap[-1] = .5
    weights = (dt[..., None] * trap / (points_per_segment - 1) / dt.sum(-1)[..., None, None]).flatten(2, 3)
    _finite(p, 'sampled positions'); _finite(v, 'sampled velocities')
    return p, v, weights


def physical_loss(positions, velocities, time_weights, boxes, *, geometry_bound,
                  radius=.12, margin=0., eps=.05, beta=0., v_ref=1., w_max=3.):
    """Return [B,6] loss and detached risk diagnostics.

    Explicit length-B geometry binding is mandatory, including empty fields.
    It is a caller assertion, NOT proof of provenance. No real-data use until
    the sidecar validator is qualified. Flight domain walls must be encoded
    explicitly as occupied slabs. No implicit floor, roof or frame transform.
    Risk = grad(SDF).v; detached, normalized to time-weighted mean one.
    """
    for x, n in [(positions, 'positions'), (velocities, 'velocities'), (time_weights, 'time_weights')]:
        _finite(x, n)
    if positions.ndim != 4 or positions.shape[1] != 6 or positions.shape[-1] != 3 or positions.shape[2] < 2:
        raise ValueError('positions must be [B,6,K,3], K>=2')
    if velocities.shape != positions.shape or time_weights.shape != positions.shape[:-1]:
        raise ValueError('sample shapes mismatch; broadcasting forbidden')
    if any(x.device != positions.device or x.dtype != positions.dtype for x in (velocities, time_weights)):
        raise ValueError('sample dtype/device mismatch')
    if not bool((time_weights >= 0).all()) or not bool((time_weights.sum(-1) > 0).all()):
        raise ValueError('positive quadrature mass required')
    if boxes is None or geometry_bound is None or len(boxes) != len(positions) or len(geometry_bound) != len(positions):
        raise ValueError('missing geometry binding')
    if any(type(flag) is not bool or not flag for flag in geometry_bound):
        raise ValueError('unbound geometry; STOP')
    if not all(math.isfinite(x) for x in [radius, margin, eps, beta, v_ref, w_max]) or min(radius, margin, beta) < 0 or min(eps, v_ref) <= 0 or w_max < 1:
        raise ValueError('invalid physical hyperparameters')
    alpha = time_weights / time_weights.sum(-1, keepdim=True)
    losses, diags, rates = [], [], []
    for b, field in enumerate(boxes):
        _finite(field, 'geometry')
        if field.ndim != 3 or field.shape[1:] != (2, 3) or field.device != positions.device or field.dtype != positions.dtype or field.requires_grad:
            raise ValueError('invalid geometry shape/dtype/device')
        p = positions[b]
        if not len(field):
            losses.append(p.sum((-1, -2)) * 0)
            diags.append(torch.ones_like(alpha[b])); rates.append(torch.zeros_like(alpha[b])); continue
        clearance = box_sdf(p, field) - radius - margin
        with torch.enable_grad():
            probe = p.detach().requires_grad_(True)
            normal = torch.autograd.grad(box_sdf(probe, field).sum(), probe)[0]
        rate = (normal * velocities[b].detach()).sum(-1).detach()
        risk = (1 + beta * F.relu(-rate / v_ref)).clamp(1, w_max).detach()
        risk = risk / (alpha[b].detach() * risk).sum(-1, keepdim=True)
        losses.append((alpha[b] * risk * eps * F.softplus(-clearance / eps)).sum(-1))
        diags.append(risk); rates.append(rate)
    loss = torch.stack(losses); _finite(loss, 'loss')
    rate = torch.stack(rates)
    return loss, {'risk_weights': torch.stack(diags).detach(), 'clearance_rate': rate,
                  'unnormalized_risk': (1 + beta * F.relu(-rate / v_ref)).clamp(1, w_max).detach()}


def reduce_set(per_query, matched=None, matched_strength=1., unmatched_strength=1.):
    """Count-corrected E3: equal strengths recover E2 for M1 and M4.

    Group-mean coefficients = a_m*M/6/Z, a_u*(6-M)/6/Z,
    Z=a_m*M/6+a_u*(6-M)/6. Total coefficient mass is fixed.
    """
    _finite(per_query, 'per_query')
    if per_query.ndim != 2 or per_query.shape[1] != 6 or not len(per_query):
        raise ValueError('expected [B,6]')
    if matched is None:
        return per_query.mean()
    if matched.shape != per_query.shape or matched.dtype != torch.bool or matched.device != per_query.device:
        raise ValueError('matched mask must be bool [B,6]')
    if not bool(((matched.sum(-1) == 1) | (matched.sum(-1) == 4)).all()):
        raise ValueError('only M1/M4 accepted')
    if not all(math.isfinite(x) and x > 0 for x in (matched_strength, unmatched_strength)):
        raise ValueError('positive finite group strengths required')
    weight = torch.where(matched, torch.full_like(per_query, matched_strength), torch.full_like(per_query, unmatched_strength))
    return ((per_query * weight).sum(-1) / weight.sum(-1)).mean()
