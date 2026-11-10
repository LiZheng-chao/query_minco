"""Collision violation/shape, physical-domain feasibility and exact jerk quality.

Geometry remains explicit SYNTHETIC_TEST_ONLY. Sampling is never certification.
"""
import torch
import torch.nn.functional as F

from query_minco.training.physics_only_v1.costs import check_synthetic_geometry
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.training.pg1.losses import analytic_samples
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss, weighted_lse
from .goal_v12 import finite_tensor, phi2, goal_terms


def sample_weights(positions, weights):
    finite_tensor(positions, "positions")
    finite_tensor(weights, "physical-time weights")
    if (positions.ndim != 4 or positions.shape[1] not in (1, 6) or positions.shape[-1] != 3 or
            not len(positions) or positions.shape[2] < 2 or weights.shape != positions.shape[:-1] or
            weights.dtype != positions.dtype or weights.device != positions.device):
        raise ValueError("sample shape/type/device mismatch")
    if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(-1), torch.ones_like(weights[..., 0]), atol=2e-6, rtol=0):
        raise ValueError("physical-time weights must have unit mass")
    return weights / weights.sum(-1, keepdim=True)


def collision_terms(positions, weights, geometry, spec):
    alpha = sample_weights(positions, weights)
    check_synthetic_geometry(geometry, len(positions), dtype=positions.dtype, device=positions.device)
    cv, cc, minimum, peak = [], [], [], []
    safe = spec.radius_m + spec.clearance_m
    for b, scene in enumerate(geometry):
        zero = positions[b].sum((-1, -2)) * 0
        if not len(scene.boxes):
            cv.append(zero); cc.append(zero); minimum.append(zero); peak.append(zero)
            continue
        distance = box_sdf(positions[b], scene.boxes)
        violation = (safe - distance).clamp_min(0) / safe
        risk = (spec.collision_epsilon_m / safe) * F.softplus((safe - distance) / spec.collision_epsilon_m)
        cv.append((alpha[b] * violation.square()).sum(-1))
        cc.append(weighted_lse(risk[None], alpha[b][None], spec.collision_tau)[0])
        minimum.append(distance.amin(-1))
        peak.append((safe - distance).clamp_min(0).amax(-1))
    return dict(C_viol=torch.stack(cv), C_shape=torch.stack(cc), min_sdf_m=torch.stack(minimum),
                clearance_violation_peak_m=torch.stack(peak),
                has_obstacles=torch.tensor([len(g.boxes) > 0 for g in geometry],
                                           device=positions.device)[:, None].expand(-1, positions.shape[1]))


def feasibility_terms(ana, geometry, spec):
    p = ana["p"]
    alpha = sample_weights(p, ana["weights"])
    check_synthetic_geometry(geometry, len(p), dtype=p.dtype, device=p.device)
    terms, peaks = {}, {}
    for field, name, limit in (("v", "speed", 3.), ("a", "acceleration", 8.), ("jerk", "jerk", 100.)):
        x = ana[field]
        finite_tensor(x, field)
        if x.shape != p.shape or x.dtype != p.dtype or x.device != p.device:
            raise ValueError("derivative sample shape/type/device mismatch")
        norms = x.norm(dim=-1)
        terms["F_" + name] = (alpha * (norms / limit - 1).clamp_min(0).square()).sum(-1)
        peaks[name + "_sample_peak"] = norms.amax(-1)
    domains, excursions = [], []
    for b, g in enumerate(geometry):
        low = (g.domain[0] - p[b]).clamp_min(0)
        high = (p[b] - g.domain[1]).clamp_min(0)
        if spec.domain_mode == "s0_domain_width_ablation":
            local = (low / (g.domain[1] - g.domain[0])).square()
            local = local + (high / (g.domain[1] - g.domain[0])).square()
        else:
            local = phi2(low / spec.domain_ref_m) + phi2(high / spec.domain_ref_m)
        domains.append((alpha[b] * local.sum(-1)).sum(-1))
        excursions.append(torch.maximum(low, high).amax((-1, -2)))
    domain = torch.stack(domains)
    dyn = sum(terms.values())
    return dict(F=dyn + spec.eta_domain * domain, F_dyn=dyn, F_domain=domain,
                domain_sample_peak_m=torch.stack(excursions), **terms, **peaks)


def smoothness_terms(trajectory, spec):
    result = dynamic_quality_loss(trajectory, j_ref=spec.jerk_ref_mps3,
                                  time_ref=spec.time_ref_s, enabled=("jerk",))
    return dict(S=result["terms"]["jerk"],
                acceleration_energy_si=result["integrals_si"]["acceleration"],
                jerk_energy_si=result["integrals_si"]["jerk"],
                acceleration_energy_normalized=result["unmasked_terms"]["acceleration"])


def trajectory_terms(trajectory, geometry, goals, enabled, task_types, spec):
    if not torch.allclose(trajectory.durations, torch.full_like(trajectory.durations, 2/3),
                          atol=1e-7, rtol=0):
        raise ValueError("V1.2 requires fixed equal durations")
    ana = analytic_samples(trajectory, spec.points_per_segment)
    return {**collision_terms(ana["p"], ana["weights"], geometry, spec),
            **feasibility_terms(ana, geometry, spec),
            **smoothness_terms(trajectory, spec),
            **goal_terms(ana["endpoint"], ana["terminal_velocity"], goals, enabled, task_types)}
