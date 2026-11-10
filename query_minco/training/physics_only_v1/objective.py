"""Standalone, teacher-free S0 objective and collection gradients. NO TRAINING ENTRY."""
from dataclasses import dataclass
import math
import torch
from query_minco.trajectory.minco_adapter import MincoAdapter
from .decoder import FixedTimeDecoder
from .costs import four_costs
from .spec import PhysicsSpec


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    metrics: dict
    parameters: object
    trajectory: object


def aggregate_costs(costs, spec):
    """No reduction over Query before J. Soft-min and guard are aggregations."""
    c, f, gt, p, s = (costs[k] for k in ("C", "F", "G_task", "Progress", "S"))
    if not all(x.ndim == 2 and x.shape == c.shape and bool(torch.isfinite(x).all()) for x in (c, f, gt, p, s)) or c.shape[1] != 6:
        raise ValueError("costs must be finite [B,6]")
    violation = spec.wc*c + spec.wf*f + spec.wg*gt
    quality = spec.wg*spec.eta_progress*p + spec.wj*s
    J = violation + quality
    pi = torch.softmax(-J/spec.set_tau, dim=1)
    if spec.aggregation == "softmin":
        set_loss = -spec.set_tau * (torch.logsumexp(-J/spec.set_tau, dim=1) - math.log(6))
    else:
        set_loss = J.mean(-1)
        pi = torch.full_like(pi, 1/6)
    guard = (c + spec.guard_feas_ratio*f).mean(-1)
    total = (set_loss + spec.guard_weight*guard).mean()
    return total, {"violation": violation, "quality": quality, "J": J, "pi": pi,
                   "set_per_scene": set_loss, "guard_per_scene": guard,
                   "K_eff": (-(pi*pi.clamp_min(1e-30).log()).sum(-1)).exp().clamp(1., 6.)}


class PhysicsOnlyObjective:
    """S0 synthetic-only path. No teacher, no optimizer, no long training."""
    def __init__(self, spec=None):
        self.spec = spec or PhysicsSpec()
        self.decoder = FixedTimeDecoder(self.spec)
        self.adapter = MincoAdapter()

    def __call__(self, raw, state, geometry, goal_enabled=None):
        if not isinstance(raw, torch.Tensor) or not isinstance(state, torch.Tensor):
            raise TypeError("raw/state must be tensors")
        if goal_enabled is None:
            goal_enabled = torch.ones(len(state), device=state.device, dtype=torch.bool)
        p = self.decoder(raw, state)
        trajectory = self.adapter(p.head_pva, p.tail_pva, p.inner_pts, p.durations)
        terms = four_costs(trajectory, geometry, state[:, 6:9].float(), goal_enabled, self.spec)
        loss, set_diag = aggregate_costs(terms, self.spec)
        return ObjectiveResult(loss, {**terms, **set_diag}, p, trajectory)
