"""Explicit V/Q generation objectives. No optimizer, ranking or active diversity."""
from dataclasses import dataclass
import math
import torch

from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from .goal_v12 import finite_tensor, validate_goal_batch
from .costs_v12 import trajectory_terms
from .spec import PhysicsV12Spec


def aggregate_terms(terms, spec):
    names = ("C_viol", "F", "G_task_train", "C_shape", "P", "S")
    tensors = [terms[k] for k in names]
    first = tensors[0]
    for x, name in zip(tensors, names):
        finite_tensor(x, name)
        if x.ndim != 2 or x.shape != first.shape or x.shape[1] not in (1, 6) or len(x) == 0:
            raise ValueError("basic terms require [B,K], K=1 or 6")
        if x.dtype != first.dtype or x.device != first.device:
            raise ValueError("term type/device mismatch")
    candidates = first.shape[1]
    scaled = {k: x * w for k, x, w in zip(
        names, tensors, (spec.w_cv, spec.w_f, spec.w_g, spec.w_cc, spec.w_p, spec.w_s))}
    violation = scaled["C_viol"] + scaled["F"] + scaled["G_task_train"]
    quality = scaled["C_shape"] + scaled["P"] + scaled["S"]
    J = violation + quality
    finite_tensor(J, "J")
    soft_pi = torch.softmax(-J / spec.set_tau, -1)
    softmin = -spec.set_tau * (torch.logsumexp(-J / spec.set_tau, -1) - math.log(candidates))
    guard = violation.mean(-1)
    if spec.phase == "A":
        scene_loss = guard + spec.beta_A * quality.mean(-1)
        pi = torch.full_like(J, 1/candidates)
    elif spec.aggregation == "softmin_guard":
        scene_loss = softmin + spec.lambda_all * guard
        pi = soft_pi
    elif spec.aggregation == "mean_guard":
        scene_loss = J.mean(-1) + spec.lambda_all * guard
        pi = torch.full_like(J, 1/candidates)
    else:
        scene_loss = J.mean(-1)
        pi = torch.full_like(J, 1/candidates)
    entropy = -(pi * pi.clamp_min(torch.finfo(pi.dtype).tiny).log()).sum(-1)
    loss = scene_loss.mean()
    finite_tensor(loss, "set loss")
    return loss, dict(V=violation, Q=quality, J=J, pi=pi, entropy=entropy,
                      max_pi=pi.amax(-1), K_eff=entropy.exp().clamp(1., float(candidates)),
                      softmin_per_scene=softmin, all_guard_per_scene=guard,
                      objective_per_scene=scene_loss, eta_P_times_P=spec.w_p * terms["P"],
                      G_total=terms["G_task_train"] + spec.w_p * terms["P"],
                      weighted_goal_total=spec.w_g * terms["G_task_train"] + spec.w_p * terms["P"],
                      **{"weighted_" + k: v for k, v in scaled.items()})


@dataclass
class V12Result:
    loss: torch.Tensor
    metrics: dict
    parameters: object
    trajectory: object


class PhysicsV12Objective:
    """Synthetic CPU FP32 MINCO interface; inputs never include privileged model features."""
    def __init__(self, spec=None, *, num_candidates=6):
        self.spec = spec or PhysicsV12Spec()
        self.decoder = FixedTimeDecoder(self.spec, num_candidates=num_candidates)
        self.num_candidates = num_candidates
        self.adapter = MincoAdapter(num_candidates=num_candidates)

    def __call__(self, raw, state, geometry, *, goal_enabled, task_types):
        finite_tensor(raw, "raw")
        finite_tensor(state, "state")
        if raw.device.type != "cpu" or state.device != raw.device or raw.dtype != torch.float32 or state.dtype != raw.dtype:
            raise ValueError("this acceptance interface requires CPU FP32")
        if raw.ndim != 3 or raw.shape[1:] != (self.num_candidates, 21) or state.shape != (len(raw), 12):
            raise ValueError("raw/state shape mismatch")
        validate_goal_batch(state[:, 6:9], goal_enabled, task_types)
        if not torch.allclose(state[:, 9:12].norm(dim=-1), torch.ones_like(state[:, 0]), atol=1e-4, rtol=0):
            raise ValueError("gravity direction must be unit length")
        p = self.decoder(raw, state)
        tr = self.adapter(p.head_pva, p.tail_pva, p.inner_pts, p.durations)
        terms = trajectory_terms(tr, geometry, state[:, 6:9], goal_enabled, task_types, self.spec)
        loss, diagnostics = aggregate_terms(terms, self.spec)
        return V12Result(loss, {**terms, **diagnostics}, p, tr)
