"""S0 fixed-time adapter; never mutates the historical parameter decoder."""
from dataclasses import replace
import torch
from query_minco.trajectory.parameter_decoder import PhysicalParameterDecoder


def _goal_ray_logits(raw, state, fixed_time_s):
    """Optional *diagnostic* logit-space bias; not an expert or learned trajectory.

    B0 forward-only legacy x bound is respected, so backward goals remain
    unrepresentable without a separate architecture ablation.
    """
    goal = state[:, 6:9].float()
    d = goal.norm(dim=-1, keepdim=True)
    direction = goal / d.clamp_min(1e-6)
    max_progress = min(2.0, 0.55 * fixed_time_s * 3.0)
    end = direction * d.clamp(max=max_progress)
    fractions = raw.new_tensor([1/3, 2/3, 1.0])[None, :, None]
    anchor = end[:, None, :] * fractions
    lower = raw.new_tensor([0.0, -4.0, -2.0]); upper = raw.new_tensor([10.0, 4.0, 2.0])
    anchor = torch.minimum(torch.maximum(anchor, lower + 0.05 * (upper - lower)), upper - 0.05 * (upper - lower))
    logits = torch.atanh((2 * (anchor - lower) / (upper - lower) - 1).clamp(-0.95, 0.95))
    result = raw.clone()
    result[..., :9] = raw[..., :9] + logits.reshape(len(raw), 1, 9)
    return result


class FixedTimeDecoder:
    def __init__(self, spec, *, num_candidates=6):
        self.spec = spec
        self.legacy = PhysicalParameterDecoder(num_candidates=num_candidates)
        self.num_candidates = num_candidates

    def __call__(self, raw, state):
        if raw.ndim != 3 or raw.shape[1:] != (self.num_candidates, 21):
            raise ValueError("fixed-time raw candidate count/Raw21 shape mismatch")
        if state.shape != (len(raw), 12):
            raise ValueError("state must be [B,12]")
        position_raw = _goal_ray_logits(raw, state, self.spec.total_time_s) if self.spec.position_anchor == "goal_ray" else raw
        parameters = self.legacy(position_raw, state)
        dt = torch.full_like(parameters.durations, self.spec.total_time_s / 3)
        total = dt.sum(-1)
        return replace(parameters, durations=dt, total_time=total,
                       diagnostics={**parameters.diagnostics,
                                    "time_sum_error_s": (total - self.spec.total_time_s).abs(),
                                    "parameter_valid": parameters.diagnostics["parameter_valid"]})
