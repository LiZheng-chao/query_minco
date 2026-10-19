"""PVA rows are P/V/A, columns xyz. No safety certification is represented."""
from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class PhysicalParameters:
    head_pva: torch.Tensor
    tail_pva: torch.Tensor
    inner_pts: torch.Tensor
    durations: torch.Tensor
    total_time: torch.Tensor
    score: torch.Tensor
    feas_logit: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class MincoTrajectory:
    coeffs: torch.Tensor
    durations: torch.Tensor
    solve_valid: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class TrajectorySamples:
    times: torch.Tensor
    p: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    jerk: torch.Tensor
    segment_indices: torch.Tensor
    local_times: torch.Tensor


@dataclass
class TrajectoryOutput:
    parameters: PhysicalParameters
    trajectory: MincoTrajectory
    samples: TrajectorySamples
