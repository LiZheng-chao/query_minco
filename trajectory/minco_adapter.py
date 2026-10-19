"""Read-only reuse of original MincoS3NU, with per-call FP32 solver state."""
import torch
from YOPO.minco.minco import MincoS3NU
from ._validation import tensor, matching
from .sampler import evaluate_piece
from .types import MincoTrajectory


class NumericalSolveError(RuntimeError):
    """No usable trajectory is returned; diagnostics describe numerical failure."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# Numerical residual limits in metres / seconds**derivative, not feasibility limits.
ABS_TOLERANCES = (0.002, 0.01, 0.05, 0.25, 2.0)
REL_TOLERANCE = 1e-4


def constraint_residuals(coeffs, durations, head, tail, inner):
    """Return per-candidate max absolute residual and acceptance at each derivative."""
    zero = torch.zeros_like(durations[..., 0, None])
    residuals, valid = {}, torch.ones(durations.shape[:2], dtype=torch.bool, device=durations.device)
    for d, atol in enumerate(ABS_TOLERANCES):
        left, right = [], []
        if d < 3:
            left.extend((evaluate_piece(coeffs[..., 0, :, :], zero, d).squeeze(-2),
                         evaluate_piece(coeffs[..., 2, :, :], durations[..., 2, None], d).squeeze(-2)))
            right.extend((head[..., d, :], tail[..., d, :]))
        for i in range(2):
            a = evaluate_piece(coeffs[..., i, :, :], durations[..., i, None], d).squeeze(-2)
            b = evaluate_piece(coeffs[..., i + 1, :, :], zero, d).squeeze(-2)
            left.append(a); right.append(b)
            if d == 0:
                left.append(a); right.append(inner[..., i, :])
        lhs, rhs = torch.stack(left, dim=-2), torch.stack(right, dim=-2)
        error = (lhs - rhs).abs()
        residuals[("p", "v", "a", "jerk", "snap")[d]] = error.amax(dim=(-2, -1))
        valid &= (error <= atol + REL_TOLERANCE * torch.maximum(lhs.abs(), rhs.abs())).all(dim=(-2, -1))
    return residuals, valid


class MincoAdapter:
    # Explicit numerical domain guard; not an assertion of flight feasibility.
    min_supported_segment_s = 1e-3
    max_supported_segment_s = 10.0

    def __call__(self, head_pva, tail_pva, inner_pts, durations):
        values = (head_pva, tail_pva, inner_pts, durations)
        for value, name, shape in zip(values, ("head_pva", "tail_pva", "inner_pts", "durations"),
                                      ((6, 3, 3), (6, 3, 3), (6, 2, 3), (6, 3))):
            tensor(value, name, shape, fp32=True)
        matching(values)
        if not bool(((durations >= self.min_supported_segment_s) &
                     (durations <= self.max_supported_segment_s)).all()):
            raise ValueError("segment durations outside supported numerical domain [0.001,10] seconds")
        batch = durations.shape[0]
        with torch.autocast(device_type=durations.device.type, enabled=False):
            # Original constants are plain tensors. Rebuild on the explicit input device,
            # never migrate a cached solver with .to() or share it between forward calls.
            solver = MincoS3NU(piece_num=3, device=durations.device)
            try:
                solver.set_parameters(head_pva.reshape(-1, 3, 3), tail_pva.reshape(-1, 3, 3),
                                      inner_pts.reshape(-1, 2, 3), durations.reshape(-1, 3))
            except RuntimeError as error:
                raise NumericalSolveError("original MINCO linear solve failed: " + str(error)) from error
            coeffs = solver.coeffs.reshape(batch, 6, 3, 6, 3).permute(0, 1, 2, 4, 3).contiguous()
            if coeffs.dtype != torch.float32 or not bool(torch.isfinite(coeffs).all()):
                raise NumericalSolveError("MINCO returned nonfinite or non-FP32 coefficients")
            with torch.no_grad():
                residuals, valid = constraint_residuals(coeffs, durations, head_pva, tail_pva, inner_pts)
                valid &= torch.stack([torch.isfinite(x) for x in residuals.values()]).all(0)
            if not bool(valid.all()):
                raise NumericalSolveError("MINCO numerical boundary/continuity residual failed",
                                          {"solve_valid": valid, "max_abs_residuals": residuals})
        return MincoTrajectory(coeffs, durations, valid, residuals)
