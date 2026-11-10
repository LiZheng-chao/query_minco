"""S0-only experiment contract. Constants are proposals, not fitted weights."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PhysicsSpec:
    method_id: str = "physics_only_v1_s0_live_20261104"
    total_time_s: float = 2.0
    points_per_segment: int = 64
    # Physics scales (SI), not certificates.
    radius_m: float = 0.12
    clearance_m: float = 0.03
    collision_epsilon_m: float = 0.05
    collision_tau: float = 0.16666666666666666
    v_max_mps: float = 3.0
    a_max_mps2: float = 8.0
    j_max_mps3: float = 100.0
    jerk_ref_mps3: float = 100.0
    time_ref_s: float = 2.2
    # S0 values are placeholders until training-set no-update calibration.
    wc: float = 1.0
    wf: float = 1.0
    wg: float = 1.0
    wj: float = 1.0
    eta_progress: float = 0.25
    eta_domain: float = 1.0
    set_tau: float = 0.3
    guard_weight: float = 0.1
    guard_feas_ratio: float = 1.0
    aggregation: str = "softmin"  # or "mean" for matched ablation
    # Diagnostic transition replaces discontinuous training goal, not hard validator.
    near_threshold_m: float = 3.4
    transition_half_width_m: float = 0.2
    # Keep legacy decoder default. Optional diagnostic only: goal_ray.
    position_anchor: str = "none"

    def __post_init__(self):
        for key, x in vars(self).items():
            if isinstance(x, float) and not isfinite(x):
                raise ValueError(f"nonfinite {key}")
        if self.method_id != "physics_only_v1_s0_live_20261104":
            raise ValueError("this snapshot implements S0 only")
        if self.total_time_s != 2.0:
            raise ValueError("S0 fixes total_time_s=2.0")
        if type(self.points_per_segment) is not int or self.points_per_segment < 4:
            raise ValueError("points_per_segment >= 4 required")
        for key in ("radius_m", "collision_epsilon_m", "collision_tau", "v_max_mps", "a_max_mps2", "j_max_mps3", "jerk_ref_mps3", "time_ref_s", "set_tau", "near_threshold_m", "transition_half_width_m"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("clearance_m", "wc", "wf", "wg", "wj", "eta_progress", "eta_domain", "guard_weight", "guard_feas_ratio"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.aggregation not in ("mean", "softmin"):
            raise ValueError("unknown aggregation")
        if self.position_anchor not in ("none", "goal_ray"):
            raise ValueError("unknown anchor diagnostic")
        if self.transition_half_width_m >= self.near_threshold_m:
            raise ValueError("transition band must not reach zero goal")
