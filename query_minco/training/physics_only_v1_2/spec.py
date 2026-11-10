"""V1.2 mathematical contract. Defaults are uncalibrated CPU0 candidates."""
from dataclasses import dataclass, fields
import math

METHOD_ID = "physics_only_v1_2_huber_goal_allquery_v1"
DOMAIN_ABLATION_ID = "physics_only_v1_2_huber_goal_allquery_domainwidth_ablation_v1"
GOAL_COMPONENT_ID = "goal_v12_phi2_near015_transition32_36_charbonnier005_v1"


@dataclass(frozen=True)
class PhysicsV12Spec:
    method_id: str = METHOD_ID
    goal_component_id: str = GOAL_COMPONENT_ID
    phase: str = "B"
    aggregation: str = "softmin_guard"
    total_time_s: float = 2.0
    points_per_segment: int = 64
    position_anchor: str = "none"
    radius_m: float = .12
    clearance_m: float = .03
    collision_epsilon_m: float = .05
    collision_tau: float = 1 / 6
    kappa: float = 2.
    domain_ref_m: float = .15
    domain_mode: str = "physical_0p15"
    v_max_mps: float = 3.
    a_max_mps2: float = 8.
    j_max_mps3: float = 100.
    jerk_ref_mps3: float = 100.
    time_ref_s: float = 2.2
    eta_domain: float = 1.
    w_cv: float = 1.
    w_f: float = 1.
    w_g: float = 1.
    w_cc: float = 1.
    w_p: float = .25
    w_s: float = 1.
    beta_A: float = .1
    set_tau: float = .3
    lambda_all: float = .1
    lambda_div: float = 0.
    selector_enabled: bool = False

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(f.default, float):
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError("finite real required: " + f.name)
                if v < 0:
                    raise ValueError("negative scale/weight: " + f.name)
        frozen = dict(total_time_s=2., position_anchor="none", radius_m=.12,
                      clearance_m=.03, collision_epsilon_m=.05, kappa=2.,
                      domain_ref_m=.15, v_max_mps=3., a_max_mps2=8.,
                      j_max_mps3=100., jerk_ref_mps3=100., time_ref_s=2.2,
                      goal_component_id=GOAL_COMPONENT_ID, lambda_div=0.,
                      selector_enabled=False)
        if any(getattr(self, k) != v for k, v in frozen.items()):
            raise ValueError("fixed V1.2 contract changed")
        if type(self.selector_enabled) is not bool:
            raise ValueError("selector_enabled requires bool")
        if type(self.points_per_segment) is not int or self.points_per_segment < 64:
            raise ValueError("at least 64 points per segment")
        if min(self.collision_tau, self.set_tau, self.eta_domain, self.w_p) <= 0:
            raise ValueError("positive tau, eta_domain and progress weight required")
        if self.phase not in ("A", "B"):
            raise ValueError("unknown phase")
        if (self.phase == "A" and self.aggregation != "foundation") or (
                self.phase == "B" and self.aggregation not in ("softmin_guard", "mean_guard", "mean")):
            raise ValueError("phase/aggregation mismatch")
        expected = {"physical_0p15": METHOD_ID, "s0_domain_width_ablation": DOMAIN_ABLATION_ID}
        if self.domain_mode not in expected or self.method_id != expected[self.domain_mode]:
            raise ValueError("domain ablation must have its own method identity")
