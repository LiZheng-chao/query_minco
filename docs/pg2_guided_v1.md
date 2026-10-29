# PG2-Guided-v1: experimental loss components

This is a new objective namespace, not a PG1 recipe change. No online control,
automatic training or optimizer is constructed by the CPU0 interface.

## Interfaces and scope

- spec.py: four distinct PG2 methods and **draft** 32-update CPU RunSpecs.
- objective.py: unchanged PG1 forward/MINCO/Hungarian for the network and base
  imitation; independently composes six PG2 top-level components.
- geometry.py: rechecks each Teacher against the original continuous hard
  validator, binds content+state+geometry, balances unmatched Queries across
  verified Teacher routes. A Teacher point free ball is not a certificate for a
  predicted continuous curve. No Teacher means UNKNOWN, not an unreachable task.
- ranking.py: raw19 rank logit and raw20 hard-label classifier, training/offline
  only. Neither logit is a calibrated safety probability or execution permission.
- gradients.py: individual, four-stratum and layer gradients; optional first-order
  quality projection, off by default. It does not guarantee nonlinear safety.
- audit.py/cli.py: bounded frozen full Frozen384 CPU0, strict JSON evidence and
  readonly sealed open-dev8 replay. Development metadata is never relabeled train.

## Mathematics and units

All sums over sampled time use alpha_k=dt_k/T (segment trapezoid endpoint halves),
64 nodes per segment. Positions/velocities/accelerations are in fixed B0 FLU, SI
metres and seconds. Body radius .12 m and required margin .03 m are fixed.

L = lambda_I I + lambda_S S + lambda_G G + lambda_C C + lambda_F F + lambda_R R.
All new components are dimensionless. Research defaults are candidates, not
optimal weights or training approval. Each top-level and safety/control/feasibility
subcomponent has an explicit coefficient/switch.

### Safety: differentiable proxies, not certificates

d_k=SDF_union(p_k); n_k is the detached local SDF normal.
closing_k=max(-v_k dot n_k,0). The optional stopping buffer is
b_k=closing_k^2/(2 a_brake), with explicit positive candidate a_brake=4 m/s^2.
d_target=.12+.03+b_k. Risk r_k=.05 softplus((d_target-d_k)/.05)/.15.
Clearance=LSE_tau(r,alpha), tau=.025/.15, normalized by physical time mass.
No closing-speed rho multiplier is silently inherited.

S = c_clearance Clearance + alpha Retreat + beta Corridor.
First discrete point violation or uncertified straight sample chord selects a
repair event. Retreat=||p_first-stopgrad(p_previous_safe)||^2/(1 m)^2.
The anchor is verified outside the union by at least radius+margin; when absent,
the term is zero with NO_PREVIOUS_SAFE_ANCHOR, never a safety success.
Chord logic is only a sampling proxy: it does not prove a curved segment safe.

Teacher routes are checked by the unchanged validate_teacher. Hungarian matches
keep their Teacher. Unmatched assignments fill the least-used Teacher first,
then minimum detached route distance, with deterministic ties. This preserves
coverage of available routes but does not invent missing modes or prove homotopy.
Target points follow each Teacher's three segments at the same local u nodes.
Each free-ball radius is min(SDF-.15, distance to center domain, .2 m)-1e-5,
clamped at zero. Corridor=mean_alpha ReLU(||p-target||-ball_radius)^2/(1 m)^2.
Targets, assignment and ball radii are detached. A zero-radius ball is logged,
not inflated to an invented free corridor. Complete curves retain the Bernstein
hard verifier as the sole offline hard authority.

### Guidance

Near is the original <=3.4 m task split, with continuous endpoint position,
terminal velocity and acceleration squared costs normalized by goal length
(minimum 1 m), 3 m/s and 8 m/s^2. This provides signal inside old success hinges.
Far uses exp(-projected_progress/max(goal_length,1 m)) plus a bounded directional
penalty; it does not require arrival at the distant global goal.
If any sampled point is inside the body radius, direct goal attraction is
replaced by verified Teacher-route terminal guidance. No Teacher means no direct
goal-pushing signal in that severe case. Trusted separating-slab infeasibility
retains the existing goal-supervision gate. The original Near/Far evaluation
criteria are unchanged.

### Feasibility and control quality

F retains the exact original dynamic_limit_loss (3 m/s,8 m/s^2,100 m/s^3),
plus optional detached curvature speed cap sqrt(a_lat/kappa), Near terminal
deceleration, center-domain loss and optional zero-drag fixed-yaw flatness proxies.
The default a_lat=4 m/s^2 is a soft research budget, not a relaxed hard threshold.
Flatness is disabled by default and requires actual B0 gravity. Specific thrust
is a-g; tilt, specific-thrust limits and perpendicular body rate ||j_perp||/||a-g||
are proxies assuming zero drag and fixed yaw. They omit yaw rate, motor dynamics
and Controller response.

C = c_a integral ||a||^2 dt/(8^2*2.2) +
    c_j integral ||jerk||^2 dt/(100^2*2.2) + c_T T/2.2.
Reuse R1's positive four-point Gauss-Legendre evaluation, which integrates these
degree-six/degree-four polynomials exactly in exact arithmetic. It is tested
against independent polynomial antiderivatives and coefficient/duration finite
differences. Unnormalized energies have units m^2/s^3 and m^2/s^5.
Smaller energies do not establish improved Controller tracking.

### Ranking and NONE

Only freshly bound original-hard-validator labels generate detached targets.
Among hard-feasible Queries, target cost is -conservative_clearance/.15 +
.01*normalized_quality - projected_progress/(1 m).
Cross entropy supervises raw19; BCE supervises raw20. All fail: rank target NONE
and zero rank CE; classifier receives negatives. Online execution still needs an
independent observed-space safety gate and must return NONE without it.
A matching digest is not itself geometric proof. Predicted-curve labels are
recomputed and bound to coefficients/durations/head, state, geometry and validator
SHA. A 100 m trajectory shift invalidates the binding.

## Gradient and training boundaries

Per-loss parameter norms/cosines, four matched/pass strata, raw15:19, decoder,
query embedding and head norms are logged. Physical terms never supervise
raw19:21; only Ranking deliberately does. Quality projection removes the component
opposing the safety gradient only as a first-order diagnostic and is off.

The original shared run_engine enforces immutable score rows for PG1. PG2 must
not disable that assertion to train Ranking. A separately reviewed per-method
score-row contract and new CPU0/run authorization are required before training.
Draft plans deliberately have no runnable training authorization.

## References and differences

[DiffPhysDrone main_cuda.py](https://github.com/HenryHuYu/DiffPhysDrone/blob/master/main_cuda.py)
uses learned actions through differentiable simulated dynamics and rollout losses.
Its acceleration/jerk terms concern controls across rollout steps. This PG2
implementation differentiates decoded MINCO curves; it does not reproduce that
physics rollout.

The local read-only YOPO-MINCO safety/guidance/feasibility/smoothness sources
supply mechanism references: bottleneck focus, retreat, detached collision goal
gating and curve-energy penalties. They are the local working-tree revision;
their SHA is recorded and no claim is made that it equals upstream.

[GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/gcopter/include/gcopter/gcopter.hpp)
uses corridor/MINCO optimization and a differential flatness mapping. This PG2
Teacher point-ball guidance is not GCOPTER's complete safe flight corridor
optimization. LSE safety alone is not a novel contribution.
