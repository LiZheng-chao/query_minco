# PG2-Guided-v2

Isolated research namespace. Baseline, Teacher, Hungarian, MINCO, original hard
validator, PG1 score-row contract, v1 files and historical outputs are unchanged.
Reports and raw CPU0 evidence live under research_outputs/pg2_guided_v2.

The objective logs 15 separate components. Confirmed point violations, uncertain
sample chords and no discrete failure are distinct. Only confirmed violations
trigger Retreat; absence of an anchor is not safety. Teacher balls are training
targets, never continuous certificates for a predicted curve. Certified hard-pass
routes are not forced into one Teacher tube. M1 cannot create unobserved modes.

The optional braking reserve is bounded by a .1 s reaction horizon and .15 m
extra distance; uncapped required stopping distance and cap exceedances remain
logged. It is a proxy, not a full stopping guarantee. The hard .12+.03 m envelope
is unchanged.

Original Near/Far task penalties remain numerically intact. During confirmed
violation/domain violation their direct-goal derivative is detached; the separate
continuous route guidance supplies repair signal. Original Goal success tests
remain the only evaluation contract.

Lateral acceleration squared is ||v cross a||^2/(||v||^2+epsilon^2), epsilon=.1m/s,
with differentiable v and a. Fixed hard limits remain 3/8/100. Acceleration energy,
jerk energy and time have independent coefficients and gradient logs. Positive
four-point quadrature is polynomial-exact for quintic a^2/j^2 and is tested against
independent antiderivatives. Flatness remains diagnostic-only under explicit
gravity/frame/zero-drag/fixed-yaw assumptions, not Controller validation.

Calibration uses the complete Frozen384 training set, no open-dev8 selection:
median per-component parameter-norm ratio sets candidate scales; a fixed scalar
cap bounds the sum of new-component norm ratios by 2 on the audited training
batches. Original task=1, dynamic limit=.1 and domain=1 are fixed. Actual combined
norms retain the original two-consecutive >10 stop rule. This initial-state audit
does not predict later training gradients. Projection is optional, off in recipes,
and guarantees only a first-order gradient dot-product property.

Ranking modes: none, YOPO-style feasible-first CE with least-bad fallback, and
PG2 physical+Goal eligibility then quality CE plus invalid classifier. Targets
are detached and GT is training/offline only. Scores choose without GT; the
hard/Goal labels grade that choice offline. Scores are uncalibrated and not
execution permission; Oracle-filtered choice is not reported as online success.

execution.py has a NEW per-method score-row contract and reuses existing atomic
checkpoint/commit primitives. It does not call/patch the PG1 run_engine. Baseline
calls the original PG1 objective/update and freezes rows19/20; YOPO rank allows19
only; PG2 rank allows19/20. Training requires clean source, exact spec, full CPU0
seal, deployment preflight, single-run authorization and exclusive output.
No training is performed by import or cpu0.

Manual entry points:
- python -m query_minco.training.pg2_guided_v2.cli cpu0 --output ABS_EXCLUSIVE_RUN
- the same command with --require-clean after source freeze
- preflight --spec ABS_READY_SPEC --output ABS_EXCLUSIVE_PREFLIGHT
- validate --spec ABS_READY_SPEC --authorization ABS_AUTH
- run --spec ABS_READY_SPEC --authorization ABS_AUTH

Draft specs are intentionally not executable. No claim of server readiness is
made without actual deployment preflight. Source freeze/commit and authorization
are separate user decisions.

Reference ranking:
https://github.com/TJU-Aerial-Robotics/YOPO/blob/YOPO-MINCO/YOPO/policy/yopo_trainer.py
The rank_yopo_style arm reproduces its target rule, not its full score-weighted
generation or inner-waypoint/radius-head system. Ranking is auxiliary, not a new
scientific contribution. New Level2-3 data remain a separate factor after original
data ablations, requiring independent Teacher hard/Goal certification and
disjoint train/dev/test map families.
