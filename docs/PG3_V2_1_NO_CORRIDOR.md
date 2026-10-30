# PG3 V2.1 Loss Simplification & Gradient Repair

Method `query_minco_pg3_v2_no_corridor_v2_1`; composition `pg3_v2_1_loss_compose_v1`.
Implementation base `651152c1b3214101b54536c6904f4f38f13ceb57`, developed on
`research/pg3-feasible-mode-v1`. The PRD is the accepted implementation contract;
its PROPOSED status is not evidence that qualification/training has occurred.

## Exact objective and PG2 isolation

For the **actual effective** weights returned by the original PG2 forward:

`L_A = L_PG2 = sum_k w_pg2[k]*L_pg2[k]`.

`L_D = L_PG2 - w_pg2[braking]*L_pg2[braking]
       + lambda_stop*L_stop + lambda_mode*L_mode + lambda_geom*L_geom
       + lambda_keep*L_keep + lambda_div*L_div`.

Only D calls `ObjectiveV21`. `stop_mode=replace_full_stop` requires lambda_stop>0.
`stop_mode=legacy` requires lambda_stop=0, retains original Braking, and has a
separate `_legacy_stop` method ID. Legacy plus all new weights zero returns the
exact PG2 loss tensor, with the same raw/parameter gradients and no new label work.
No PG2 source, old PG3 V1 source, architecture, inference, MINCO or hard validator
is modified. Six queries, two decoder layers, 21D output, input shapes, three
quintic segments, Goal and v/a/j=3/8/100 are unchanged. Scores raw19/raw20 remain
unsupervised; finite/zero-score-gradient and exact parameter-row guards remain.

Historical V2 APIs (`Objective`, `section_side_proxy`, `full_stopping_risk`,
`PerSampleFairAssignment`, old audit) remain available under their original names.
They are NOT the V2.1 path. The new entry imports versioned V2.1 APIs explicitly.
Historical V2 CLI is available only with `--legacy-v2`, using `pilot_v2.json`.
Previous evidence and docs are retained, never rewritten as V2.1 evidence.

## Formula-to-code map and branch tree

| PRD term | Implementation | Definition |
|---|---|---|
| PG2/base | `objective.ObjectiveV21.augment` | Read actual PG2 loss, braking term and effective weight; subtract exactly the complete weighted Braking term only in replace mode. Clearance, Retreat and PG2 point-ball guidance stay intact. |
| Mode | Original `pg3_feasible_mode_v1.losses.mode_query` | Smooth L1 of two internal waypoints, endpoint and terminal velocity, coefficients 1/1/.1; no terminal acceleration/time targets. Mean selected dangerous unmatched queries within task, then mean batch, as V2. |
| Geom | `objective.geometry_query_v21` | Exactly one source per query: SECTION, BNECK_MATCHED, BNECK_FALLBACK or NONE. |
| Section | `revision.section_events_v21` | b=box_max[axis]+.15 (positive face) or box_min[axis]-.15 (negative); gap=sign*(p_sec[axis]-b); event=ReLU(-gap/.15)^2. delta_extra=0. No extra .03 in the numerator. |
| Bottleneck | Original `bottleneck_loss` | Teacher detached normalized progress, student physical time; original transverse Smooth L1 plus .1 axial Smooth L1 and original SI scales. |
| Stop | `revision.full_stopping_risk_v21` | vc=ReLU(-v dot n), dstop=vc^2/(2*4); h=.15+dstop-SDF; node=[.05*softplus(h/.05)/.15]^2. |
| Keep | `FrozenSafeReferenceV21` | Same V2 trust region: .08m waypoint/endpoint, .30m/s terminal-v (weight .1), .08s duration L1; squared hinge, mean over initial hard-safe queries only. Adds monitoring, not a new loss. |
| Div | Existing V2 teacher-directed `diversity` | Reliable distinct certified AABB face pairs only; no unconditional query repulsion and no homotopy assertion. |

Every Geom candidate requires a routed certified representable Teacher and current
hard FAIL. Original matched dangerous queries use BNECK_MATCHED only when valid
Bottleneck labels exist, else NONE/NO_VALID_BNECK. Selected dangerous unmatched
queries use SECTION when at least one legal section exists; otherwise valid
Bottleneck gives BNECK_FALLBACK, otherwise NONE. Safe, unrouted, untrusted and
unselected unmatched queries receive NONE. Mode keeps its own unchanged selection.

A legal Section label has a certified y/z face (axis 1/2, sign +/-1), valid Teacher
label and UNIQUE_FORWARD_CROSSING Teacher status. Student must have exactly one
crossing event, forward vx>1e-4 m/s, non-tangent, >1e-6s from endpoints and joints.
The detached root selects a local branch; evaluate differentiably at its normalized
progress and use `p_sec=p(t0)+v(t0)*[-(p_x(t0)-plane_x)/stopgrad(v_x(t0))]`.
Missing/tangent/multiple/backward/join/slow events do not contribute Section or an
axial progress penalty. They fall back to the ORIGINAL Bottleneck. This differs
from historical V2's teacher-progress Section proxy.

All legal Section events of a query are averaged; any active Section shuts off
the query's entire Bottleneck contribution. Geom averages only nonempty eligible
query values within a task; empty task gives graph-connected zero; batch reduction
is an arithmetic mean. Records contain source, event denominator, task query
denominator, accepted box/event IDs and all skipped reasons. Branch diagnostics
use the SAME denominators and sum to the single Geom term.

All coordinates are B0 metres, time seconds, velocity m/s, acceleration m/s^2.
Stop is dimensionless: tau=.05m, d_ref=.15m, d_safe=.15m. It uses original PG2
normalized time_weights per query, then mean queries/batch. No stopping-distance
cap is applied. The active nearest-AABB/face normal and its exterior-corner
position derivative are retained by scaling the historical V2 primitive exactly
by (.05/.15)^2. Certified empty geometry gives connected zero; unknown or
contradictory empty metadata fails closed. Hard box/face ties are nonsmooth.
Stop still overlaps PG2 Clearance at zero velocity; its cosine with Clearance
is reported, never claimed eliminated. None of these proxies certifies flight.

## Unique sample exposure and checkpoint semantics

`UniqueSampleFairAssignment.begin_step` declares batch IDs. `choose_once` binds
depth/state/Teacher tensors, authenticated scene, matching, hard labels, prediction
and TeacherBank identity. Same SID returns exactly the cached assignment;
conflicting inputs/Teacher/labels raise DUPLICATE_SAMPLE_CONFLICT. Pending is a
SID dictionary. Commit advances eligible/direct/wait once per successful update
and unique SID; receipt records both batch_occurrences and unique_samples.
Hungarian IDs, Teacher counts, capacity2 and safe-unmatched exclusion are unchanged.

`optimizer_step_v21` checks finite and frozen score rows before/after a successful
step, then commits. Pre-step/optimizer errors clear pending history. Post-step
errors reject the run; parameters are NOT claimed rolled back. No unvalidated
continuation exists in the V2.1 CLI. Pending checkpoints/double commits/repeated
steps are rejected. Atomic exclusive checkpoints bind model+optimizer+exposure,
source/spec hash, seed/order and Torch RNG; exact restore/replay is tested.

## Full gradient audit and preregistered calibration

`audit.audit_v21` records all original PG2 terms, all five V2.1 terms, the three
Geom branch diagnostics, PG2 total and base total. Every vector uses the same
named parameter order/dtype/dimension. The frozen reference is excluded. Reports
include value/norm/weighted norm, Gram, cosine matrix, negative conflicts,
parameter-group norms, module identity, matched/unmatched x hard PASS/FAIL strata,
raw/time/coefficient routes and raw19:21 zero checks. Value and parameter-gradient
expansions are checked at fixed atol=3e-5/rtol=3e-4 for gradients and
atol=1e-7/rtol=1e-6 for scalar values; tolerances are not changed after qualification.
Zero imitation gradient fails closed.

Let g_pg2=grad L_PG2, g_b=grad L_braking, g_im=grad L_imitation.
For replace mode g_base=g_pg2-w_b*g_b and g_D=g_base+sum lambda_k*g_k.
Reports separately record R_PG2=norm(g_pg2-g_im)/norm(g_im),
R_base=norm(g_base-g_im)/norm(g_im), R_D=norm(g_D-g_im)/norm(g_im),
R_increment=norm(g_D-g_pg2)/norm(g_im). The last includes NEGATIVE removed Braking.

`pilot.json` preregisters first 8 FrozenOrder Seed42 Training384 Batch8 rows,
positive budgets mode=.5, geom=.3, stop=.1, keep=.05, div=.05, CPU0 limit10,
auxiliary increment limit2. Mode/Geom/Stop/active Div use budget divided by median
positive component/imitation norm ratio. Keep uses the maximum positive ratio
from active clone probes so every measured probe stays within its .05 budget.
A single common scale enforces a triangle upper bound on increment including the
removed Braking; joint main/probe Gram ratios are then recorded. No Development192
or historical14 search is used. The actual input expectation is 64 occurrences,
62 unique tasks, with source/order/tensor/geometry identities checked.

Initial Keep loss/gradient must be zero. `make_keep_probe_v21` deep copies the same
C1 and adds a fixed +.12 to trajectory-head y biases indices [1,4,7]; no score bias
changes. This is PROBE_ONLY, with fixed config/model hashes, no optimizer, no real
model or committed exposure changes. All 8 probe batches must have active finite
Keep gradients. Missing Keep signal blocks qualification; it is never silently
set to a default pilot weight. Diversity without a usable pair is lambda=0,
INACTIVE_NO_PAIR. Critical Geom/Stop below lambda1e-12 or median weighted ratio1e-4
are EFFECTIVELY_INACTIVE and block qualification. R_base>10 blocks even if new
terms cancel it. The historical R_D<=10 remains the main CPU0 gate; runtime's
continuous two-step strategy is preserved. `step_guard_v21` monitors first active
Keep and fails its budget before clipping; it also records the increment limit.

## Stages and evidence lifecycle

Stage0 records original SHA, old evidence and the unmodified 136-test baseline.
Stages1--4 implement and unit-test formulas/geometry/transactions/probes/audit;
real evidence tests stay NOT_RUN until Stage5. Stage5 occurs AFTER commit in a
clean detached worktree of that exact commit, with external read-only data and
new output. Strict verify_code keeps allow_uncommitted_sources=False. Only the
historical reference JSON's location changes to the worktree; its content SHA
must be identical. Unrelated original untracked files are preserved in mainline.

Stage5 runs qualified real Frozen384 audit/probes, CPU0 and a single isolated
AdamW shadow step on the first of the same 8 batches with selected dangerous
geometry. It records real spatial deficit changes, safe-query drift, original
model immutability, actual Adam delta/negative-gradient cosines and a PROBE_ONLY
checkpoint. No improvement above the preregistered 1e-7m threshold gives
NO_REPAIR_SIGNAL and blocks training. This is not an online Shadow episode or
an assertion of global hard-safety improvement. Full qualification requires every
T01--T19 PASS, all old136 tests PASS, CPU0 and activation/source/data checks PASS.

Files under `research_outputs/pg3_v2_1_no_corridor/_preparation/<id>/` include
BASELINE_AUDIT, DATA_QUALIFICATION, GRADIENT_AUDIT_V2_1, CPU0_V2_1,
SHADOW_OPTIMIZER_PROBE, TEST_REPORT, QUALIFICATION, RUN_SPEC_A/D and
IMPLEMENTATION_RESULT_V2_1. These are exclusive, hashed, final-HEAD-bound files.
Actual numerical results/status live there; this document does not fabricate them.

The old FullTrain import also reads three AST functions from the historical,
untracked `research_outputs/pg1_v2_diagnostics_v1/_preparation/diagnose.py`.
A clean worktree needs an explicit file symlink to that original read-only file.
Its original absolute path, content SHA and worktree link are preregistered in
`pilot.json.external_readonly_code` and verified in every source snapshot/RunSpec.
No historical function or input is copied or edited, and the dependency is not
silently omitted from the frozen source identity. The first missing-dependency
attempt is retained as FAILED, not reported as calibration or CPU0 success.

## Reproduction and authorization boundary

```bash
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
# In a clean worktree of the final implementation commit; OUTPUT is external.
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.run prepare \
  --expected-commit <full-commit> --output <new-exclusive-output>
$PY -m experiments.pg3_v2_no_corridor.run tests \
  --evidence <new-exclusive-output> --output <new-exclusive-output>/TEST_REPORT.json
$PY -m experiments.pg3_v2_no_corridor.run finalize --output <new-exclusive-output>
```

The V2.1 CLI exposes prepare/tests/finalize ONLY. It never creates a formal-training
authorization or executes128 updates. RunSpec A uses original PG2 weights/method;
D uses only calibrated V2.1 weights. Both bind the same C1/data/order/seed,
FP32/CPU/Batch8, AdamW(.9,.999), eps1e-8, weight_decay0, clip10 and original
cosine2e-5->1e-6 schedule. A QUALIFIED status only permits requesting a separate
explicit128-step authorization; it does not authorize training. Any FAILED,
BLOCKED, INACTIVE critical loss or NOT_RUN acceptance item blocks that request.
Training/paired evaluation/forest/online Shadow/flight remain NOT_RUN. Online
selection, observed-map continuous validation, age checks, fallback, tracking
monitoring and emergency stop are independent later qualifications.
