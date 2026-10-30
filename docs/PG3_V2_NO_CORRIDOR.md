# PG3 V2 No Corridor implementation and audit

Method: `query_minco_pg3_v2_no_corridor_v1`. Experiment: `pg3_v2_no_corridor`.
Base: `9259211471d5ecaa34ebe48d003e5445bc5afcd8`, branch `research/pg3-feasible-mode-v1`.

The current user request takes precedence over the handoff's older proposal to
perform Fair-only first. The handoff and patch are evidence and proposed code,
not authority to overwrite old A/B/C definitions or launch training.

## Result and remaining blocker

Implementation, finite differences, forward/backward integration and related
regressions pass (136 tests: 79 PG2, 32 old PG3, 19 V2, 6 FullTrain128).
Real training, paired safety/Goal/Oracle results, latency measurement, forest
simulation, Shadow and flight: **NOT_RUN**. No performance improvement is claimed.

Frozen384 qualification was executed with the original data authority and SHA
checks. Calibration uses only the first 8 predeclared Batch8 rows of the original
Seed42 training order; it is NOT an audit of every training batch or a new dataset.
No Development192 outputs or old 14 failure cases enter calibration. Qualification
still verifies the existing development split separation; this is not weight search.

The CPU0 gradient startup guard **FAILS** at the requested pilot constants. On the
8 calibration batches, non-Imitation/Imitation norm ratios are about 89.63--846.36.
The first batch direct autograd ratio is 302.38846 (Gram reconstruction 302.38978;
small FP32 summation differences). Weighted Section/Imitation norms are
86.73--843.47 and Fullstop/Imitation 4.09--41.30. These are norms, not signed
contributions or evidence of hard safety improvement. Mode calibration is
0.010860012923447687; Bottleneck 0.01244014027163337; Diversity 0 because none
of these batches has an active reliable pair gradient. Safe Keep has zero loss
and gradient at the initial reference, as designed. Its active gradient is checked
with a separate synthetic finite-difference fixture.

The constants .10/.05/.01 remain pilot values, not validated best weights. They
were NOT silently reduced to pass the guard. `prepare` persists all records and
RunSpecs before returning nonzero; `A` and `D` reject a failed CPU0 preparation.
Further training-only calibration of the pilot terms requires an explicitly
versioned recipe; Development192 must not be used to select it.

## Formula to implementation

All positions are in B0 metres, velocities m/s, times seconds. Section planes
are x-normal; certified avoidance faces use y/z (`axis` 1/2). Normalized teacher
progress is detached; student segment times remain differentiable.

`query_minco/training/pg3_v2_no_corridor/objective.py::Objective` composes:

`L = PG2_safe + lambda_mode*mode + lambda_bneck*bottleneck + lambda_div*diversity
     + .10*section + .05*safe_keep + .01*full_stop`.

| Term | Code and exact reduction |
|---|---|
| PG2 Safe | Calls unchanged `pg2_guided_v2.objective.forward(...safe_v1...)`, with unchanged old calibrated weights, Hungarian and task/physical definitions. Its historical point-ball guidance component is retained; it is distinct from PG3 convex corridors. |
| Mode | Reuses old `mode_query`: sum of two waypoint Smooth L1 coordinate means, plus endpoint Smooth L1 coordinate mean, plus .1 terminal velocity coordinate mean. alpha_A=alpha_duration=0, all scales=1 in SI units. Only extra selected dangerous unmatched queries; mean queries then batch. |
| Bottleneck | Reuses old `bottleneck_loss`: at teacher normalized progress, Smooth L1 of transverse vector (coordinate mean), plus .1 axial Smooth L1; mean valid events, eligible dangerous routed queries, then batch. Existing dangerous matched-query routing remains. |
| Diversity | New V2 `diversity`: only selected query pairs with certified/representable teachers and different reliable AABB face signatures on a shared obstacle. For teacher transverse difference h and student difference delta at corresponding progress, `ReLU(.5*norm(h)-dot(delta,h/norm(h)))^2`; mean relevant pairs. No reliable pair => differentiable zero. Teacher-directed separation fixes the old norm-based zero gradient at exact collapse. |
| Section | `revision.section_side_proxy`: bound=box_max[axis]+.15 or box_min[axis]-.15; gap=sign*(p_axis-bound); `ReLU((.03-gap)/.15)^2`. Mean valid events per selected query, then selected valid queries, then batch (empty sample contributes zero). Note the requested formula asks for .03 additional face gap beyond the .15 bound. |
| Keep | `revision.FrozenSafeReference.loss`: `mean_{initial hard-safe queries}[mean_{2 waypoints} ReLU(norm(W-W0)-.08)^2 + ReLU(norm(P-P0)-.08)^2 + .1 ReLU(norm(V-V0)-.30)^2 + ReLU(sum_{3 segments} abs(T-T0)-.08)^2]`. Empty safe set => differentiable zero; unsafe queries excluded from numerator and denominator. |
| Fullstop | `revision.full_stopping_risk`: vclose=ReLU(-dot(v,n)); dstop=vclose^2/(2*4); risk=softplus((.15+dstop-SDF(p))/.05)^2; normalize positive trapezoid time weights per query, sum nodes, then mean queries and batch. No cap, no extra eps multiplier. |

No PG3 convex corridor is constructed or evaluated in V2. V2 does not call
conditional Control, Ranking, Head Projection, or the old PG3 objective. The only
legacy edit adds `build_corridor` (default True) to TeacherBank; old methods and
old tests retain their defaults. Zero all six new weights and the exact original
PG2 loss tensor and gradient are returned without new Teacher/reference work.

## Gradient and branch semantics

For one well-conditioned interior forward crossing away from MINCO joins,
`p_section = p(t0) + v(t0)*[-(p_x(t0)-x_section)/stopgrad(v_x(t0))]`.
The polynomial root and branch are detached; at the root this gives the local
implicit-section derivative. Require exactly one event, vx>1e-4 m/s, and distance
from global endpoints/segment joins >1e-6 s. Otherwise use teacher progress and
add `.1*((p_x-x_section)/1m)^2`. Slow, tangent, multiple, missing, coincident and
join crossings therefore have a consistent fallback. This is piecewise smooth;
no claim of differentiability across branch switches is made. Teacher section
labels must be valid and have UNIQUE_FORWARD_CROSSING status and a y/z face.

Fullstop uses an analytic active AABB normal that keeps the normal's position
derivative at exterior corners. A detached normal omits part of this formula's
gradient. Box/face ties and absolute-value center planes are nonsmooth and use
Torch's branch/subgradient choices; an exactly symmetric interior point may
have zero SDF normal. No sampled loss is a continuous safety certificate.

The reference is a deep copied eval-mode, requires_grad=False C1 model, checked
against the qualified initial weight SHA. Each cache key binds depth, state,
authenticated scene, certified-empty flag, reference identity and validator SHA;
reusing a sample ID with changed content raises. Current and reference parameter
gradients are isolated. The trust region cannot guarantee continued hard safety.

## Assignment and scientific identity

`PerSampleFairAssignment` reuses the original capacity2 slot Hungarian solver.
Original Hungarian matches remain; safe unmatched queries cannot obtain extra
Mode targets. Priority uses committed `(sample_id,query_id)` direct counts,
then waiting occurrences and query ID. Pending occurrences affect duplicates
within the same batch. `commit(step)` runs only after successful optimizer.step,
finite-parameter checks and frozen score-row checks. `discard` removes pending
history. Receipts store every occurrence and committed state digest; state_dict
supports deterministic restore/replay and rejects checkpointing pending work.

The six learnable queries, Transformer decoder, 21D head, MINCO, original hard
validator, Near/Far Goal and v/a/j=3/8/100 stay intact. raw19/raw20 have zero loss
gradients; fresh AdamW with weight_decay=0 plus exact row checks preserves them.
Training-only teachers, real geometry, labels and caches are not imported into
model/inference modules. No selector safety block was changed.

## Patch audit

The ZIP was reviewed rather than executed. Its apply script would replace old
fulltrain128 A/B/C definitions and output path, breaking historical entry
semantics. This implementation adds a separate entry instead.

Other corrected findings: duplicate occurrences were collapsed to any(selected)
at commit; history lacked checkpoint restore; reference cache only used sample
ID; averaging Keep over empty samples diluted safe-query supervision; low-vx
section fallback omitted its axial penalty; detached SDF normals omitted corner
derivatives; unconditional convex-corridor work could still run through the old
objective; norm-based separation had zero gradient at exact path collapse.
New teacher/face filtering excludes ambiguous event labels from Section/Diversity.

The ZIP release_gate was not adopted: its finite_metrics only checks numeric
Python types (not finiteness), ignores Goal/matched-safe regressions and input
provenance, and cannot qualify online execution. No offline summary grants a
flight release. Source mutation in an exception is not made atomic by AST parsing.

## Reproduction

Use the existing WSL environment (Torch 2.4.1+cu118, NumPy 1.22.3, Python 3.8;
pytest is absent, so tests use unittest). From the project root:

```bash
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
$PY -m unittest discover -s tests -p 'test_pg2*.py' -v
$PY -m unittest discover -s tests -p 'test_pg3*.py' -v
$PY -m unittest experiments.pg3_feasible_mode_v1.test_fulltrain128 -v
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.run prepare --preparation-id cpu0_postcommit_001
```

Preparation names are exclusive: an existing path must never be overwritten.
`cpu0_002` is the precommit audit, retained with its exact historical code identity.
The postcommit preparation binds final HEAD/package/entry/config SHAs and the
same authenticated dataset, geometry, C1 and Seed42 order. It is expected to
return nonzero at the pilot constants; inspect its CPU0.json and GRADIENT_AUDIT.json.
Preparation creates no optimizer and executes zero optimizer updates.

A/D execution entry (available but NOT_RUN; current CPU0 blocks it):

```bash
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.run A --preparation-id <passed-id> --authorization /path/to/A_authorization.json
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.run D --preparation-id <passed-id> --authorization /path/to/D_authorization.json
$PY -m experiments.pg3_v2_no_corridor.run compare
```

Authorization fields: approved=true, run_spec_sha256=digest of the exact generated
RunSpec, max_updates=128, scope=PG3_V2_SINGLE_CPU128_RUN, authorized_by=<researcher>.
No authorization file is fabricated by the new entry. Code must be committed
before preparation for an eventual training run. Source snapshots still include
the six preexisting untracked exposure files without staging them.

Both arms use C1/Seed42/FP32/Batch8/frozen order/AdamW (.9,.999), eps1e-8,
weight_decay0, clip10, original cosine 2e-5 to1e-6 over128 updates. Original
1800s/4096MiB/min-available3072MiB/4GiB resource limits and two-checkpoint
Goal/Oracle/mode degradation alarm are retained. Exact source/row/finite checks
run during training. Checkpoints bind optimizer+fair history and spec digest;
all new outputs live under research_outputs/pg3_v2_no_corridor.

Evaluation reuses original hard/Goal evaluation at 0/32/64/96/128. `compare`
requires identical initial CSVs and paired input/optimizer/order identities;
reports Physics, Goal, Oracle@6, FAIL->PASS/PASS->FAIL, initial matched-safe
regressions using the INITIAL query identities, per-query exposure, geometric
routes, QA/QJ/time and CPU network/MINCO latency. No online Selected@6 is implied.
The full A/D executor and compare on real trained outputs remain NOT_RUN; failed
runs remain failed, with partial evidence. This entry does not resume failed runs.

A/D tests a combination. Fair-only, +Section, +Keep and +Fullstop registered
ablations and multi-seed/blind evaluation are still future work, NOT_RUN.
Development192 is used development data, not a blind test. The 14 deep-penetration
cases are diagnostic only. Forest/Shadow/flight remain NOT_RUN: observable-map
continuous verification, selector, freshness, fallback, tracking/error monitoring
and emergency stopping must be qualified independently before execution.
