# Independent PG3 V2.1 paired128 execution

This execution profile adds no loss or weight change. A calls the unchanged PG2
Safe scalar objective; D calls existing ObjectiveV21 with the exact qualified
weights from baaa590. The old qualification/formal recipes and outputs remain
read-only. The new namespace is research_outputs/pg3_v21_paired128_v1.

The only existing core edit makes step_guard_v21's already declared increment
limit2 actually reject an update before optimizer.step, rather than only return
a Boolean. Main CPU0 limit10, two-step runtime strategy, Keep activation budget
.05, clipping10, formulas, matching, sampling, inference, score rows and original
hard/Goal validators remain unchanged. Tests explicitly prove rejection leaves
parameters/optimizer/exposure uncommitted. A uses its original PG2 ratio guard,
not D's replacement-base or incremental budget.

`executor_v21.Engine` uses Seed42, CPU/FP32/single-thread, Batch8, original
FrozenOrder, fresh AdamW (.9,.999), eps1e-8, weight_decay0, clip10 and the original
128-index cosine2e-5->1e-6 schedule. A short4 validation uses the FIRST FOUR values
of the128 schedule; it does not compress the schedule to4. All loss components,
raw score zero gradients and reconstruction are audited before clipping.

Every update records pre/post original hard Physics, Goal, Oracle@6, FAIL->PASS,
PASS->FAIL, fixed C1 initial-matched-safe regressions, candidate IDs/reasons,
Section coverage, Geom branch proportions, Keep activation and coordinate/time
drift, all four gradient ratios, component Gram/norms and actual Adam delta
cosines. A's new-term diagnostics are structural zeros, never added to its loss.
Initial matching and safety identities are evaluated with the immutable C1.
Metrics on a batch count occurrences; full frozen-split evaluations use distinct
tasks/candidates. Oracle is not an online selector.

Each accepted update is exclusively published in commits/stepNNNN: model,
optimizer, unique-SID exposure, monitor/gradient alarm state, Torch/Python/NumPy
RNG, RunSpec digest, SHA inventory and parent commit digest. Restore verifies the
entire contiguous chain, all hashes, optimizer steps/LR/options, exposure cursor,
model and frozen score rows. Pending/duplicate/incomplete commits fail closed.
A post-optimizer exception rejects the run; parameters are not claimed rolled
back. Only intentionally paused short verification runs are resumable through
this CLI. Failed formal runs are never automatically retried or resumed.

Short preflight executes independent A/D uninterrupted4 and2+restore+2 replicas.
The final model, optimizer, exposure and RNG fingerprints and per-step IDs/model/
exposure hashes must be exactly equal within each arm. Both must complete before
formal execution is permitted. Any real gradient/Keep/increment failure is kept
as FAILED; it is never fixed by lowering weights or relaxing thresholds.

Formal runs evaluate full Training384/Development192 at0/32/64/96/128 and retain
original open-dev8 two-checkpoint degradation alarms. Existing1800s,4096MiB RSS,
3072MiB available and4GiB per-run bounds remain. Development metrics are evaluation
and frozen alarms only; no weight/recipe search or automatic adaptation exists.

`formal_v21` exposes plan/run/compare-short/summarize. RunSpec and authorization
are separate immutable files; authorization must bind exact spec SHA/scope/limit.
The user request in this session explicitly authorizes short verification and
then the paired128 experiment if preflight succeeds. No new user confirmation
is needed; the executor itself does not fabricate authorization files. The new
source is committed and requalified in a strict clean worktree; frozen weights
and config must compare exactly to baaa590 qualification before any run.

```bash
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
# After new clean-source CPU0/Qualification and exact frozen-weight verification:
$PY -m experiments.pg3_v2_no_corridor.formal_v21 plan --preparation <new-prep> --output <new-base>/plans
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.formal_v21 run --spec <spec> --authorization <auth>
# Short resume only, after intentional --pause-after2:
CUDA_VISIBLE_DEVICES='' $PY -m experiments.pg3_v2_no_corridor.formal_v21 run --spec <short-r-spec> --authorization <short-auth> --resume
$PY -m experiments.pg3_v2_no_corridor.formal_v21 compare-short --output <new-base>/short
$PY -m experiments.pg3_v2_no_corridor.formal_v21 summarize --output <new-base>
```

Baseline same-Batch0 one-step evidence is in stage0/PAIRED_STEP1.json: both A and D
make f00_narrow3d_g5_v1 Q3 fail hard, with collision_or_required_margin_violation;
both have1 PASS->FAIL,1 FAIL->PASS and Oracle8->7 on this batch. Thus that specific
single-step regression is not PG3-exclusive. This is not a conclusion about long
runs or global safety. The actual new qualification/short/formal statuses and
numbers live in the new evidence directory; this document does not invent them.
Forest, online control and flight are prohibited and never invoked by this entry.
