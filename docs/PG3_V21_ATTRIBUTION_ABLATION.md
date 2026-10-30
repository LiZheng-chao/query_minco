# PG3 V2.1 safety attribution and fixed sequential ablations

Base experiment commit: 0e1c004e42b5666e7ff4fe982c7bca722619ab5d.
No existing tracked source is edited. New modules add read-only evidence analysis and component switches around existing ObjectiveV21/Engine, never a new loss or re-calibration.

A: sealed original PG2. S: remove the actual original weighted Braking term and add existing Full Stop. M: S+existing Mode/fair assignment. G: M+existing exclusive Section/Bottleneck Geom. K: G+existing Keep, aliases sealed D after real Batch8 scalar/gradient exact-equivalence test. Diversity0 throughout. Frozen stop/mode/geom/keep coefficients are .000307330688447153/.010859947391365976/.00010354880977059005/.012369432948913293.

S never constructs a Teacher bank or assignment transaction and commits only an optimizer cursor (empty history). M/G/K retain the original unique-sample committed assignment. Disabled terms are graph-connected zeros, have zero audited gradients, and never receive fake exposure. M may compute existing diagnostic terms internally before excluding them; they do not enter its objective or gradient. Neither old D identity nor old weights are modified.

Every new S/M/G arm requires synthetic tests, eight original real Training8 forward/backward audits, unchanged CPU0 limit10/increment2/Keep .05 gates, and uninterrupted4 vs2+restore+2 fingerprints including model/optimizer/exposure/RNG/monitor. No weight calibration. New clean-source data bindings change code authority only; data/order/C1 are exact original artifacts. Formal execution verifies this transport explicitly.

The new formal executor reuses original Engine.step, score-row freezing, AdamW options, original128-index schedule, transactional SHA checkpoint chain, resources1800s/4096MiB RSS/3072MiB available/4GiB per-run, full train/dev eval0/32/64/96/128 and fixed original open-dev8 alarms. It requires a separately approved authorization bound to exact spec digest, scope PG3_V21_ABLATION128 and128 updates. The current task explicitly requests short qualification but reserves formal authorization to concrete RunSpecs; no new formal authorization is fabricated. Failure remains BLOCKED/FAILED without relaxation or automatic retry.

Entry points:

- `python -m experiments.pg3_v2_no_corridor.attribution_v21 --base <sealed-pair-root> --output <fresh-analysis-root>`
- `python -m experiments.pg3_v2_no_corridor.local_attribution_v21 --output <analysis-root>`
- `CUDA_VISIBLE_DEVICES='' python -m experiments.pg3_v2_no_corridor.ablation_v21 prepare --output <fresh-ablation-root>`
- `CUDA_VISIBLE_DEVICES='' python -m experiments.pg3_v2_no_corridor.ablation_v21 run --spec <S/M/G.json> --authorization <approved-exact-spec-auth.json>`

Run from a clean checkout with the exact pinned legacy diagnose.py link required by pilot.json. Reusing an existing output/run directory is rejected. Preparation does not launch128 updates.

Attribution uses original hard flags plus4097-node signed-surface-clearance diagnostics (margin .03m, body .12m); no new hard criterion. Query CSVs include all arms/checkpoints and parameters reconstructed from archived polynomial coefficients. Training exposure is joined only for Training384; development samples are evaluation-only. Norms from historical raw-gradient rows must not be described as per-query parameter gradients.

Local replay loads archived checkpoints into disposable models, recreates original batch loss gradients, separates learned_queries from remaining Decoder, and differentiates a fixed-time/nearest-box sampled clearance. It never creates an optimizer or modifies saved weights. Component negative-gradient derivatives are not Adam deletion counterfactuals and do not prove long-run causality.

Task-level bootstrap uncertainty resamples sample IDs, never independent queries. Development192 is not blind; all conclusions are Seed42/static-task/128-update conditional effects. S-A/M-S/G-M/K-G are sequential marginal effects, not order-independent attribution. If an arm is NOT_RUN, its comparisons remain blank. No new method, forest, ROS control, flight, loss tuning or selector bypass is implemented.
