# Physics-only V1 S0 on live HEAD

**Status: isolated CPU mathematics and independent SHA-bound reader; not training-ready.**
Method ID: `physics_only_v1_s0_live_20261104`.

The reference patch was compared against live HEAD and the historical archive before integration. See [acceptance report](../../../docs/physics_only_v1/ACCEPTANCE_20261104.md) for exact commands, corrections, results and blockers, and [data contract](../../../docs/physics_only_v1/DATA_CONTRACT.md) for loader schemas.

The objective retains four per-query physical costs, explicit violation/quality diagnostics and fixed 2.0s equal durations. Mean/Soft-Min with configured guard are controlled alternatives. Strict existing goal evaluation metrics are unchanged. The continuous near/far training surrogate uses a changed normalization that requires ablation.

`dataset.PhysicsOnlyDataset` independently reads depth/state/native-AABB geometry and pinned binding certificates. It does not use the legacy frozen dataset, geometry sidecar or teacher arrays. SHA identity does not certify physical truth. Its production data has not been qualified.

`PhysicsOnlyObjective` still requires explicitly synthetic geometry. No forest conversion, trainer, optimizer, K=1 baseline, coverage objective or selector is introduced. C/F are sampled proxies; exact jerk integration and finite gradients do not establish flight safety.

CPU tests:
```bash
PYTHONPATH=. python -m pytest -q query_minco/training/physics_only_v1/tests
```
Use repository-compatible dependencies; the acceptance report records the existing-environment command. A no-update real-network diagnostic is available as `python -m query_minco.training.physics_only_v1.smoke --output <new-file.json>`. It never overwrites a result.
