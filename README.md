# Query-MINCO

This repository is the future development mainline. Historical experiments remain frozen in their original research workspace; changes here are never synchronized back into that workspace.

Reference method: `query_minco_b2c1_c1_e1_v1`.

The reference is CNN + 12D state encoder + six learnable queries + two Transformer decoder layers + a shared 21D head + three-piece quintic MINCO. Raw channels 19/20 have no effective imitation supervision and are **not usable scores or feasibility probabilities**.

## Install

Python 3.8 is the numerically verified environment; PyTorch 2.4.1+cu118, NumPy 1.22.3, SciPy 1.10.1, CPU FP32, one thread. Cross-version/device bitwise equivalence has not been established.

```bash
python -m pip install --no-deps -e .
python -m query_minco.registry
```

Dependency installation is separate; no checkpoint or dataset is distributed in this repository.

## Inference

```python
from query_minco.inference import load_planner

planner = load_planner(
    method_id="query_minco_b2c1_c1_e1_v1",
    checkpoint=verified_checkpoint_path,
)
candidates = planner.plan(depth_input=depth, state=state)
```

Inputs are NumPy FP32 depth/mask [B,2,128,160] and state [B,12]. State is velocity, kinematic acceleration, local goal, and gravity unit direction in B0 FLU. Depth is optical Z / 20 with a binary validity mask. This reference entry accepts only the sealed A42 C1 step2880 checkpoint SHA. Batch evaluation is serial, preserving the historical single-sample arithmetic. An explicit [B,6,S] physical-time array is optional.

Outputs include raw, head/tail PVA, inner points, durations, coefficients, and P/V/A/J. These are candidate trajectories; this API neither selects a safe trajectory nor executes commands.

## Validation and scope

Synthetic CPU inference: 228 bitwise comparisons passed. A synthetic M1/M4 batch, matching, losses, parameter/time gradients and a C1A canonical teacher: 129 bitwise comparisons passed. No optimizer updates or scientific evaluation were performed. A wheel was installed into an isolated target and used with the historical source tree hidden.

Teacher C1A/R2 implementations are separate. The full R2 search and full frozen-cache replay are not validated here. A complete production training launcher has not been migrated; `training.objective.forward` and `training.recipe` are verified building blocks, not a claim of end-to-end training reproduction.

Near/Far contracts and the historical scene-domain versus diagnostic-domain difference are retained. No selector or controller is registered. No ROS/plant execution occurs on import.

See [architecture](docs/PACKAGE_ARCHITECTURE.md), [workflow](docs/NEXT_DEVELOPMENT_WORKFLOW.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Existing teacher-wrapper provenance gaps remain disclosed in the method manifest.

## Licensing

Vendored YOPO MINCO is MIT, with its original notice included. Query-MINCO's project-wide license remains pending author confirmation. Do not treat this branch as approved for publication.
