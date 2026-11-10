"""True six-query CPU0 model diagnostics with explicit per-arm source/config hashes."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.training.io import digest
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.physics_only_v1.smoke import parameter_hash
from .spec import PhysicsV12Spec
from .set_objective_v12 import PhysicsV12Objective
from .diagnostics import hard_diagnostics


def distribution(x):
    value = x.detach().double().reshape(-1)
    if not bool(torch.isfinite(value).all()) or not value.numel():
        raise ValueError("invalid logged distribution")
    return dict(min=value.min().item(), median=value.quantile(.5).item(), p90=value.quantile(.9).item(),
                max=value.max().item(), mean=value.mean().item(), l2=value.norm().item(),
                rms=value.square().mean().sqrt().item(), nonzero_rate=(value != 0).double().mean().item())


def cosine(a, b):
    a, b = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    den = a.norm() * b.norm()
    return None if den == 0 else (a @ b / den).item()


def run(spec):
    torch.manual_seed(20261104)
    torch.set_num_threads(1)
    model = QueryMincoPlanner().cpu().eval()
    before = parameter_hash(model)
    state = torch.zeros(2, 12)
    state[:, 0] = .8
    state[:, 3] = .1
    state[:, 6] = torch.tensor([2.4, 6.])
    state[:, 11] = -1.
    depth = torch.zeros(2, 2, 128, 160)
    depth[:, 0] = .1; depth[:, 1] = 1.
    domain = torch.tensor([[0., -4., -2.], [10., 4., 2.]])
    geometry = [SyntheticGeometry(torch.empty(0, 2, 3), domain, True),
                SyntheticGeometry(torch.tensor([[[.9, -.3, -.3], [1.2, .3, .3]]]), domain, False)]
    enabled, tasks = torch.ones(2, dtype=torch.bool), ("normal", "normal")
    shared = model.depth_encoder.d1[0].weight
    report = dict(seed=20261104, device="cpu", optimizer_updates=0, episodes=0,
                  qualification="SYNTHETIC_TEST_ONLY", input_note="constant depth, not rendered",
                  query_minco_file=sys.modules["query_minco"].__file__, python=platform.python_version(),
                  torch=torch.__version__, numpy=np.__version__, scipy=importlib.metadata.version("scipy"),
                  method_id=spec.method_id, shared_parameter="depth_encoder.d1.0.weight", arms={})
    arms = [replace(spec, phase="A", aggregation="foundation")] + [
        replace(spec, phase="B", aggregation=m) for m in ("mean", "mean_guard", "softmin_guard")]
    for config in arms:
        key = config.aggregation
        raw = model(depth, state)
        out = PhysicsV12Objective(config)(raw, state, geometry, goal_enabled=enabled, task_types=tasks)
        component_stats, gradients = {}, {}
        for name, weight in (("C_viol", config.w_cv), ("F", config.w_f), ("G_task_train", config.w_g),
                             ("C_shape", config.w_cc), ("P", config.w_p), ("S", config.w_s)):
            rg, sg = torch.autograd.grad(out.metrics[name].mean(), (raw, shared), retain_graph=True)
            gradients[name] = (rg, sg)
            component_stats[name] = dict(unweighted=distribution(out.metrics[name]),
                                         weighted=distribution(weight * out.metrics[name]),
                                         raw_gradient=distribution(rg), shared_gradient=distribution(sg),
                                         weighted_raw_gradient=distribution(weight * rg),
                                         weighted_shared_gradient=distribution(weight * sg))
        vg = torch.autograd.grad(out.metrics["V"].mean(), (raw, shared), retain_graph=True)
        qg = torch.autograd.grad(out.metrics["Q"].mean(), (raw, shared), retain_graph=True)
        setg = torch.autograd.grad(out.metrics["softmin_per_scene"].mean(), (raw, shared), retain_graph=True)
        rg, sg = torch.autograd.grad(out.loss, (raw, shared))
        if not torch.isfinite(rg).all() or not torch.isfinite(sg).all() or torch.count_nonzero(rg[..., 15:21]):
            raise ValueError("finite/fixed-time/scoring gradient contract failed")
        pairs = {"Cviol_P": (gradients["C_viol"], gradients["P"]),
                 "Cshape_P": (gradients["C_shape"], gradients["P"]),
                 "Smooth_GoalTask": (gradients["S"], gradients["G_task_train"]),
                 "V_Q": (vg, qg), "Guard_Set": (vg, setg)}
        hard = hard_diagnostics(out, state, geometry, enabled, tasks)
        metrics = {k: v.detach().tolist() for k, v in out.metrics.items()}
        hard = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in hard.items()}
        report["arms"][key] = dict(
            config=asdict(config), config_sha256=digest(asdict(config)), loss=out.loss.item(),
            components=component_stats, raw_gradient=distribution(rg), shared_gradient=distribution(sg),
            gradient_cosines={name: dict(raw=cosine(a[0], b[0]), shared=cosine(a[1], b[1]))
                              for name, (a, b) in pairs.items()},
            zero_norm_cosine_semantics="null means undefined",
            time_score_gradient_max=rg[..., 15:21].abs().max().item(),
            position_tanh_saturation_rate=(raw[..., :9].tanh().abs() > .99).double().mean().item(),
            winner_counts=torch.bincount(out.metrics["J"].argmin(-1), minlength=6).tolist(),
            per_query=metrics, hard=hard,
            continuity_residuals={k: v.detach().tolist() for k, v in out.trajectory.diagnostics.items()})
    after = parameter_hash(model)
    if before != after:
        raise ValueError("model parameters changed")
    source = sorted({Path(m.__file__).resolve() for n, m in sys.modules.items()
                     if n.startswith("query_minco") and getattr(m, "__file__", "").endswith(".py")})
    root = Path(__file__).resolve().parents[3]
    fixture = dict(state=state.tolist(), depth_channel_values=[.1, 1.],
                   domain=domain.tolist(), boxes=[g.boxes.tolist() for g in geometry])
    report.update(parameters_unchanged=True, model_sha256_before=before, model_sha256_after=after,
                  commit_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  fixture=fixture, dataset_sha256=digest(fixture), checkpoint_sha256=None,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                  diversity_enabled=False, selector_enabled=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run(PhysicsV12Spec(**config))
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k: dict(loss=v["loss"], K_eff=v["per_query"]["K_eff"])
                      for k, v in result["arms"].items()}))
