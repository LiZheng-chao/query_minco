"""CPU network forward/backward diagnostics. No optimizer is constructed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
from dataclasses import replace

import torch

from query_minco.models import QueryMincoPlanner
from query_minco.training.io import digest
from .costs import SyntheticGeometry
from .objective import PhysicsOnlyObjective
from .spec import PhysicsSpec


def stats(x):
    x = x.detach().float().reshape(-1)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("nonfinite diagnostic")
    return dict(min=x.min().item(), median=x.median().item(), max=x.max().item(),
                mean=x.mean().item(), rms=x.square().mean().sqrt().item(),
                l2=x.norm().item(), nonzero=int((x != 0).sum()), count=x.numel())


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run():
    torch.manual_seed(20261104)
    torch.set_num_threads(1)
    model = QueryMincoPlanner().cpu().eval()
    before = parameter_hash(model)
    state = torch.zeros(2, 12)
    state[:, 0] = .8
    state[:, 6] = torch.tensor([2.4, 6.])
    state[:, 11] = -1.
    depth = torch.zeros(2, 2, 128, 160)
    depth[:, 0] = .1
    depth[:, 1] = 1.
    # Reproducible empty and single-obstacle diagnostics, not rendered forest data.
    domain = torch.tensor([[0., -4., -2.], [10., 4., 2.]])
    geometry = [
        SyntheticGeometry(torch.empty(0, 2, 3), domain, True),
        SyntheticGeometry(torch.tensor([[[.9, -.3, -.3], [1.2, .3, .3]]]), domain, False)]
    spec = PhysicsSpec()
    # Use shared visual encoder weights to measure gradients reaching the backbone.
    shared = model.depth_encoder.d1[0].weight
    result = dict(seed=20261104, torch_version=torch.__version__, device="cpu",
                  qualification="SYNTHETIC_TEST_ONLY", optimizer_steps=0,
                  data_note="constant depth channels; geometry synthetic, no renderer consistency claim",
                  shared_parameter="depth_encoder.d1.0.weight", modes={})
    for mode in ("mean", "softmin"):
        raw = model(depth, state)
        out = PhysicsOnlyObjective(replace(spec, aggregation=mode))(raw, state, geometry)
        components = {}
        scales = dict(C=spec.wc, F=spec.wf, G_task=spec.wg,
                      Progress=spec.wg * spec.eta_progress, S=spec.wj)
        for name, scale in scales.items():
            value = out.metrics[name]
            rg, sg = torch.autograd.grad(value.mean(), (raw, shared), retain_graph=True)
            components[name] = dict(unweighted=stats(value), scaled=stats(scale * value),
                                    raw_gradient=stats(rg), shared_gradient=stats(sg),
                                    scaled_raw_gradient=stats(scale * rg),
                                    scaled_shared_gradient=stats(scale * sg))
        rg, sg = torch.autograd.grad(out.loss, (raw, shared))
        assert torch.count_nonzero(rg[..., 15:21]) == 0
        result["modes"][mode] = dict(
            loss=out.loss.item(), components=components, raw_gradient=stats(rg),
            shared_gradient=stats(sg), max_time_score_gradient=rg[..., 15:21].abs().max().item(),
            per_query={k: v.detach().tolist() for k, v in out.metrics.items()},
            max_continuity_residual={k: v.max().item() for k, v in out.trajectory.diagnostics.items()})
    after = parameter_hash(model)
    assert before == after, "zero-update contract violated"
    result.update(parameters_unchanged=True, model_sha256_before=before, model_sha256_after=after)
    package = Path(__file__).resolve().parents[1]
    source_files = sorted({Path(module.__file__).resolve()
                           for name, module in sys.modules.items()
                           if name.startswith("query_minco") and
                           getattr(module, "__file__", "").endswith(".py")})
    fixture = dict(state=state.tolist(), depth_channel_values=[.1, 1.],
                   boxes=[g.boxes.tolist() for g in geometry],
                   domain=domain.tolist(), empty_certified=[True, False])
    result["bindings"] = dict(
        commit_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        method_id=spec.method_id, config=asdict(spec), config_sha256=digest(asdict(spec)),
        dataset_sha256=digest(fixture), fixture=fixture, checkpoint_sha256=None,
        initialization="random seed 20261104, model state bound above",
        source_sha256={str(p.relative_to(package.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_files})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New JSON file; never overwrite existing results.")
    args = parser.parse_args()
    report = run()
    with open(args.output, "x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({m: dict(loss=v["loss"], K_eff=v["per_query"]["K_eff"])
                      for m, v in report["modes"].items()}))
