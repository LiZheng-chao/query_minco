"""P1-only synthetic shape/gradient/timing evidence; no optimizer or trajectory."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from query_minco import QueryMincoPlanner

SEED = 42


def inputs(batch, device):
    depth = torch.rand(batch, 2, 128, 160, device=device)
    depth[:, 1] = torch.randint(0, 2, (batch, 128, 160), device=device).float()
    state = torch.randn(batch, 12, device=device)
    state[:, 9:12] = torch.nn.functional.normalize(state[:, 9:12], dim=1)
    return depth, state


def gradients(model):
    groups = {
        "DepthEncoder": list(model.depth_encoder.parameters()),
        "StateEncoder": list(model.state_encoder.parameters()),
        "learned_queries": [model.query_decoder.learned_queries],
        "QueryDecoder": [p for n, p in model.query_decoder.named_parameters() if n != "learned_queries"],
        "TrajectoryHead": list(model.trajectory_head.parameters()),
    }
    result = {}
    for name, params in groups.items():
        gs = [p.grad for p in params]
        result[name] = {
            "parameter_tensors": len(gs), "with_gradient": sum(g is not None for g in gs),
            "all_finite": all(g is not None and bool(torch.isfinite(g).all()) for g in gs),
            "nonzero_tensors": sum(g is not None and bool(g.abs().sum() > 0) for g in gs),
            "l2_norm": sum(float(g.float().square().sum()) for g in gs if g is not None) ** 0.5}
    return result


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_device(device, iterations):
    torch.manual_seed(SEED)
    model = QueryMincoPlanner().to(device).eval()
    depth, state = inputs(2, device)
    raw, diag = model(depth, state, return_diagnostics=True)
    with torch.no_grad():
        plain = model(depth, state)
    shape_info = {k: list(v.shape) for k, v in diag.items() if isinstance(v, torch.Tensor)}
    difference = float((raw.detach() - plain).abs().max())
    loss = (raw - torch.randn_like(raw)).square().mean()
    loss.backward()
    grad = gradients(model)
    assert all(x["all_finite"] and x["l2_norm"] > 0 and x["with_gradient"] == x["parameter_tensors"]
               for x in grad.values())
    result = {"raw_shape": list(raw.shape), "diagnostic_shapes": shape_info,
              "default_debug_max_abs_difference": difference,
              "loss": float(loss.detach()), "gradients": grad}
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            amp_raw = model(depth, state)
            amp_loss = amp_raw.float().square().mean()
        amp_loss.backward()
        result["amp"] = {"dtype": str(amp_raw.dtype), "finite": bool(torch.isfinite(amp_raw).all()),
                         "gradients": gradients(model)}
        assert result["amp"]["finite"]
        assert all(x["all_finite"] and x["l2_norm"] > 0 for x in result["amp"]["gradients"].values())
        model.zero_grad(set_to_none=True)
    # Representative B=1, eval + inference_mode, five warmups, validation included.
    depth, state = inputs(1, device)
    result["timing_ms"] = {}
    with torch.inference_mode():
        for diagnostics in (False, True):
            for _ in range(5):
                model(depth, state, return_diagnostics=diagnostics)
            synchronize(device)
            elapsed = []
            for _ in range(iterations):
                synchronize(device)
                start = time.perf_counter()
                output = model(depth, state, return_diagnostics=diagnostics)
                synchronize(device)
                elapsed.append((time.perf_counter() - start) * 1000)
            ordered = sorted(elapsed)
            result["timing_ms"]["diagnostics" if diagnostics else "default"] = {
                "mean": statistics.mean(elapsed), "median": statistics.median(elapsed),
                "p95": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    cpu_name = platform.processor()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_name = line.split(":", 1)[1].strip()
            break
    result = {
        "seed": SEED, "torch": torch.__version__, "python": platform.python_version(),
        "cpu": cpu_name, "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
        "parameter_count": sum(p.numel() for p in QueryMincoPlanner().parameters()),
        "timing_protocol": {"batch": 1, "warmups": 5, "iterations": args.iterations,
                            "mode": "eval + inference_mode; input validation included"},
        "cpu_results": run_device("cpu", args.iterations)}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["cuda_results"] = run_device("cuda", args.iterations)
    if args.output:
        # Evidence is confined to the new method directory; never overwrite previous runs.
        args.output.resolve().relative_to(Path(__file__).resolve().parents[1])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
