"""Thin, zero-update qualification entry. No trainer, optimizer or ROS entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1_2_gate1.k1_interface import (
    make_planner, make_objective, identity)
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
from query_minco.evaluation.validation import numerical_reference, evaluate

OUT = Path(os.environ.get("G1_K1_AUDIT_ROOT", ROOT / "research_outputs/g1_k1_current_baseline_20261110_v2"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def arr(x):
    return x.detach().cpu().numpy()


def prepare():
    """New binding only; every changed pinned source must match the pre-edit SHA."""
    entry = json.loads((OUT / "ENTRY_AND_WHITELIST.json").read_text())
    idx = json.loads((ROOT / "research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json").read_text())
    rows = []
    for row in idx:
        if not row["data_eligible"]:
            continue
        previous = ROOT / "research_outputs/bug01_hard_duration_contract_20261110_01/rebound" / row["sample_id"] / "SAMPLE_MANIFEST.json"
        m = json.loads(previous.read_text())
        replacements = {}
        for name, expected in list(m["files_sha256"].items()):
            actual = sha(name)
            if actual != expected:
                rel = str(Path(name).relative_to(ROOT))
                if rel not in entry["allowed_modified"] or entry["source_before"][name] != expected:
                    raise ValueError("UNAUTHORIZED_SOURCE_DRIFT " + name)
                m["files_sha256"][name] = actual
                replacements[name] = dict(before=expected, after=actual)
        dest = OUT / "bindings" / row["sample_id"]
        dest.mkdir(parents=True)
        state = dest / "STATE.json"
        state.write_bytes((previous.parent / "STATE.json").read_bytes())
        m["files_sha256"][str(state)] = sha(state)
        m["g1_interface_binding"] = dict(**identity(), previous_manifest=str(previous),
                                        previous_sha256=sha(previous), source_revisions=replacements)
        path = dest / "SAMPLE_MANIFEST.json"
        write(path, m)
        # Reuse unchanged production loader, including every file/geometry identity check.
        batch = load_e1_bound(path, sha(path))
        assert batch.manifest["scene_identity"] == m["scene_identity"]
        rows.append(dict(sample_id=row["sample_id"], split=row["split"],
                         manifest=str(path), manifest_sha256=sha(path),
                         previous=str(previous), previous_sha256=sha(previous),
                         original_p0c_manifest=row["p0c_manifest_path"],
                         original_p0c_sha256=row["p0c_manifest_sha256"],
                         source_revisions=replacements, state_sha256=sha(state)))
    assert len(rows) == 21 and sum(x["split"] == "train" for x in rows) == 13
    write(OUT / "BINDING_INDEX.json", rows)
    method = ROOT / "query_minco/methods/k1_interface_v1.json"
    write(OUT / "RUN_IDENTITY.json", dict(**identity(), method_manifest=str(method),
          method_manifest_sha256=sha(method), source_after={
              n:sha(ROOT / n) for n in entry["allowed_modified"] + [
                  "query_minco/training/physics_only_v1_2_gate1/k1_interface.py"]},
          seed=20261104, dtype="float32", device="cpu", input_contract=["depth+mask", "state12"],
          training_updates=0, teacher_loaded=False, run_stage="zero_update"))
    print(json.dumps(dict(stage="prepare", status="PASS", bound_tasks=len(rows))))


def load_rows():
    run = json.loads((OUT / "RUN_IDENTITY.json").read_text())
    if sha(run["method_manifest"]) != run["method_manifest_sha256"]:
        raise ValueError("METHOD_DRIFT")
    for name, expected in run["source_after"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("SOURCE_DRIFT " + name)
    return json.loads((OUT / "BINDING_INDEX.json").read_text())


def numeric(result):
    p, tr = result.parameters, result.trajectory
    assert tr.coeffs.shape == (1, 1, 3, 3, 6)
    c = arr(tr.coeffs)[0, 0].astype(float); dt = arr(tr.durations)[0, 0]
    ref = numerical_reference(arr(p.head_pva)[0, 0], arr(p.tail_pva)[0, 0], arr(p.inner_pts)[0, 0], dt)
    errors = [max(float(np.abs(evaluate(a, np.linspace(0, t, 65), d) -
                              evaluate(b, np.linspace(0, t, 65), d)).max())
                  for a, b, t in zip(c, ref, dt)) for d in (0, 1, 2)]
    assert all(x < tol for x, tol in zip(errors, [.002, .01, .05]))
    assert tr.solve_valid.all()
    assert torch.equal(tr.durations, torch.full_like(tr.durations, 2/3))
    return errors


class ResourceSample:
    def __init__(self):
        self.stop = threading.Event(); self.peak = 0; self.samples = 0
    def sample(self):
        while not self.stop.is_set():
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    self.peak = max(self.peak, int(line.split()[1]) * 1024)
            self.samples += 1
            self.stop.wait(.005)
    def __enter__(self):
        self.wall = time.monotonic(); self.cpu = time.process_time()
        self.thread = threading.Thread(target=self.sample, daemon=True); self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join()
        self.record = dict(wall_s=time.monotonic()-self.wall, cpu_s=time.process_time()-self.cpu,
                           sampled_self_peak_rss_bytes=self.peak, sampling_s=.005,
                           samples=self.samples, process_hwm_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def zero():
    for name in ("K1_INITIAL_STATE.npz", "K1_ZERO_UPDATE_ARRAYS.npz", "K1_ZERO_UPDATE.json"):
        if (OUT / name).exists():
            raise FileExistsError("ZERO_UPDATE_OUTPUT_OCCUPIED " + str(OUT / name))
    rows = load_rows()
    torch.set_num_threads(1); torch.manual_seed(20261104)
    model = make_planner().cpu().eval(); obj = make_objective()
    before = state_hash(model); reports = []; arrays = {}
    np.savez(OUT / "K1_INITIAL_STATE.npz", **{n:arr(p) for n, p in model.state_dict().items()})
    for row in rows:
        with ResourceSample() as monitor:
            batch = load_e1_bound(row["manifest"], row["manifest_sha256"])
            model.zero_grad(set_to_none=True)
            raw = model(batch.depth, batch.state); raw.retain_grad()
            assert raw.shape == (1, 1, 21) and model.query_decoder.learned_queries.shape == (1, 128)
            result = obj(raw, batch)
            assert torch.isfinite(result.loss)
            result.loss.backward()
            errors = numeric(result)
            assert torch.equal(result.parameters.head_pva[0, 0],
                torch.stack([torch.zeros(3), batch.state[0, :3], batch.state[0, 3:6]]))
            groups = {}
            for name, mod in [("depth", model.depth_encoder), ("state", model.state_encoder),
                              ("decoder", model.query_decoder), ("head", model.trajectory_head)]:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())
                norm = float(torch.cat([p.grad.flatten() for p in mod.parameters()]).norm())
                assert norm > 0
                groups[name] = norm
            assert raw.grad[..., :15].norm() > 0 and raw.grad[..., 15:].count_nonzero() == 0
            assert model.trajectory_head.mlp[-1].weight.grad[15:].count_nonzero() == 0
            assert model.trajectory_head.mlp[-1].bias.grad[15:].count_nonzero() == 0
            assert torch.isfinite(raw.grad).all() and state_hash(model) == before
            terms = result.metrics
            direct = (terms["C_viol"]+terms["F"]+terms["G_task_train"]+
                      .1*terms["C_shape"]+.025*terms["P"]+.1*terms["S"]).mean()
            torch.testing.assert_close(result.loss, direct, rtol=1e-6, atol=1e-6)
            assert torch.equal(terms["pi"], torch.ones(1, 1)) and torch.equal(terms["K_eff"], torch.ones(1))
            hard = validate_polynomial(arr(result.trajectory.coeffs)[0, 0],
                    arr(result.trajectory.durations)[0, 0], batch.geometry[0], arr(batch.state)[0, 6:9])
            for name, x in [("raw", raw), ("raw_grad", raw.grad), ("coeffs", result.trajectory.coeffs),
                            ("loss", result.loss), ("state", batch.state)]:
                arrays[row["sample_id"]+"::"+name] = arr(x)
            for name, x in terms.items():
                if isinstance(x, torch.Tensor):
                    arrays[row["sample_id"]+"::metric::"+name] = arr(x)
            reports.append(dict(sample_id=row["sample_id"], split=row["split"], raw_shape=list(raw.shape),
                loss=float(result.loss), module_gradient_norms=groups, raw15_21_nonzero=0,
                raw0_15_gradient=arr(raw.grad[0, 0, :15]).tolist(), independent_minco_pva_errors=errors,
                outside_points=sum(int(x["outside_mask"].sum()) for x in result.training_geometry_records),
                hard=hard, weight_sha_before=before, weight_sha_after=state_hash(model)))
        reports[-1]["resources"] = monitor.record
        print("TASK_PASS", row["sample_id"], "loss", reports[-1]["loss"], flush=True)
    np.savez(OUT / "K1_ZERO_UPDATE_ARRAYS.npz", **arrays)
    summary = dict(status="PASS", method=identity(), tasks=reports, task_count=len(reports),
        parameters=sum(p.numel() for p in model.parameters()),
        weight_sha_before=before, weight_sha_after=state_hash(model), optimizer_updates=0,
        initial_hard_passed=sum(x["hard"]["passed"] for x in reports),
        note="Qualification of random initialization; not planning quality or training success.")
    write(OUT / "K1_ZERO_UPDATE.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "tasks"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "zero"], required=True)
    args = parser.parse_args()
    with patch.object(torch.optim.Optimizer, "__init__", side_effect=RuntimeError("NO_OPTIMIZER_ALLOWED")), \
         patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("NO_UPDATES_ALLOWED")):
        if args.stage == "prepare":
            prepare()
        else:
            zero()


if __name__ == "__main__":
    main()
