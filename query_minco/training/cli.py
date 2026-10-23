"""Explicit plan/preflight/train/audit. No automatic escalation from planning to training."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import numpy as np
import scipy
from .io import read, write, require, sha
from .run_spec import load_spec, identity, PROFILES, run_path, authorize
from .qualification import qualify

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "train"):
        sp = sub.add_parser(name); sp.add_argument("--config", required=True)
        if name == "preflight":
            sp.add_argument("--output", required=True)
        if name == "train":
            sp.add_argument("--authorization", required=True); sp.add_argument("--preflight", required=True)
    sp = sub.add_parser("audit"); sp.add_argument("--run-dir", required=True)
    sp = sub.add_parser("_worker"); sp.add_argument("--config", required=True); sp.add_argument("--run-dir", required=True)
    sp.add_argument("--action", choices=["preflight", "train"], required=True)
    sp.add_argument("--authorization"); sp.add_argument("--preflight")
    a = p.parse_args(argv)
    if a.command == "audit":
        from .audit import audit_run
        result = audit_run(a.run_dir)
    else:
        spec = load_spec(a.config)
        if a.command == "plan":
            _, _, gate = qualify(spec)
            result = dict(spec_sha256=identity(spec), profile=PROFILES[spec["profile"]], data=gate,
                          run_dir=str(run_path(spec)), training_started=False,
                          components=__import__("query_minco.registry", fromlist=["describe"]).describe(spec["reference_method_id"])["component_ids"],
                          authorization_template=dict(schema=1,
                            scope="FORMAL_TRAINING" if spec["profile"] == "b2c1_reference_v1" else "DISPOSABLE_SYNTHETIC_ONLY",
                            run_id=spec["run_id"], spec_sha256=identity(spec),
                            max_updates=PROFILES[spec["profile"]]["updates"], authorized_by="", approved=False))
        elif a.command == "_worker":
            from .isolation import worker_probe
            from .initialization import initialize, configure
            run = Path(a.run_dir)
            worker_probe(run, spec)
            configure(spec)
            if a.action == "train":
                require(a.authorization is not None and a.preflight is not None, "WORKER_TRAIN_AUTHORIZATION")
                authorize(spec, a.authorization)
                receipt = read(a.preflight)
                require(receipt["passed"] and receipt["spec_sha256"] == identity(spec), "WORKER_PREFLIGHT")
                require(read(a.preflight + ".seal.json")["sha256"] == sha(a.preflight), "WORKER_PREFLIGHT_SEAL")
                require(receipt.get("gradient_gate", {}).get("passed") is True, "PREFLIGHT_GRADIENT_GATE")
                require(receipt["environment"]["executable"] == sys.executable and
                        receipt["environment"]["torch"] == torch.__version__, "RUNTIME_DRIFT")
            write(run / "ENVIRONMENT.json", dict(python=sys.version, executable=sys.executable,
                  torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  device=spec["device"], FP32=True, AMP=False, TF32=False, E1="explicit",
                  strict_deterministic=True, threads=spec["threads"]))
            if a.action == "preflight":
                from query_minco.inference.identity import state_sha
                ds, order, gate = qualify(spec)
                model = initialize(spec)
                from .run_spec import components
                from .initialization import head_rows, rows_unchanged
                before = state_sha(model); rows = head_rows(model)
                ids, roles = order.batch(0)
                model.train()
                loss, matches, raw, decoded, costs = components(spec["training_method_id"]).objective(
                    model, ds.load(ids), "B_set", spec["device"])
                decoded.parameters.durations.retain_grad()
                require(bool(torch.isfinite(loss)) and bool(decoded.trajectory.solve_valid.all()), "PREFLIGHT_MINCO_LOSS")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        "PREFLIGHT_MODEL_GRADIENT")
                require(raw.shape == (len(ids), 6, 21) and raw.dtype == torch.float32 and
                        raw.grad is not None and bool((raw.grad[:, :, 19:21] == 0).all()), "PREFLIGHT_RAW")
                require(decoded.parameters.durations.grad is not None and
                        bool(torch.isfinite(decoded.parameters.durations.grad).all()), "PREFLIGHT_TIME_GRADIENT")
                require(rows_unchanged(model, rows) and state_sha(model) == before, "PREFLIGHT_PARAMETER_UPDATE")
                for b, match in enumerate(matches):
                    un = sorted(set(range(6)) - set(match["queries"]))
                    require(bool((raw.grad[b, un] == 0).all()), "PREFLIGHT_UNMATCHED_GRADIENT")
                gradient_gate = dict(passed=True, loss=float(loss.detach().cpu()), ids=ids, roles=roles,
                    M=[ds.rows[sid]["M"] for sid in ids], matches=matches, FP32=True,
                    duration_gradient_finite=True, unsupervised_raw_gradient_zero=True,
                    parameters_unchanged=True, optimizer_created=False, E1="explicit")
                result = dict(passed=True, spec_sha256=identity(spec), data=gate,
                      initialized_weights_sha256=state_sha(model), optimizer_updates=0, forward_calls=1,
                      gradient_gate=gradient_gate,
                      scope="data/code/initialization/isolation and one disposable batch forward/backward; no optimizer",
                      environment=read(run / "ENVIRONMENT.json"))
                write(run / "PREFLIGHT.json", result)
            else:
                from .loop import execute
                execute(spec, run)
                result = dict(worker_finished=True)
        else:
            from .isolation import launch
            if a.command == "train":
                authorize(spec, a.authorization)
                receipt = read(a.preflight)
                require(receipt["passed"] and receipt["spec_sha256"] == identity(spec), "PREFLIGHT_REQUIRED")
                result = launch(spec, a.config, run_path(spec), "train", a.authorization, a.preflight)
            else:
                dest = Path(a.output)
                require(dest.is_absolute() and dest.parent == Path(spec["output_root"]) and dest != run_path(spec),
                        "PREFLIGHT_OUTPUT_PATH")
                result = launch(spec, a.config, dest, "preflight")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
