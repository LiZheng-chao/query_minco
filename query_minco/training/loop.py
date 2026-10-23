"""Run lifecycle arithmetic only; original objective/MINCO/matching/Loss are called directly."""
import time
import hashlib
import traceback
import numpy as np
import torch
from query_minco.inference.identity import state_sha
from . import objective
from .recipe import lr_at, optimizer
from .initialization import initialize, head_rows, rows_unchanged
from .checkpoints import save_checkpoint
from .io import write, require, digest
from .run_spec import PROFILES, components
from .qualification import qualify

def update(model, opt, samples, index, rows, context, objective_forward=objective.forward):
    model.train()
    opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:
        group["lr"] = lr_at(index)
    started = time.perf_counter()
    loss, matches, raw, out, costs = objective_forward(model, samples, "B_set", next(model.parameters()).device)
    context.update(raw=raw, output=out, ids=[s.sample_id for s in samples])
    require(raw.shape == (len(samples), 6, 21) and raw.dtype == loss.dtype == torch.float32, "OUTPUT_CONTRACT")
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(out.trajectory.coeffs).all()) and
            bool(out.trajectory.solve_valid.all()), "NONFINITE_LOSS_OR_MINCO")
    out.parameters.durations.retain_grad()
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "GRADIENT")
    require(raw.grad is not None and bool(torch.isfinite(raw.grad).all()) and
            bool((raw.grad[:, :, 19:21] == 0).all()), "UNSUPERVISED_RAW_GRADIENT")
    require(out.parameters.durations.grad is not None and
            bool(torch.isfinite(out.parameters.durations.grad).all()), "MINCO_DURATION_GRADIENT")
    for b, m in enumerate(matches):
        unmatched = sorted(set(range(6)) - set(m["queries"]))
        require(bool((raw.grad[b, unmatched] == 0).all()), "UNMATCHED_DIRECT_GRADIENT")
    for n, p in model.named_parameters():
        if n in rows:
            require(bool((p.grad[19:21] == 0).all()), "UNSUPERVISED_HEAD_GRADIENT")
    qgrad = model.query_decoder.learned_queries.grad.detach().norm(dim=-1).cpu().tolist()
    module_grads = {n: float(torch.sqrt(sum(p.grad.detach().square().sum() for p in mod.parameters())).cpu())
                    for n, mod in [("CNN", model.depth_encoder), ("state", model.state_encoder),
                                   ("decoder", model.query_decoder), ("head", model.trajectory_head)]}
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10).detach().cpu())
    require(np.isfinite(norm), "CLIP_NORM")
    context["optimizer_step_started"] = True
    opt.step()
    context["optimizer_step_started"] = False
    # Set immediately after step, before any further check or serialization can fail.
    context["completed_optimizer_updates"] = index + 1
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "NONFINITE_PARAMETER_AFTER_STEP")
    require(rows_unchanged(model, rows), "UNSUPERVISED_HEAD_CHANGED")
    if raw.is_cuda:
        torch.cuda.synchronize()
    record = dict(step=index + 1, ids=[s.sample_id for s in samples], M=[len(s.teacher_coeffs) for s in samples],
                  lr=lr_at(index), loss=float(loss.detach().cpu()), matches=matches,
                  costs=[c.detach().cpu().tolist() for c in costs],
                  raw_query_gradient_norms=raw.grad.detach().norm(dim=-1).cpu().tolist(),
                  learned_query_gradient_norms=qgrad, module_gradient_norms=module_grads,
                  duration_gradient_max=out.parameters.durations.grad.detach().abs().flatten(0, 1).max(0).values.cpu().tolist(),
                  global_gradient_norm_before_clip=norm, score_feasibility_rows_unchanged=True,
                  raw_sha256=hashlib.sha256(raw.detach().cpu().numpy().tobytes()).hexdigest(),
                  elapsed_s=time.perf_counter() - started)
    return record, raw

def execute(spec, run):
    import json
    completed = 0; context = {}; model = None; opt = None
    try:
        ds, order, data_gate = qualify(spec)
        model = initialize(spec); rows = head_rows(model); binding = components(spec["training_method_id"]); opt = binding.optimizer_factory(model)
        require(not opt.state, "OPTIMIZER_NOT_FRESH")
        cache = {s.sample_id: s for s in ds.load(list(ds.rows))}
        write(run / "WORKER_DATA_GATE.json", data_gate)
        save_checkpoint(run, model, opt, spec, 0)
        with (run / "steps.jsonl").open("x") as log:
            exposures = 0
            previous = {}
            for index in range(PROFILES[spec["profile"]]["updates"]):
                ids, roles = order.batch(index)
                rec, raw = update(model, opt, [cache[sid] for sid in ids], index, rows, context, binding.objective)
                completed = context["completed_optimizer_updates"]
                exposures += sum(rec["M"])
                for match in rec["matches"]:
                    mapping = {t: q for q, t in zip(match["queries"], match["teachers"])}
                    switches = []
                    for teacher, query in mapping.items():
                        key = (match["sample_id"], teacher)
                        if key in previous and previous[key] != query:
                            switches.append(dict(teacher=teacher, previous_query=previous[key], query=query))
                        previous[key] = query
                    match["switches"] = switches
                rec.update(roles=roles, task_exposures_completed=completed * len(ids),
                           teacher_exposures_completed=exposures)
                # Exact raw arrays, exclusive per-step file; no overwrite after interruption.
                with (run / ("raw_step%04d.npy" % completed)).open("xb") as f:
                    np.save(f, raw.detach().cpu().numpy(), allow_pickle=False)
                log.write(json.dumps(rec, allow_nan=False) + "\n"); log.flush()
                if completed in PROFILES[spec["profile"]]["checkpoints"]:
                    save_checkpoint(run, model, opt, spec, completed)
                print("UPDATE", completed, "loss", rec["loss"], "lr", rec["lr"], flush=True)
        require(completed == order.summary["updates"] and exposures == order.summary["teacher_exposures"], "EXACT_BUDGET")
        write(run / "WORKER_RESULT.json", dict(status="WORKER_FINISHED_AWAITING_AUDIT",
              completed_optimizer_updates=completed, weights_sha256=state_sha(model),
              task_exposures=order.summary["task_exposures"], teacher_exposures=exposures,
              formal_evaluations=0, profile=spec["profile"]))
    except BaseException as error:
        completed = max(completed, context.get("completed_optimizer_updates", 0))
        write(run / "ERROR.json", dict(error=repr(error), traceback=traceback.format_exc(),
              completed_optimizer_updates=completed, possible_partial_update=bool(context.get("optimizer_step_started", False)),
              ids=context.get("ids"), retry=False, resume=False))
        if model is not None:
            try:
                save_checkpoint(run, model, opt, spec, completed, "FAILED_STATE_NOT_A_PRESET_CHECKPOINT.pt")
            except BaseException as save_error:
                write(run / "FAILURE_SAVE_ERROR.json", dict(error=repr(save_error)))
        arrays = {}
        if context.get("raw") is not None:
            arrays["raw"] = context["raw"].detach().cpu().numpy()
        if context.get("output") is not None:
            arrays["coefficients"] = context["output"].trajectory.coeffs.detach().cpu().numpy()
            arrays["durations"] = context["output"].trajectory.durations.detach().cpu().numpy()
        with (run / "FAILURE_ARRAYS.npz").open("xb") as f:
            np.savez_compressed(f, **arrays)
        raise
