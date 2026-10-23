"""Content audit. Reads artifacts only: no forward, optimizer step, repair, or reseal."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from query_minco.inference.identity import state_sha
from .io import require, read, sha, digest
from .run_spec import PROFILES, identity
from .recipe import lr_at
from .io import artifact

def audit_run(directory, candidate=False):
    run = Path(directory)
    require(not any((run / n).exists() for n in ("ERROR.json", "STOP.json", "LAUNCH_ERROR.json")), "FAILED_RUN")
    spec = read(run / "RUN_SPEC.json")
    cfg = PROFILES[spec["profile"]]
    artifact(spec["dataset"]); artifact(spec["order"]); artifact(spec["initialization"]["artifact"])
    require(read(run / "PROTECTION_BEFORE.json")["files"] == read(run / "PROTECTION_AFTER.json")["files"],
            "EXTERNAL_PROTECTION_LIST_CHANGED")
    require(read(run / "PROCESS_EXIT.json")["returncode"] == 0, "WORKER_EXIT")
    for stage in ("BEFORE", "AFTER"):
        require(read(run / ("PROTECTION_" + stage + ".json"))["passed"] is True, "EXTERNAL_PROTECTION")
    require(read(run / "ISOLATION.json")["passed"] is True, "ISOLATION")
    result = read(run / "WORKER_RESULT.json")
    require(result["status"] == "WORKER_FINISHED_AWAITING_AUDIT" and
            result["completed_optimizer_updates"] == cfg["updates"], "UPDATE_BUDGET")
    require(result["formal_evaluations"] == 0, "UNAUTHORIZED_EVALUATION")
    with np.load(spec["order"]["path"], allow_pickle=False) as order:
        ids, roles = order["task_ids"], order["roles"]
    rows = {r["id"]: r for r in read(spec["dataset"]["path"])["rows"]}
    records = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    require(len(records) == cfg["updates"], "LOG_UPDATE_COUNT")
    teacher_count = 0
    for i, record in enumerate(records):
        require(record["step"] == i + 1 and record["ids"] == ids[i].tolist() and
                record["roles"] == roles[i].tolist() and record["M"] == [rows[s]["M"] for s in record["ids"]], "LOG_ORDER")
        require(record["lr"] == lr_at(i) and record["score_feasibility_rows_unchanged"], "LOG_RECIPE")
        for key in ("lr", "loss", "elapsed_s", "global_gradient_norm_before_clip"):
            require(np.isfinite(record[key]), "LOG_FINITE")
        require(np.isfinite(record["raw_query_gradient_norms"]).all() and
                np.isfinite(record["duration_gradient_max"]).all(), "LOG_GRADIENT_FINITE")
        losses = []
        for sid, M, match, matrix in zip(record["ids"], record["M"], record["matches"], record["costs"]):
            c = np.asarray(matrix); q, t = match["queries"], match["teachers"]
            require(c.shape == (6, M) and np.isfinite(c).all() and match["sample_id"] == sid, "MATCH_COST")
            require(len(q) == len(t) == M and len(set(q)) == M and set(t) == set(range(M)) and
                    all(0 <= x < 6 for x in q), "MATCH_BIJECTION")
            oq, ot = linear_sum_assignment(c)
            actual = c[q, t].mean()
            require(np.isclose(actual, c[oq, ot].mean(), rtol=1e-6, atol=1e-7), "HUNGARIAN_OPTIMUM")
            losses.append(actual)
        require(np.isclose(record["loss"], np.mean(losses), rtol=1e-5, atol=1e-7), "LOSS_RECONSTRUCTION")
        raw = np.load(run / ("raw_step%04d.npy" % (i + 1)), allow_pickle=False)
        require(raw.dtype == np.float32 and raw.shape == (cfg["batch_size"], 6, 21) and np.isfinite(raw).all(), "RAW_RECORD")
        if spec["profile"] == "b2c1_reference_v1" or "raw_sha256" in record:  # Earlier synthetic fixtures predate this extra field.
            require(record["raw_sha256"] == __import__("hashlib").sha256(raw.tobytes()).hexdigest(), "RAW_LOG_SHA")
        teacher_count += sum(record["M"])
        require(record["task_exposures_completed"] == (i + 1) * cfg["batch_size"] and
                record["teacher_exposures_completed"] == teacher_count, "EXPOSURE_LOG")
    require(result["teacher_exposures"] == teacher_count and
            result["task_exposures"] == cfg["updates"] * cfg["batch_size"], "EXPOSURE_RESULT")
    expected = ["checkpoint_step%04d.pt" % s for s in cfg["checkpoints"]]
    require(sorted(p.name for p in run.glob("checkpoint_step*.pt")) == expected, "CHECKPOINT_SET")
    initial_rows = None
    final = None
    for step, name in zip(cfg["checkpoints"], expected):
        p = run / name; receipt = read(str(p) + ".json")
        require(receipt["complete"] and receipt["sha256"] == sha(p) and receipt["step"] == step, "CHECKPOINT_SEAL")
        blob = torch.load(str(p), map_location="cpu", weights_only=True)
        require(blob["spec"] == spec and blob["spec_sha256"] == receipt["spec_sha256"] == identity(spec), "CHECKPOINT_IDENTITY")
        require(blob["completed_optimizer_updates"] == blob["order_cursor"] == step, "CHECKPOINT_UPDATES")
        sd = blob["state_dict"]
        h = __import__("hashlib").sha256()
        for key, tensor in sorted(sd.items()):
            require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), "CHECKPOINT_FINITE")
            h.update(key.encode()); h.update(tensor.contiguous().numpy().tobytes())
        require(h.hexdigest() == blob["weights_sha256"] == receipt["weights_sha256"], "CHECKPOINT_STATE_SHA")
        head = {k: v[19:21] for k, v in sd.items() if k.startswith("trajectory_head.") and v.shape[0] == 21}
        require(len(head) == 2, "HEAD_ROWS")
        if step == 0:
            require(h.hexdigest() == spec["initialization"]["weights_sha256"], "PAIRED_INITIALIZATION")
            initial_rows = head
        require(all(torch.equal(v, initial_rows[k]) for k, v in head.items()), "UNSUPERVISED_ROWS_CHANGED")
        opt = blob["optimizer_state_dict"]
        require(len(opt["param_groups"]) == 1, "OPTIMIZER_GROUPS")
        g = opt["param_groups"][0]
        require(g["weight_decay"] == 0 and tuple(g["betas"]) == (.9, .999) and g["eps"] == 1e-8 and
                g["lr"] == (2e-4 if step == 0 else lr_at(step - 1)), "OPTIMIZER_RECIPE")
        require(len(opt["state"]) == (0 if step == 0 else len(g["params"])), "OPTIMIZER_STATE_COUNT")
        for value in opt["state"].values():
            require(int(value["step"]) == step and all(
                not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all()) for v in value.values()), "OPTIMIZER_STATE")
        require(set(blob["rng"]) == {"python", "numpy", "torch", "cuda"} and blob["next_lr_index"] == step, "RNG_AND_SCHEDULER")
        final = blob["weights_sha256"]
    require(final == result["weights_sha256"], "FINAL_IDENTITY")
    if not candidate:
        require(read(run / "RESULT.json")["status"] == "COMPLETED" and read(run / "CONTENT_AUDIT.json")["passed"], "NOT_COMPLETED")
        seal = read(run / "RUN_SEAL.json")
        actual = {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file() and p.name != "RUN_SEAL.json"}
        require(actual == seal["files"], "RUN_SEAL_OR_EXTRA_FILES")
    return dict(passed=True, profile=spec["profile"], completed_optimizer_updates=cfg["updates"],
                task_exposures=cfg["updates"] * cfg["batch_size"], teacher_exposures=teacher_count,
                weights_sha256=final, formal=spec["profile"] == "b2c1_reference_v1")
