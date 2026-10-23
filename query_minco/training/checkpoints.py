"""Exclusive full-state checkpoints. Restore never starts or resumes a Run."""
import os
import random
import numpy as np
import torch
from query_minco.inference.identity import state_sha
from .io import write, sha, read, require
from .run_spec import identity

def cpu_tree(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    if isinstance(v, dict):
        return {k: cpu_tree(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(cpu_tree(x) for x in v)
    return v

def save_checkpoint(run, model, optimizer, spec, step, name=None, scheduler_id="recipe.lr_at; no separate scheduler object"):
    filename = name or ("checkpoint_step%04d.pt" % step)
    p = run / filename
    npstate = np.random.get_state()
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
               numpy=dict(kind=npstate[0], keys=torch.from_numpy(npstate[1].astype(np.int64)),
                          position=npstate[2], has_gauss=npstate[3], cached_gaussian=npstate[4]),
               cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()] if spec["device"] == "cuda" else [])
    blob = dict(schema=1, state_dict=cpu_tree(model.state_dict()),
                optimizer_state_dict=cpu_tree(optimizer.state_dict()),
                completed_optimizer_updates=step, order_cursor=step, spec=spec,
                spec_sha256=identity(spec), weights_sha256=state_sha(model), rng=rng,
                next_lr_index=step, scheduler=scheduler_id)
    with p.open("xb") as f:
        torch.save(blob, f)
        f.flush(); os.fsync(f.fileno())
    receipt = dict(step=step, sha256=sha(p), weights_sha256=blob["weights_sha256"],
                   spec_sha256=identity(spec), complete=True)
    write(run / (filename + ".json"), receipt)
    return receipt

def load_checkpoint(path, model, optimizer, expected_spec, restore_rng=False):
    receipt = read(str(path) + ".json")
    require(receipt["complete"] is True and sha(path) == receipt["sha256"], "CHECKPOINT_SEAL")
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    require(blob["spec"] == expected_spec and blob["spec_sha256"] == identity(expected_spec) ==
            receipt["spec_sha256"], "CHECKPOINT_SPEC")
    require(blob["completed_optimizer_updates"] == blob["order_cursor"] == receipt["step"], "CHECKPOINT_CURSOR")
    model.load_state_dict(blob["state_dict"], strict=True)
    require(state_sha(model) == blob["weights_sha256"] == receipt["weights_sha256"], "CHECKPOINT_MODEL_SHA")
    optimizer.load_state_dict(blob["optimizer_state_dict"])
    if restore_rng:
        rng = blob["rng"]; n = rng["numpy"]
        random.setstate(rng["python"]); torch.set_rng_state(rng["torch"])
        np.random.set_state((n["kind"], n["keys"].numpy().astype(np.uint32), n["position"],
                             n["has_gauss"], n["cached_gaussian"]))
        if rng["cuda"]:
            require(expected_spec["device"] == "cuda", "RESTORE_DEVICE")
            torch.cuda.set_rng_state_all(rng["cuda"])
    return blob
