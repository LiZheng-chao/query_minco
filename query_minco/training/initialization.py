"""Strict explicit initialization; never imports an old trainer or inherits its optimizer."""
import random
import numpy as np
import torch
from query_minco.inference.identity import state_sha
from .io import artifact, require, read
from .run_spec import components

def configure(spec):
    torch.set_num_threads(spec["threads"])
    torch.use_deterministic_algorithms(True)
    require(not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled(), "AUTOCAST_DISABLED_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(spec["device"] != "cuda" or torch.cuda.is_available(), "CUDA_UNAVAILABLE_NO_FALLBACK")
    if spec["device"] == "cuda":
        import os
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS_WORKSPACE_CONFIG")
    random.seed(spec["seed"]); np.random.seed(spec["seed"]); torch.manual_seed(spec["seed"])
    if spec["device"] == "cuda":
        torch.cuda.manual_seed_all(spec["seed"])

def head_rows(model):
    return {n: p.detach()[19:21].clone() for n, p in model.named_parameters()
            if n.startswith("trajectory_head.") and p.shape[0] == 21}

def rows_unchanged(model, rows):
    return len(rows) == 2 and all(torch.equal(dict(model.named_parameters())[n].detach()[19:21], v)
                                  for n, v in rows.items())

def initialize(spec):
    configure(spec)
    blob = torch.load(str(artifact(spec["initialization"]["artifact"])), map_location="cpu", weights_only=True)
    sd = blob["state_dict"] if "state_dict" in blob else blob
    if spec["profile"] == "b2c1_reference_v1":
        origin = spec["initialization"]["origin"]
        receipt = read(artifact(origin["identity_receipt"]))
        artifact(origin["original_checkpoint"])
        require(origin["seed"] == spec["seed"] and origin["step"] == 5760 and
                receipt["original_sha256"] == origin["original_checkpoint"]["sha256"] and
                receipt["transport_sha256"] == spec["initialization"]["artifact"]["sha256"] and
                receipt["original_weights_sha256"] == spec["initialization"]["weights_sha256"] and
                receipt["optimizer_state_restored"] is False, "B1_TRANSPORT_LINEAGE")
    model = components(spec["training_method_id"]).model_factory()
    result = model.load_state_dict(sd, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, "STRICT_LOAD")
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters()), "INITIALIZATION_FP32")
    require(state_sha(model) == spec["initialization"]["weights_sha256"], "INITIALIZATION_STATE_SHA")
    return model.to(spec["device"])
