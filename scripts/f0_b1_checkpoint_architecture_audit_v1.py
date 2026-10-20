#!/usr/bin/env python3
"""F0 read-only B1 checkpoint architecture audit.

- SHA256 verifies four step5760 sealed checkpoints.
- Uses safe torch.load(weights_only=True, map_location='cpu'); never unpickles unsafe objects.
- Compares tensor state dictionaries to the current QueryMincoPlanner without loading
  optimizer or taking inference/training steps.
- Prints source anchors from the B1 training source freeze manifest if available.

Does NOT establish the training-time valid-mask/state contract or perform forest inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("A42", "A43", "A44", "B42")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def printable_type(x):
    return type(x).__module__ + "." + type(x).__name__


def collect_tensor_dicts(payload, torch, target_keys, max_depth=3):
    found = []
    seen = set()

    def walk(v, path, depth):
        if id(v) in seen or depth > max_depth:
            return
        seen.add(id(v))
        if not isinstance(v, Mapping):
            return
        items = list(v.items())
        tv = {k: val for k, val in items if isinstance(k, str) and torch.is_tensor(val)}
        # Valid state dicts can contain a small number of parameters; require
        # some overlap with the model to avoid mistaking optimizer buffers.
        if tv and set(tv).intersection(target_keys):
            found.append((path, tv, len(items)))
        for k, child in items:
            if isinstance(child, Mapping):
                walk(child, (str(k) if path == "root" else path + "." + str(k)), depth + 1)

    walk(payload, "root", 0)
    return found


def scan_source_paths(doc, root, max_results=35):
    paths = []
    def walk(v):
        if isinstance(v, Mapping):
            for k, x in v.items():
                if isinstance(k, str) and (k.endswith(".py") or k.endswith(".json")):
                    paths.append(k)
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            if v.endswith(".py") or v.endswith(".json"):
                paths.append(v)
    walk(doc)
    return [dict(path=p, exists=(root / p).exists()) for p in dict.fromkeys(paths)][:max_results]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root))
    import torch
    from query_minco.planner import QueryMincoPlanner

    model = QueryMincoPlanner()
    target = model.state_dict()
    target_keys = set(target)
    parameters = sum(p.numel() for p in model.parameters())
    print("PYTHON:", sys.executable)
    print("TORCH:", torch.__version__)
    print("MODEL:", type(model).__name__, "parameters=", parameters, "state_dict_tensors=", len(target))
    report = {
        "schema": "F0_B1_SAFE_CHECKPOINT_ARCHITECTURE_AUDIT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "python": sys.executable, "torch_version": str(torch.__version__),
        "model": "query_minco.planner.QueryMincoPlanner", "parameter_count": parameters,
        "model_tensor_keys": len(target), "weights_only_safe_loading": True,
        "forward_performed": False, "training_performed": False,
        "input_mask_contract_verified": False,
        "runs": {},
    }
    if parameters != 470461:
        print("WARNING: model parameter count differs from B1 expected 470461")

    for name in NAMES:
        p = root / "query_minco/outputs/p2c2c1b1_train_v1/runs" / name
        ckpt = p / "checkpoint_step5760.pt"
        seal = p / "checkpoint_step5760_SEALED.json"
        row = {"checkpoint": str(ckpt.relative_to(root)), "status": "BLOCKED"}
        report["runs"][name] = row
        try:
            seal_data = json.loads(seal.read_text(encoding="utf-8"))
            digest = sha256_file(ckpt)
            row["checkpoint_sha256"] = digest
            row["seal_hash_match"] = seal_data.get("sha256") == digest
            row["seal_step5760"] = seal_data.get("step") == 5760
            row["sealed_before_evaluation"] = seal_data.get("sealed_before_evaluation") is True
            if not all(row[k] for k in ("seal_hash_match", "seal_step5760", "sealed_before_evaluation")):
                raise ValueError("checkpoint seal/step/sha256 mismatch")

            payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            row["payload_type"] = printable_type(payload)
            row["payload_top_keys"] = [str(k) for k in list(payload)[:30]] if isinstance(payload, Mapping) else []
            candidates = collect_tensor_dicts(payload, torch, target_keys)
            row["tensor_dict_candidates"] = [{"path": where, "tensor_keys": len(sd), "all_items": n} for where, sd, n in candidates]
            if not candidates:
                row["status"] = "BLOCKED_NO_MATCHING_STATE_DICT"
                print(f"{name}: {row['status']} top_keys={row['payload_top_keys']}")
                continue
            # Score every tensor dictionary against the model, and require an
            # exact key+shape match. Never silently strip prefixes or ignore keys.
            best = None
            for where, state, n in candidates:
                missing = sorted(target_keys - set(state))
                unexpected = sorted(set(state) - target_keys)
                shape_bad = sorted(k for k in target_keys.intersection(state) if tuple(target[k].shape) != tuple(state[k].shape))
                score = len(missing) + len(unexpected) + len(shape_bad)
                item = (score, where, missing, unexpected, shape_bad, state)
                if best is None or item[0] < best[0]:
                    best = item
            _, where, missing, unexpected, shape_bad, state = best
            row["selected_tensor_dict_path"] = where
            row["missing_count"] = len(missing)
            row["unexpected_count"] = len(unexpected)
            row["shape_mismatch_count"] = len(shape_bad)
            row["missing_examples"] = missing[:10]
            row["unexpected_examples"] = unexpected[:10]
            row["shape_mismatch_examples"] = shape_bad[:10]
            row["status"] = "PASS_ARCHITECTURE_MATCH" if not (missing or unexpected or shape_bad) else "BLOCKED_STATE_DICT_MISMATCH"
            print(f"{name}: {row['status']} tensor_dict={where} missing={len(missing)} unexpected={len(unexpected)} shapes={len(shape_bad)}")
            if row["status"] != "PASS_ARCHITECTURE_MATCH":
                print("  examples:", row["missing_examples"], row["unexpected_examples"], row["shape_mismatch_examples"])
        except Exception as ex:
            row["status"] = "BLOCKED_SAFE_LOADING_OR_SEAL"
            row["error_type"] = type(ex).__name__
            row["error"] = str(ex)[:1400]
            print(f"{name}: {row['status']} {type(ex).__name__}: {str(ex)[:300]}")

    source = root / "query_minco/outputs/p2c2c1b1_train_v1/TRAINING_SOURCE_FREEZE.json"
    if source.exists():
        try:
            doc = json.loads(source.read_text(encoding="utf-8"))
            report["training_source_manifest_sha256"] = sha256_file(source)
            report["training_source_manifest_top_keys"] = list(doc)[:35] if isinstance(doc, Mapping) else []
            report["training_source_paths"] = scan_source_paths(doc, root)
            print("TRAINING_SOURCE_FREEZE keys:", report["training_source_manifest_top_keys"])
            print("TRAINING_SOURCE_FREEZE source files:", report["training_source_paths"][:25])
        except Exception as ex:
            report["training_source_manifest_error"] = type(ex).__name__ + ": " + str(ex)
            print("TRAINING_SOURCE_FREEZE read error:", report["training_source_manifest_error"])
    else:
        print("WARNING: TRAINING_SOURCE_FREEZE.json not present")
    report["all_four_architecture_match"] = parameters == 470461 and all(report["runs"][n]["status"] == "PASS_ARCHITECTURE_MATCH" for n in NAMES)
    report["gate"] = "READY_FOR_INPUT_CONTRACT_REVIEW" if report["all_four_architecture_match"] else "BLOCKED_CHECKPOINT_COMPATIBILITY"
    output = args.output or (root / "query_minco/outputs/forest_f0_v1/frozen_bridge/F0_B1_ARCHITECTURE_LOAD_AUDIT_V1.json")
    output = (root / output).resolve() if not output.is_absolute() else output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("GATE=" + report["gate"])
    print("AUDIT=" + str(output))
    return 0 if report["all_four_architecture_match"] else 2


if __name__ == "__main__":
    sys.exit(main())
