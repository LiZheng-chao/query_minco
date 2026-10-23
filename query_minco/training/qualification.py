"""Read-only data qualification, with declared evidence rather than invented Teacher proofs."""
from pathlib import Path
from query_minco.registry import describe
from .datasets.manifest import FrozenDataset
from .datasets.order import FrozenOrder
from .io import read, require, sha, artifact

def qualify(spec):
    ds = FrozenDataset(spec["dataset"], spec["profile"])
    order = FrozenOrder(spec["order"], ds, spec["profile"])
    # Verify every declared training sample, including shape/FP32/M and original Near/Far semantics.
    ds.load(list(ds.rows))
    if spec["profile"] == "b2c1_reference_v1":
        evidence = {p.name: read(p) for p in ds.files if p.suffix == ".json" and
                    p.name in ("TEACHER_DATA_GATE.json", "DATA_CONTRACT_GATE.json", "DATASET_MANIFEST.json",
                               "REPLAY_MANIFEST.json", "PROTOCOL_SEAL.json", "DATA_BINDINGS.json")}
        frozen = describe()["dataset_manifest_bindings"]
        for original_path, expected_sha in frozen.items():
            name = Path(original_path).name
            supplied = [b for b in ds.manifest["evidence"] if Path(b["path"]).name == name]
            require(len(supplied) == 1 and supplied[0]["sha256"] == expected_sha, "FROZEN_MANIFEST_BINDING " + name)
        expected_orders = {Path(r["path"]).name: r["sha256"] for r in evidence["DATA_BINDINGS.json"]["orders"]}
        require(expected_orders.get(order.path.name) == sha(order.path), "FROZEN_ORDER_BINDING")
        for name in ("TEACHER_DATA_GATE.json", "DATA_CONTRACT_GATE.json"):
            require(evidence.get(name, {}).get("passed") is True, "MISSING_OR_FAILED_DATA_GATE")
        seal = evidence.get("PROTOCOL_SEAL.json", {}).get("files", {})
        for b in ds.manifest["evidence"]:
            p = artifact(b)
            if p.name in seal:
                require(b["sha256"] == seal[p.name], "ORIGINAL_EVIDENCE_SEAL")
        require(evidence["TEACHER_DATA_GATE.json"]["dataset_manifest_sha"] ==
                seal.get("DATASET_MANIFEST.json"), "TEACHER_MANIFEST_BINDING")
        require(order.path.name == "%s_seed%d.npz" % (spec["experiment_arm"], spec["seed"]) and
                seal.get("orders/" + order.path.name) == sha(order.path), "ORIGINAL_ORDER_BINDING")
        original = {}
        for row in evidence.get("DATASET_MANIFEST.json", {}).get("rows", []):
            if row["block"] == "train_orth":
                original[row["id"]] = (row["M"], row["npz_sha256"], "teachers")
        for row in evidence.get("REPLAY_MANIFEST.json", []):
            original[row["id"]] = (row["M"], row["npz_sha"], "experts")
        for sid, row in ds.rows.items():
            require(all(Path(c["path"]).resolve().parent ==
                        Path(row["artifact"]["path"]).resolve().parent
                        for c in row["certificates"]), "TEACHER_CERTIFICATE_TASK_PATH")
            require(sid in original and original[sid][:2] == (row["M"], row["artifact"]["sha256"]), "UNREGISTERED_TRAIN_CACHE")
            key = original[sid][2]  # C1A "experts" and B2-C0 "teachers" are distinct frozen formats.
            certificates = [read(c["path"]) for c in row["certificates"]]
            require(any(key in cert and len(cert[key]) == row["M"] and all(
                t.get("verification", {}).get("passed") is True and t.get("task", {}).get("passed") is True
                for t in cert[key]) for cert in certificates), "TEACHER_CERTIFICATE_CONTENT")
    return ds, order, dict(passed=True, qualification=ds.manifest["qualification"],
                          dataset_sha256=sha(ds.path), order_sha256=sha(order.path),
                          **order.summary, blind_or_eval_npz_reads=0,
                          full_teacher_regeneration_verified=False)
