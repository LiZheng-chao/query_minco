"""Training identity is separate from the frozen inference-only reference registry."""
import subprocess
from pathlib import Path
from query_minco.registry import METHOD_ID, describe
from .io import artifact, exact, require, read, sha, digest

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ID = "query_minco_imitation_e1_trainer_v1"
PROFILES = {
    "b2c1_reference_v1": dict(updates=2880, batch_size=8, checkpoints=[0, 960, 1920, 2880]),
    "disposable_single_step_v1": dict(updates=1, batch_size=2, checkpoints=[0, 1]),
}
def source_files():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "query_minco").rglob("*"))
            if p.is_file() and (p.suffix in (".py", ".json") or p.name == "LICENSE")}

def verify_code(code):
    exact(code, ["git_commit", "files", "allow_uncommitted_sources"], "code")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    require(head == code["git_commit"], "GIT_COMMIT")
    require(source_files() == code["files"], "PACKAGE_SOURCE_SHA")
    if not code["allow_uncommitted_sources"]:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                         cwd=str(ROOT), text=True)
        require(not status, "CODE_WORKTREE_NOT_CLEAN")
    return head

def components(training_method_id):
    """One explicit implementation binding; future methods add a new audited branch here."""
    require(training_method_id == TRAINING_ID, "UNKNOWN_TRAINING_METHOD")
    from types import SimpleNamespace
    from query_minco.models.planner import QueryMincoPlanner
    from .objective import forward
    from .recipe import optimizer
    return SimpleNamespace(model_factory=QueryMincoPlanner, objective=forward, optimizer_factory=optimizer)

def load_spec(path):
    s = read(path)
    exact(s, ["schema", "run_id", "training_method_id", "reference_method_id", "profile",
              "seed", "experiment_arm", "objective_arm", "device", "threads", "initialization",
              "dataset", "order", "output_root", "code"], "RunSpec")
    require(s["schema"] == 1 and s["training_method_id"] == TRAINING_ID, "TRAINING_METHOD")
    m = describe(s["reference_method_id"])
    require(s["reference_method_id"] == METHOD_ID and s["objective_arm"] == "B_set", "METHOD_BINDING")
    require(s["profile"] in PROFILES, "PROFILE")
    require(type(s["seed"]) is int and 0 <= s["seed"] < 2**32, "SEED")
    require(s["device"] in ("cpu", "cuda") and type(s["threads"]) is int and s["threads"] >= 1, "DEVICE_THREADS")
    require(isinstance(s["experiment_arm"], str) and s["experiment_arm"], "EXPERIMENT_ARM")
    rid = s["run_id"]
    require(isinstance(rid, str) and rid and all(c.isalnum() or c in "_-" for c in rid), "RUN_ID")
    out = Path(s["output_root"])
    require(out.is_absolute() and out.resolve() == out and out.is_dir(), "OUTPUT_ROOT")
    require("reach_planner" not in out.parts and out != ROOT, "OUTPUT_HISTORY_OR_SOURCE")
    # Outputs may not shadow installed modules or protected source trees.
    require(not out.is_relative_to(ROOT / "query_minco") if hasattr(out, "is_relative_to")
            else not str(out).startswith(str(ROOT / "query_minco") + "/"), "OUTPUT_IN_PACKAGE")
    if s["profile"] == "b2c1_reference_v1":
        require(s["experiment_arm"] in ("C0_replay", "C1_orthogonal") and s["seed"] in (42, 43, 44), "REFERENCE_ARM_SEED")
        require(not s["code"]["allow_uncommitted_sources"], "FORMAL_REQUIRES_CLEAN_CODE")
    verify_code(s["code"])
    artifact(s["dataset"]); artifact(s["order"])
    exact(s["initialization"], ["artifact", "weights_sha256", "origin"], "initialization")
    artifact(s["initialization"]["artifact"])
    require(s["initialization"]["origin"]["kind"] == (
        "B1_step5760" if s["profile"] == "b2c1_reference_v1" else "synthetic_validation"), "INITIALIZATION_LINEAGE")
    return s

def identity(spec):
    return digest(spec)

def run_path(spec):
    return Path(spec["output_root"]) / spec["run_id"]

def authorize(spec, path):
    a = read(path)
    exact(a, ["schema", "scope", "run_id", "spec_sha256", "max_updates", "authorized_by", "approved"], "authorization")
    require(a["schema"] == 1 and a["approved"] is True and bool(a["authorized_by"]), "AUTHORIZATION_REQUIRED")
    scope = "FORMAL_TRAINING" if spec["profile"] == "b2c1_reference_v1" else "DISPOSABLE_SYNTHETIC_ONLY"
    require(a["scope"] == scope and a["run_id"] == spec["run_id"] and
            a["spec_sha256"] == identity(spec) and a["max_updates"] == PROFILES[spec["profile"]]["updates"],
            "AUTHORIZATION_SCOPE")
    return a
