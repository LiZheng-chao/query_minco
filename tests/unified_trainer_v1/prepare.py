"""Create exclusively named synthetic fixtures and explicit RunSpecs. No Teacher search."""
import argparse
import json
from pathlib import Path
import numpy as np
from query_minco.training.io import sha, write
from query_minco.training.run_spec import source_files, ROOT, TRAINING_ID
from query_minco.registry import METHOD_ID
import subprocess

def binding(path):
    return dict(path=str(path.resolve()), sha256=sha(path))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args()
    out = Path(a.output); out.mkdir()
    (out / "cache").mkdir(); (out / "runs").mkdir()
    old = ROOT / "research_outputs/migration_v1"
    rows = []
    with np.load(old / "runs/training_new001/training_arrays.npz", allow_pickle=False) as baseline:
        for i, (name, M) in enumerate([("near", 1), ("far", 4)]):
            with np.load(old / "fixtures" / (name + ".npz"), allow_pickle=False) as f:
                depth, state = f["depth_input"][0], f["state"][0]
            target = out / "cache" / (name + ".npz")
            with target.open("xb") as f:
                np.savez(f, depth_input=depth, state=state, teacher_coeffs=baseline["teacher_coeff_" + str(i)],
                         teacher_durations=baseline["teacher_durations_" + str(i)])
            rows.append(dict(id="synthetic_" + name, artifact=binding(target), split="train", regime=name, M=M,
                        roles=["backbone_" + name], teacher_status="verified_positive", certificates=[]))
    manifest = out / "dataset.json"
    write(manifest, dict(schema=1, qualification="SYNTHETIC_TEST_ONLY", roots=[str(out)],
                         rows=rows, evidence=[]))
    order = out / "order.npz"
    with order.open("xb") as f:
        np.savez(f, task_ids=np.array([["synthetic_near", "synthetic_far"]]),
                 roles=np.array([["backbone_near", "backbone_far"]]))
    cp = Path("/home/lizheng/reach_planner/query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt")
    spec = dict(schema=1, run_id="disposable_step001", training_method_id=TRAINING_ID,
                reference_method_id=METHOD_ID, profile="disposable_single_step_v1", seed=42,
                experiment_arm="synthetic_only", objective_arm="B_set", device="cpu", threads=1,
                initialization=dict(artifact=binding(cp),
                    weights_sha256="68570b63e15f21ae9e9d512393391c9a4f3cf8569dfc730cfbeb9cb8f436d63b",
                    origin=dict(kind="synthetic_validation")), dataset=binding(manifest), order=binding(order),
                output_root=str(out / "runs"), code=dict(git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
                    files=source_files(), allow_uncommitted_sources=True))
    write(out / "config.json", spec)
    print(json.dumps(dict(config=str(out / "config.json"), synthetic=True, optimizer_updates=0)))

if __name__ == "__main__": main()
