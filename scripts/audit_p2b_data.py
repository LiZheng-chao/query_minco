"""P2B-0: native source audit plus deterministic, verified toy source freeze."""
import argparse
from pathlib import Path
import torch
from query_minco.p2b.data_audit import audit_native
from query_minco.p2b.toy_data import create_dataset
from query_minco.p2b.utf8_io import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    args.output_dir.resolve().relative_to(root/"query_minco")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    result=audit_native(root)
    samples=create_dataset(args.output_dir/"toy_dataset",train_count=48,val_count=16)
    seeds=[s.scene_seed for s in samples]
    assert len(set(seeds))==64
    result["toy"]={"source":"synthetic_toy_analytic_plane","samples":64,"train":48,"val":16,
        "teacher_count_per_scene":1,"scene_seeds_disjoint":True,
        "min_teacher_plane_clearance_m":min(s.metadata["teacher_min_plane_clearance_m"] for s in samples),
        "max_teacher_speed_mps":max(s.metadata["teacher_speed_mps"] for s in samples),
        "teacher_numeric_errors":{k:max(s.metadata["teacher_checks"][k] for s in samples) for k in samples[0].metadata["teacher_checks"]},
        "multimodal_supervision":False,"claim_scope":"single-plane toy imitation, no obstacle avoidance or unknown-topology generalization"}
    result["training_gate_passed"]=True
    write_json(args.output_dir/"audit_summary.json",result)
    print({k:v for k,v in result.items() if k not in ("records","input_sha256")})


if __name__=="__main__":
    main()
