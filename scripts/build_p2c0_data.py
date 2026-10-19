"""Build, reopen/audit and visualize P2C-0 data only."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
import torch
from query_minco.p2b.utf8_io import write_json
from query_minco.p2c0.dataset import build_dataset
from query_minco.p2c0.audit import audit_dataset
from query_minco.p2c0.visualize import export_visualizations


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();method=Path(__file__).resolve().parents[1]
    args.output_dir.resolve().relative_to(method/"outputs")
    started=time.perf_counter()
    build_dataset(args.output_dir)
    gate=audit_dataset(args.output_dir)
    write_json(args.output_dir/"data_gate_previsualization.json",gate)
    if not gate["passed"]:
        print(json.dumps(gate,indent=2));raise SystemExit(1)
    figures=export_visualizations(args.output_dir,gate)
    gate["visualizations"]=figures
    gate["execution"]={"elapsed_s":time.perf_counter()-started,"device":"cpu",
        "torch":torch.__version__,"python":platform.python_version(),
        "CUDA_training_used":False,"network_instantiations":0,"optimizer_steps":0,
        "git_head":subprocess.check_output(["git","-C",str(method.parent),"rev-parse","HEAD"],text=True).strip()}
    write_json(args.output_dir/"data_gate.json",gate)
    print(json.dumps({k:v for k,v in gate.items() if k not in ["multiteacher_scenes","counterfactual_pairs","physics_limits"]},indent=2))


if __name__=="__main__":main()
