import argparse,json
from pathlib import Path
from query_minco.p2b.visualize import export_run,plot_attention
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path)
    p.add_argument("--dataset-dir",type=Path)
    p.add_argument("--attention-npz",type=Path)
    p.add_argument("--output-png",type=Path)
    p.add_argument("--layer",type=int,default=0);p.add_argument("--head",type=int,default=0)
    p.add_argument("--query",type=int)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.attention_npz:
        if not a.output_png:p.error("--output-png required")
        a.output_png.resolve().relative_to(root/"outputs")
        with np.load(a.attention_npz,allow_pickle=False) as arrays:
            plot_attention(a.output_png,dict(arrays),a.layer,a.head,a.query)
    else:
        if not a.run_dir or not a.dataset_dir:p.error("provide --run-dir and --dataset-dir")
        a.run_dir.resolve().relative_to(root/"outputs")
        print(json.dumps(export_run(a.run_dir,a.dataset_dir),indent=2))


if __name__=="__main__":main()
