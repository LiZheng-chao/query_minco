import argparse
import json
from pathlib import Path
from query_minco.p2c1.visualize import export_figures


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True);p.add_argument("--figures-dir-name",default="figures");a=p.parse_args()
    method=Path(__file__).resolve().parents[1];a.experiment_dir.resolve().relative_to(method/"outputs")
    if a.figures_dir_name not in ["figures","figures_readable"]:p.error("invalid figure directory")
    print(json.dumps(export_figures(a.experiment_dir,a.dataset_dir,a.figures_dir_name),indent=2))


if __name__=="__main__":main()
