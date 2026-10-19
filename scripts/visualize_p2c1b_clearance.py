import argparse
import json
from pathlib import Path
from query_minco.p2c1b.plots import export,export_failure_figures


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--resume-failure-figures",action="store_true");a=p.parse_args()
    a.experiment_dir.resolve().relative_to(Path(__file__).resolve().parents[1]/"outputs")
    fn=export_failure_figures if a.resume_failure_figures else export
    print(json.dumps(fn(a.experiment_dir,a.dataset_dir),indent=2))


if __name__=="__main__":main()
