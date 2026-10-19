import argparse
from pathlib import Path
from query_minco.p2c1.final_test import run_final_test


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True);a=p.parse_args()
    method=Path(__file__).resolve().parents[1]
    a.experiment_dir.resolve().relative_to(method/"outputs")
    run_final_test(a.experiment_dir,a.dataset_dir)


if __name__=="__main__":main()
