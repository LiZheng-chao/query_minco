import argparse
from pathlib import Path
from query_minco.p2c1b.trainer import train_all


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True);a=p.parse_args()
    train_all(a.experiment_dir,a.dataset_dir)


if __name__=="__main__":main()
