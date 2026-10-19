import argparse
from pathlib import Path
from query_minco.p2c2a.trainer import run
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True);a=p.parse_args()
    print(run(a.experiment_dir,a.dataset_dir))
