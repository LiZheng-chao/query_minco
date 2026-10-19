import argparse
import json
from pathlib import Path
from query_minco.p2c1.trainer import train_all


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--dataset-dir",type=Path,required=True);a=p.parse_args()
    summary=train_all(a.experiment_dir,a.dataset_dir)
    print(json.dumps({"runs":len(summary["runs"]),"seconds":summary["total_training_experiment_seconds"],
                      "test_loaded":summary["test_loaded"]},indent=2))


if __name__=="__main__":main()
