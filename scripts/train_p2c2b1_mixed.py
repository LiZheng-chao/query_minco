"""Independent P2C-2B1 entrypoint, with no historical writes."""
import argparse
from pathlib import Path
from query_minco.p2c2b1.trainer import run
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True);p.add_argument("--dataset-dir",type=Path,required=True);a=p.parse_args()
 raise SystemExit(0 if run(a.experiment_dir,a.dataset_dir)=="COMPLETED" else 1)
