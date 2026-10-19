import argparse
from pathlib import Path
from query_minco.p2c1b.holdout import final_protocol


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True);a=p.parse_args()
    a.experiment_dir.resolve().relative_to(Path(__file__).resolve().parents[1]/"outputs")
    final_protocol(a.experiment_dir)


if __name__=="__main__":main()
