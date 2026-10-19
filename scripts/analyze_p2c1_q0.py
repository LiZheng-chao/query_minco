import argparse
import json
from pathlib import Path
from query_minco.p2c1.analysis import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True);a=p.parse_args()
    a.experiment_dir.resolve().relative_to(Path(__file__).resolve().parents[1]/"outputs")
    r=analyze(a.experiment_dir)
    print(json.dumps({k:r[k] for k in ["decision","preregistered_stage_passed","evaluation_scene_records","test_protocol_invocations"]},indent=2))


if __name__=="__main__":main()
