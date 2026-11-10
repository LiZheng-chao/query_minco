"""Thin protected E1-R2 CLI; only the approved data experiment."""
import argparse
import sys
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import freeze,command,protect,OUT,PYTHON
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("freeze","build","tests","qualify"),required=True);a=p.parse_args()
    protect()
    if a.stage=="freeze":freeze();return
    # command() supervises the whole child process tree and immutable history.
    folder=OUT/"audit/qualification_001"
    command([PYTHON,"-B","-m","query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification","--worker",a.stage],
            folder,"stage_"+a.stage)
if __name__=="__main__":main()
