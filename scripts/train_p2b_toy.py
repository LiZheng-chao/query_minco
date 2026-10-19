import argparse
import json
from pathlib import Path
import torch
from query_minco.p2b.trainer import train_experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--overfit-steps",type=int,default=450)
    parser.add_argument("--small-steps",type=int,default=650)
    args=parser.parse_args()
    config={"seed":42,"source":"synthetic_toy","overfit_samples":12,"microbatch":4,
            "overfit_steps":args.overfit_steps,"small_steps":args.small_steps,
            "overfit_lr":4e-4,"small_lr":2e-4,"device":"cuda" if torch.cuda.is_available() else "cpu",
            "amp":False,"loss_weights":{"position":1.,"velocity":.1,"terminal_position":.5,"teacher_time":.2},
            "collision":False,"dynamics":False,"jerk":False,"ranking":False,"feasibility":False}
    print(json.dumps(train_experiment(args.dataset_dir,args.output_dir,config),indent=2))


if __name__=="__main__":
    main()
