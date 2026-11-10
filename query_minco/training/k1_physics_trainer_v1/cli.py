"""CLI stages remain separate; explicit approval is required for formal train."""
import argparse
import json
from pathlib import Path
import torch
from .trainer import preflight, smoke, train, verify_spec

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["preflight","smoke","train"])
    parser.add_argument("--spec",required=True)
    parser.add_argument("--output-run",required=True)
    parser.add_argument("--approval")
    parser.add_argument("--seed",type=int,default=20261104)
    parser.add_argument("--initialization",choices=["A_random","A_reasonable"],default="A_random")
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    occupied_before=Path(args.output_run).exists()
    try:
        if args.stage=="train":
            result=train(args.spec,args.approval,args.output_run,args.seed,args.initialization)
        else:
            spec=verify_spec(args.spec)
            if args.seed not in spec["seeds"]:
                raise ValueError("SEED_NOT_PREREGISTERED")
            owner=Path(spec["engineering_owner_root"]).resolve()
            if Path(args.output_run).resolve().parent!=owner:
                raise ValueError("QUALIFICATION_OUTPUT_ROOT")
            if args.stage=="preflight":
                result=preflight(args.spec,args.output_run,args.seed,args.initialization)
            else:
                result=smoke(args.spec,args.output_run)
    except BaseException as error:
        destination=Path(args.output_run)
        if not occupied_before and destination.is_dir() and not (destination/"STOP.json").exists():
            from .trainer import write
            write(destination/"STOP.json",dict(error=repr(error),automatic_resume=False))
        raise
    print(json.dumps(dict(stage=args.stage,status="COMPLETED",output=str(result))),flush=True)

if __name__=="__main__":
    main()
