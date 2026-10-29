"""Explicit commands; no automatic promotion from CPU0 into training."""
import argparse
from pathlib import Path
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
 a=sub.add_parser("cpu0");a.add_argument("--output",type=Path,required=True);a.add_argument("--require-clean",action="store_true")
 a=sub.add_parser("preflight");a.add_argument("--spec",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
 a=sub.add_parser("validate");a.add_argument("--spec",type=Path,required=True);a.add_argument("--authorization",type=Path,required=True)
 a=sub.add_parser("run");a.add_argument("--spec",type=Path,required=True);a.add_argument("--authorization",type=Path,required=True)
 args=p.parse_args()
 if args.command=="cpu0":
  from .audit import run
  run(args.output,args.require_clean)
 elif args.command=="preflight":
  from .execution import preflight
  preflight(args.spec,args.output)
 elif args.command=="validate":
  from .execution import validate_training_spec
  from query_minco.training.io import read
  validate_training_spec(read(args.spec),read(args.authorization))
  print("VALIDATED_NOT_EXECUTED")
 else:
  from .execution import execute
  execute(args.spec,args.authorization)
if __name__=="__main__":main()
