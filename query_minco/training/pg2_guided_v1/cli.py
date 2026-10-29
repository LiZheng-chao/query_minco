"""CPU zero-update CLI only. Training has no enabled command."""
import argparse
from pathlib import Path
def main():
 parser=argparse.ArgumentParser()
 sub=parser.add_subparsers(dest="command",required=True)
 cpu=sub.add_parser("cpu0")
 cpu.add_argument("--output",type=Path,required=True)
 cpu.add_argument("--wall-cap-s",type=int,default=1800)
 args=parser.parse_args()
 if args.command=="cpu0":
  from .audit import run
  run(args.output,args.wall_cap_s)
if __name__=="__main__":main()
