"""Explicit planning/CPU0 or separately authorized CPU-only run."""
import argparse
from pathlib import Path
from query_minco.training.io import read,write,sha,require
from .spec import make_spec,authorization,validate,check_authorization
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 t=sub.add_parser('cpu0');t.add_argument('--output',type=Path,required=True);t.add_argument('--require-clean',action='store_true')
 for name in ['validate','run']:
  t=sub.add_parser(name);t.add_argument('--spec',required=True);t.add_argument('--authorization',required=True)
 a=p.parse_args()
 if a.command=='cpu0':
  from .cpu0 import run
  import torch
  from unittest.mock import patch
  with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('R12_CPU0_OPTIMIZER_CONSTRUCTION_FORBIDDEN')) as guard:
   run(a.output,a.require_clean)
   assert guard.call_count==0
 else:
  s=read(a.spec);check_authorization(s,read(a.authorization))
  if a.command=='validate':
   from .pilot import preflight
   preflight(s)
   print('R12_AUTHORIZED_CPU_PLAN_VALIDATED_NO_UPDATES')
  else:
   from .pilot import execute
   execute(a.spec,a.authorization)
if __name__=='__main__':main()
