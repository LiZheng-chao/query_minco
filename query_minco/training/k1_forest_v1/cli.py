"""Explicit forest lifecycle CLI; default preflight, never default training."""
import argparse,json
from pathlib import Path
import torch
from . import trainer as t
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','smoke-first','smoke-resume','approve','train'],default='preflight')
 p.add_argument('--spec',required=True);p.add_argument('--output-run');p.add_argument('--approval');p.add_argument('--researcher-statement')
 p.add_argument('--seed',type=int,default=20261104);p.add_argument('--resume');p.add_argument('--resume-sha')
 args=p.parse_args();torch.set_num_threads(1)
 if args.stage=='approve':
  spec=t.verify(args.spec)
  if not args.approval or not args.researcher_statement or len(args.researcher_statement)<20:p.error('manual researcher statement and new approval path required')
  run_id='A_reasonable_seed'+str(args.seed)+('_resume' if args.resume_sha else '')
  if run_id not in spec['resume_run_directories' if args.resume_sha else 'formal_run_directories']:p.error('unregistered run')
  out=Path(args.approval);out.parent.mkdir(parents=True,exist_ok=True)
  t.write(out,dict(decision='APPROVE_EXACT_RUNSPEC',runspec_sha256=t.sha(args.spec),run_id=run_id,researcher_statement=args.researcher_statement,authority='researcher_manual_shell',resume_sha256=args.resume_sha))
  print('APPROVAL_CREATED_NO_TRAINING',str(out));return
 if not args.output_run:p.error('--output-run required')
 if args.stage=='preflight':out=t.preflight(args.spec,args.output_run)
 elif args.stage=='smoke-first':out=t.smoke(args.spec,args.output_run)
 elif args.stage=='smoke-resume':
  if not args.resume:p.error('--resume required')
  out=t.smoke(args.spec,args.output_run,args.resume)
 else:
  if bool(args.resume)!=bool(args.resume_sha):p.error('resume path and approved checksum required together')
  out=t.train(args.spec,args.approval,args.output_run,args.seed,args.resume,args.resume_sha)
 print('COMPLETED',str(out))
if __name__=='__main__':main()
