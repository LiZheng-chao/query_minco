"""PG1 phase1 CLI: plans only, plus explicitly bound CPU0/audit. No train verb."""
import argparse,json,subprocess
from pathlib import Path
import torch,numpy as np
from query_minco.training.io import write,read,digest,sha,require
from query_minco.training.run_spec import source_files,ROOT
from query_minco.models import QueryMincoPlanner
from .spec import METHODS,PROFILES,draft,authorization_template,qualify_bindings
from .objective import cpu_zero_update
from .audit import audit_cpu
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 plan=sub.add_parser('plan');plan.add_argument('--output',required=True)
 cpu=sub.add_parser('cpu0');cpu.add_argument('--bindings',required=True);cpu.add_argument('--bindings-sha256',required=True)
 cpu.add_argument('--seed',type=int,choices=[42,43,44],required=True);cpu.add_argument('--method',choices=list(METHODS),required=True);cpu.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output);require(out.is_absolute() and out.resolve()==out and ROOT/'research_outputs/pg1_task_complete_v1' in out.parents,'PG1_OUTPUT_TREE')
 out.mkdir(parents=True,exist_ok=False)
 if a.action=='plan':
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
  code=dict(git_commit=head,files=source_files(),allow_uncommitted_sources=False)
  specs=[]
  for profile_id in ('pg1_pilot32_v1','pg1_formal2880_v1'):
   for seed in PROFILES[profile_id]['seeds']:
    for method in METHODS:specs.append(draft(method,profile_id,seed,code))
  write(out/'RUN_SPEC_DRAFTS.json',specs)
  write(out/'AUTHORIZATION_TEMPLATES.json',[authorization_template(s) for s in specs])
  dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=str(ROOT),text=True))
  write(out/'READINESS.json',dict(code_worktree_clean=not dirty,status='NOT_READY',G2='NOT_QUALIFIED',bindings=None,
   reasons=(['WORKTREE_NOT_CLEAN'] if dirty else [])+['NO_USER_DATA_ORDER_DEV_INITIALIZATION_BINDING',
            'NO_REAL_G2_AUDIT','NO_FROZEN_MODE_CLASSIFIER_OR_CALIBRATED_SELECTOR','NO_GPU_TIMING','TRAINING_EXECUTOR_NOT_IMPLEMENTED_IN_PHASE1','NO_SINGLE_RUN_HUMAN_AUTHORIZATION'],
   training_profiles=list(PROFILES),optimizer_steps=0,training_commands='NOT_AVAILABLE_PHASE1'))
 else:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  ds,geo,init,qualification=qualify_bindings(dict(path=a.bindings,sha256=a.bindings_sha256),a.seed)
  ids,_=geo.orders[a.seed].batch(0);samples=ds.load(ids)
  m=QueryMincoPlanner();blob=torch.load(init['checkpoint']['path'],map_location='cpu',weights_only=True);m.load_state_dict(blob['state_dict'],strict=True);m.train()
  result=cpu_zero_update(m,samples,geo,a.method)
  require(result['record']['weights_before']==init['weights_sha256'],'PG1_CPU_INITIALIZATION_IDENTITY')
  audit=audit_cpu(result['record'],result['arrays'],samples,geo)
  with (out/'ARRAYS.npz').open('xb') as f:np.savez(f,**result['arrays'])
  write(out/'RECORD.json',result['record']);write(out/'AUDIT.json',audit);write(out/'QUALIFICATION.json',qualification)
 print(json.dumps(dict(output=str(out),action=a.action,optimizer_steps=0)))
if __name__=='__main__':main()
