"""Thin zero-update signal entry. The TRAIN label denotes data split, never a training run."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch
from query_minco.audit.k1_signal_diagnostics_v1 import run_train,seal_train_decision,run_dev
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_c_nonlocal_signal_20261110_v1'
SOURCES=[Path(__file__).resolve(),R/'query_minco/audit/nonlocal_signal_v1.py',R/'query_minco/audit/k1_signal_diagnostics_v1.py']
def verify():
    expected=json.loads((O/'EXPERIMENT_SOURCE_FREEZE.json').read_text())
    for p,h in expected.items():
        if sha(p)!=h:raise ValueError('EXPERIMENT_SOURCE_DRIFT')
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train-diagnostics','seal','dev-forward'],required=True);args=p.parse_args()
    torch.set_num_threads(1)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER_ALLOWED')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_OPTIMIZER_STEP')):
        if args.stage=='train-diagnostics':
            if (O/'train_diagnostics').exists():raise FileExistsError('EXCLUSIVE_OUTPUT_OCCUPIED')
            if not (O/'EXPERIMENT_SOURCE_FREEZE.json').exists():write(O/'EXPERIMENT_SOURCE_FREEZE.json',{str(p):sha(p) for p in SOURCES})
            verify()
            with torch.enable_grad():result=run_train(R,O/'PREREGISTRATION.json',O/'train_diagnostics')
        elif args.stage=='seal':
            verify();result=seal_train_decision(O/'PREREGISTRATION.json',O/'train_diagnostics',O/'TRAIN_ONLY_DECISION.json')
        else:
            verify()
            with torch.no_grad():result=run_dev(R,O/'PREREGISTRATION.json',O/'TRAIN_ONLY_DECISION.json',O/'dev_forward')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
