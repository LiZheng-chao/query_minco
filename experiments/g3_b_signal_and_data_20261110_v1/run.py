"""Thin G3-B stages; never instantiate a model or optimizer."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
from query_minco.audit.loss_gradient_direction_v1 import run_gradient_audit
from query_minco.evaluation.avoidance_families_v1 import qualify_dataset
from query_minco.evaluation.mirror_holdout_v1 import sha
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_b_signal_and_data_20261110_v1';A=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['gradient','data'],required=True);args=p.parse_args()
    if not (O/'ENTRY_PROTECTION.json').is_file():raise ValueError('PROTECTION_BASELINE_REQUIRED')
    torch.set_num_threads(1)
    with patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('NO_MODEL_INSTANTIATION')),patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_UPDATE')):
        if args.stage=='gradient':
            result=run_gradient_audit(A,O/'PREREGISTRATION.json',O/'gradient')
        else:
            binary=R/'research_outputs/physics_only_v1_2_e1_r2a_supervisor_v2/runs/build_001/artifacts/e1_renderer_bridge'
            if sha(binary)!='7cb7ead45a858da5f56fc0d9ebc3b9c35a6274ede690f54c58564fcae5b0069b':raise ValueError('NATIVE_BINARY_SHA')
            records=json.loads((R/'research_outputs/physics_only_v1_2_e1_dataset_v2/runs/qualification_001/ALL_SCENE_IDENTITIES.json').read_text())
            old_ids={r[k] for r in records for k in ['scene_source_sha256','spatial_occupancy_sha256','canonical_occupied_geometry_sha256']}
            records=json.loads((A/'scenes/QUALIFICATION_RESULT.json').read_text())['scenes']
            old_ids.update(v for r in records for v in r['identity'].values())
            q=qualify_dataset(O/'PREREGISTRATION.json',O/'dataset',binary,old_ids)
            result=dict(status=q['status'],families=q['families'],train_scenes=q['train_scenes'],dev_scenes=q['dev_scenes'],
                        scenes=[dict(scene_id=x['scene_id'],status=x['status'],reason=x.get('reason'),observability=x.get('observability')) for x in q['scenes']],network_updates=0)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
