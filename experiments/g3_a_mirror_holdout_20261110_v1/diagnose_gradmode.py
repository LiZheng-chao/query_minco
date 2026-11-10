from pathlib import Path
import sys,json,torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.models import QueryMincoPlanner
from query_minco.training.k1_physics_trainer_v1 import trainer as T
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from unittest.mock import patch
r=Path.cwd();o=r/'research_outputs/g3_a_mirror_holdout_20261110_v1';s=T.verify_spec(r/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/K1_TRAINING_RUNSPEC_DRAFT.json')
torch.set_num_threads(1)
with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
 torch.manual_seed(20261104);m=QueryMincoPlanner().eval();row=s['train'][0];b=T.load(row,'train')
 objective=RoiSafeTrainingObjective(PhysicsV12Spec(phase='A',aggregation='foundation'))
 rg=m(b.depth,b.state);og=objective(rg,b)
 with torch.no_grad():rn=m(b.depth,b.state);on=objective(rn,b)
 baseline=np.load(r/'research_outputs/g1_k1_current_baseline_20261110_v2/K6_REAL21_BEFORE.npz');z=baseline[row['sample_id']+'::raw']
 record=dict(grad_enabled_matches_original_baseline=bool(np.array_equal(rg.detach().numpy(),z)),no_grad_matches_original_baseline=bool(np.array_equal(rn.numpy(),z)),raw_gradmode_max_abs=float((rg-rn).abs().max()),coeff_gradmode_max_abs=float((og.trajectory.coeffs-on.trajectory.coeffs).abs().max()),explanation='Original K6 reference uses eval with autograd enabled. no_grad may select a fused attention implementation. Match original mode for exact regression; no numerical tolerance or physics changes.',network_updates=0)
 T.write(o/'K6_GRAD_MODE_DIAGNOSIS_RECORDED.json',record)
 print(json.dumps(record))
