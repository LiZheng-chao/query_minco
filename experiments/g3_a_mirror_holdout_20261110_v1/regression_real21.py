"""Read-only K1(21) / K6(126) candidate-helper production-data regression."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_result
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner,make_objective
from query_minco.training.k1_physics_trainer_v1 import trainer as T
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
torch.set_num_threads(1)
spec=T.verify_spec(R/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/K1_TRAINING_RUNSPEC_DRAFT.json')
saved=json.loads((R/'research_outputs/g1_k1_current_baseline_20261110_v2/K6_HARD_BEFORE.json').read_text())
check=R/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104/step128.pt'
side=json.loads(Path(str(check)+'.json').read_text());assert T.sha(check)==side['file_sha256']
reports=[]
with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
 torch.manual_seed(20261104);k6=QueryMincoPlanner().eval();obj6=RoiSafeTrainingObjective(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=6)
 k1=make_planner().eval();k1.load_state_dict(torch.load(check,map_location='cpu',weights_only=True)['model'],strict=True)
 assert T.state_sha(k1)==side['model_sha256']
 before=[T.state_sha(k6),T.state_sha(k1)];obj1=make_objective()
 for row in spec['train']+spec['dev']:
  b=T.load(row,row['split']);out6=obj6(k6(b.depth,b.state),b)
  legacy=validate_result(out6,b);new=validate_result_candidates(out6,b)
  assert legacy==new
  original=next(x['hard'] for x in saved if x['sample_id']==row['sample_id'])
  for a,x in zip(original,new):
   assert all(x[key]==value for key,value in a.items())
  with torch.no_grad():out1=obj1(k1(b.depth,b.state),b)
  errors,direct=T.result_numeric(out1,b);one=validate_result_candidates(out1,b)[0]
  assert all(one[key]==value for key,value in direct.items())
  assert one['passed']
  reports.append(dict(sample_id=row['sample_id'],k6_full_legacy_new_equal=True,k6_original_frozen_physical_fields_equal=True,k1_direct_equal=True,k1_hard_passed=one['passed']))
 assert before==[T.state_sha(k6),T.state_sha(k1)]
T.write(O/'HARD_REAL21_REGRESSION.json',dict(status='PASS',tasks=reports,k6_candidates=126,k1_candidates=21,new_network_updates=0,model_sha_unchanged=True))
print('PASS: K6 full legacy/new 126 rows; frozen physical fields126; K1 direct21; updates0')
