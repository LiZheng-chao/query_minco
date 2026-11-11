from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
from query_minco.evaluation.avoidance_families_v1 import load_family_batch
from query_minco.evaluation.mirror_holdout_v1 import write
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.trajectory.minco_adapter import MincoAdapter
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_b_signal_and_data_20261110_v1';A=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
torch.set_num_threads(1);rows=[]
with patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('NO_MODEL')),patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')),torch.no_grad():
 raw=torch.from_numpy(np.load(A/'FROZEN_MIRROR_ARRAYS.npz')['offset_passage_A_left_fixed_forward_raw21'].copy())
 for row in json.loads((O/'dataset/DATASET_INDEX.json').read_text()):
  b=load_family_batch(row['manifest'],row['manifest_sha256'],row['split']);q=json.loads((Path(b.manifest['folder'])/'SCENE_QUALIFICATION.json').read_text())
  c=np.array(q['witness_coefficients']);wrong=c.copy();wrong[:,1]*=-1;dt=np.array(q['durations'])
  wrong_cross=[crossing_metric(wrong,dt,s['planes_B0'],s['gap_B0'],q['side']) for s in q['sections']]
  p=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1)(raw,b.state)
  tr=MincoAdapter(num_candidates=1)(p.head_pva,p.tail_pva,p.inner_pts,p.durations);straight=tr.coeffs[0,0].double().numpy();d=tr.durations[0,0].numpy()
  hard=validate_polynomial(straight,d,b.geometry[0],b.state[0,6:9].numpy())
  cross=[crossing_metric(straight,d,s['planes_B0'],s['gap_B0'],q['side']) for s in q['sections']]
  assert q['witness_hard']['passed'] and all(x['passed'] for x in q['witness_crossings'])
  assert not q['wrong_side_hard']['passed'] and not any(x['passed'] for x in wrong_cross)
  assert not hard['passed'] and not any(x['passed'] for x in cross)
  rows.append(dict(scene_id=row['scene_id'],correct_hard_and_all_crossing=True,wrong_side_hard=q['wrong_side_hard'],wrong_side_crossing=wrong_cross,straight_hard=hard,straight_crossing=cross))
write(O/'REQUIRED_ROUTE_SEPARATION.json',dict(status='PASS',records=rows,straight_passes=0,wrong_side_passes=0,correct_witness_passes=6,new_model_inference=False,network_updates=0))
print('PASS: six correct-side witnesses; six wrong-side/straight candidates rejected by Hard and required crossing; no model')
