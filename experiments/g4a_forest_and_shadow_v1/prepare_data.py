"""Prepare pre-registered independent forest seeds, without network construction."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.forest_adapter_v1 import prepare,verify_files
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from unittest.mock import patch
from query_minco.models import QueryMincoPlanner
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def main():
 old=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
 plan=json.loads((old/'PREREGISTRATION.json').read_text())
 for row in plan['maps']:
  row['seed']+=1000;row['map_id']='forest_'+row['density']+'_'+str(row['seed'])
 plan.update(component='g4a_native_forest_train_dev_preregistered_v1',scientific_seed_split={str(x['seed']):('train' if x['seed']%100==1 else 'dev') for x in plan['maps']},
  g3d_test_seeds_excluded=[3101,3102,3201,3202,3301,3302],engineering_max_optimizer_attempts=2,
  method='original A_reasonable K1 / L0 P0c only',mirror_source='G3B qualified TRAIN4 DEV2,read-only no split changes',
  shadow=dict(mode='non-control only',forest_seed=3101,odometry='scripted kinematic input fixture,not executed aircraft',methods=['fixed_forward','state_goal_quintic','K1_step128']))
 plan['pose_anchor_fractions']=[[.3,.35],[.45,.65]]
 plan['states']=[dict(v=[.6,.1,0],a=[.05,0,0]),dict(v=[1.,0,0],a=[-.05,.05,0])]
 write(O/'PREREGISTRATION.json',plan)
 source=json.loads((old/'YOPO_SOURCE_IDENTITY.json').read_text());verify_files(source);write(O/'YOPO_SOURCE_IDENTITY.json',source)
 (O/'build').mkdir();write(O/'build/BUILD_IDENTITY.json',json.loads((old/'build/BUILD_IDENTITY.json').read_text()))
 with patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('NO_MODEL_DURING_QUALIFICATION')):
  q=prepare(R,O/'PREREGISTRATION.json',O/'forest_source')
 print(json.dumps(q,indent=2))
if __name__=='__main__':main()
