"""Independent strict-prefix/brake geometry study; no formal-rule integration."""
from pathlib import Path
import sys,json,time
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from query_minco.integration.observed_space_v1.map import Config
from query_minco.integration.observed_space_v2.core import ObservedMap
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot
from query_minco.integration.yopo_execution_v4.trajectory import Plan
from query_minco.evaluation.validation import evaluate

def main():
 torch.set_num_threads(1);rows=json.loads((O/'SIM_REPLAY.json').read_text())['rows'];inputs=json.loads((B/'capture_ordered/INPUTS.json').read_text())['records'];item=inputs[0];now=item['stamp'];ledger=ObservedMap(Config(shape=(32,24,16)));ledger.add(np.load(item['path'])['depth'],item['pose'],now,now,now);sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,BirthGrant((.25,-.25,2.),.6,now+.2,'OFFLINE_ONLY'),now,sensor.epoch,lambda:sensor.epoch);proof=rows[0]['runtime_diagnostic_oracle'];plan=Plan(rows[0]['runtime_diagnostic_plan']);checks=[]
 for label,duration,radius,sampler in [('PREFIX_REACTION',proof['prefix_proof']['duration'],proof['prefix_proof']['radius'],lambda t:plan.sample(t)[0]),('BRAKE_FAMILY',proof['brake_duration'],proof['brake']['radius'],lambda t:evaluate(np.array(proof['brake_coefficients']),t,0))]:
  first=None
  for t in np.linspace(0,duration,proof['prefix_proof']['samples'] if label=='PREFIX_REACTION' else proof['brake']['samples']):
   result=snap.ball(sampler(float(t)),radius,now)
   if not result['passed']:first=dict(time=float(t),query=result);break
  checks.append(dict(kind=label,passed=first is None,first_failure=first))
 (O/'PREFIX_STUDY_RESULT.json').write_text(json.dumps(dict(scope='OFFLINE_GEOMETRY_ONLY_NOT_FORMAL_PREFIX_AUTHORIZATION',checks=checks,continuous_nominal_maxima=plan.maxima,robust_brake_maxima=proof['robust_brake_maxima'],delay=.35,prefix=.1,strict_unknown_rejected=True,integrated_into_formal_F1=False,live_tracking_and_brake_execution='NOT_RUN',simulator_freeze_not_real_uav_brake=True,exit_code=0),indent=2));print(checks)
if __name__=='__main__':main()
