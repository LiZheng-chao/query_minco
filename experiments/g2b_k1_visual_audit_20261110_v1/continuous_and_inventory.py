from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from query_minco.evaluation.validation import norm_max
r=Path.cwd();p=r/'research_outputs/g2b_k1_visual_audit_20261110_v1'
a=np.load(p/'VISUAL_PAIR_ARRAYS.npz');paired=json.loads((p/'VISUAL_PAIRED_RESULTS.json').read_text());records=[];summary={}
for step in [0,32,64,128]:
 vals=[]
 for key in a.files:
  if key.startswith('step%s_'%step) and key.endswith('_coeff_a'):
   prefix=key[:-8];ca=a[key];cb=a[prefix+'_coeff_b']
   # Decoder FP32 durations are exactly the same fixed values for each candidate.
   dt=np.full(3,float(np.float32(2/3)))
   maximum=max(norm_max(c,float(t),0) for c,t in zip(ca-cb,dt));vals.append(maximum)
   records.append(dict(pair_key=prefix,step=step,continuous_max_separation_m=maximum))
 summary[step]=dict(pairs=len(vals),continuous_max_separation_m=max(vals))
with (p/'CONTINUOUS_PAIR_DIFFERENCE.json').open('x') as f:json.dump(dict(method='Stationary roots of squared norm of difference polynomial plus segment endpoints, original norm_max(order=0). Not sampled maximum.',summary=summary,records=records),f,indent=2)
spec=json.loads((r/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/K1_TRAINING_RUNSPEC_DRAFT.json').read_text());scenes={}
for row in spec['train']+spec['dev']:
 m=json.loads(Path(row['manifest']).read_text());scene=m['scene_id']
 scenes.setdefault(scene,dict(scene_id=scene,split=m['split'],identity=m['scene_identity'],scene_dir=m['scene_dir'],pose_dirs=[]))
 if m['pose_dir'] not in scenes[scene]['pose_dirs']:scenes[scene]['pose_dirs'].append(m['pose_dir'])
plan=dict(status='NOT_QUALIFIED',inventory=list(scenes.values()),qualified_independent_mirror_pairs=0,
 reason='Existing qualified scenes belong to the same Train/Dev families. No reserved left/right mirror pair with jointly bound geometry, mirrored camera and continuous feasible witness is present in current qualified binding registry.',
 preregistration={'family':'Independent counterfactual mirrored asymmetric passage; register before any model evaluation','pair_operation':'Reflect occupancy in B0 y; regenerate through original renderer and camera pipeline, not flip padded network channels','requirements':['No TRAIN/DEV family leakage; pair_id held out together','Grid/occupancy/cloud/source/camera/pose/depth/state hashes and same B0 state/goal','Pose free-sphere certificate and mask/depth calibration original contract','Continuous Hard v2 witness for each side, same T=2 and MINCO constraints; witness only qualification and never network input/fit/selection','Opposite bypass must actually be necessary: exclude straight/pre-gate stop loopholes with predeclared crossing metric','Evaluate one frozen K1 candidate on each side once; report failure without direction retuning'],
 'proposed_not_executed':{'T_s':2,'goal_B0':[2.4,0,0],'initial_v_a':[0,0,0,0,0,0],'slab_x_m':[.8,1.2],'positive_passage_y_m':[.35,1.15],'negative_passage_y_m':[-1.15,-.35],'qualification':'Feasibility not assumed; reject pair before inference if independent fixed witness fails; no iterative map tuning against network'}},
 renderer_generation='NOT_RUN',witness_feasibility='NOT_RUN',mirror_model_inference='NOT_RUN',network_updates=0)
with (p/'MIRROR_QUALIFICATION.json').open('x') as f:json.dump(plan,f,indent=2)
print('CONTINUOUS_DIFFERENCE',summary);print('MIRROR_HOLDOUT=NOT_QUALIFIED; qualified existing scenes',len(scenes))
