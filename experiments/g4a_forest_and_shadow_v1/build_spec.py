"""Freeze new split/data/source/initial weights before qualification and updates."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from query_minco.training.k1_physics_trainer_v1.initialization import build
from query_minco.training.k1_physics_trainer_v1.trainer import canonical,state_sha
from query_minco.training.k1_forest_v1.trainer import ID,verify
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def main():
 torch.set_num_threads(1);plan=json.loads((O/'PREREGISTRATION.json').read_text());rows=[];excluded=[]
 for x in json.loads((O/'forest_source/FOREST_INDEX.json').read_text()):
  if x['classification']!='CERTIFIED_FEASIBLE':excluded.append(x);continue
  split=plan['scientific_seed_split'][str(x['seed'])]
  rows.append(dict(sample_id=x['snapshot_id'],kind='forest',split=split,map_seed=x['seed'],manifest=x['manifest'],manifest_sha256=x['manifest_sha256']))
 b=R/'research_outputs/g3_b_signal_and_data_20261110_v1';index=b/'dataset/DATASET_INDEX.json';manifest=json.loads((b/'FILES_AND_SHA.json').read_text())
 if sha(index)!=manifest['artifacts']['dataset/DATASET_INDEX.json']['sha256']:raise ValueError('G3B_SOURCE_IDENTITY')
 for x in json.loads(index.read_text()):
  rows.append(dict(sample_id=x['scene_id'],kind='mirror',split=x['split'],family_id=x['family_id'],manifest=x['manifest'],manifest_sha256=x['manifest_sha256']))
 write(O/'SCIENTIFIC_DATASET_BINDING.json',dict(schema='g4a_seed_family_split_v1',component='k1_forest_and_mirror_scientific_split_binding_v1',rows=rows,excluded_undetermined=excluded,
  generic_source_loader_role='forest_test is an internal old source-export tag;new scientific split is explicit and independently SHA-bound here,original source manifests not edited',
  test_seeds_excluded=plan['g3d_test_seeds_excluded'],gt_usage='offline original L0/P0c and Hard only,not network input or labels'))
 initial={};(O/'initial_weights').mkdir();seeds=[20261104,20261105,20261106]
 for seed in seeds:
  model=build('A_reasonable',seed);p=O/'initial_weights'/('A_reasonable_seed%d.npz'%seed)
  with p.open('xb') as f:np.savez_compressed(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
  initial[str(seed)]=dict(path=str(p),sha256=sha(p),model_sha256=state_sha(model))
 src={}
 for p,h in json.loads((O/'ENTRY_PROTECTION.json').read_text())['source_before'].items():
  if '/query_minco/' in p:
   if sha(p)!=h:raise ValueError('OLD_SOURCE_DRIFT')
   src[p]=h
 for p in (R/'query_minco/training/k1_forest_v1').glob('*.py'):src[str(p)]=sha(p)
 for p in [Path(__file__),R/'experiments/g4a_forest_and_shadow_v1/run.py']:src[str(p)]=sha(p)
 train=[x for x in rows if x['split']=='train'];dev=[x for x in rows if x['split']=='dev'];order=[x['sample_id'] for x in train]
 spec=dict(schema='k1_forest_runspec_v1',trainer_id=ID,owner_root=str(O),initialization='A_reasonable',seeds=seeds,initial_states=initial,device='cpu',dtype='float32',batch_size=1,
  max_updates=128,evaluation_steps=[0,8,32,64,128],engineering_max_steps=2,optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False),clip_norm=10,
  train=train,dev=dev,train_order=order,order_sha256=canonical(order),source_sha256=src,
  data_binding=str(O/'SCIENTIFIC_DATASET_BINDING.json'),data_binding_sha256=sha(O/'SCIENTIFIC_DATASET_BINDING.json'),
  source_cost='unchanged Phase-A L0 + original P0c;no nonlocal Loss',validator='physics_v12_independent_hard_positive_duration_v2',
  formal_run_directories={'A_reasonable_seed'+str(s):str(O/'formal'/('A_reasonable_seed'+str(s))) for s in seeds},
  resume_run_directories={'A_reasonable_seed'+str(s)+'_resume':str(O/'formal'/('A_reasonable_seed'+str(s)+'_resume')) for s in seeds},
  formal_authorized=False,approval='manual researcher statement bound to complete spec SHA,seed,exact output and optional parent checkpoint SHA')
 write(O/'K1_FOREST_TRAINING_RUNSPEC.json',spec);verify(O/'K1_FOREST_TRAINING_RUNSPEC.json')
 write(O/'EXTENSION_512_POLICY.json',dict(status='DESIGN_ONLY_NOT_AUTHORIZED_NOT_RUN',requires=['new exclusive RunSpec and source/version qualification','explicit512-budget approval','parent checkpoint SHA and RNG/optimizer restore audit','new output/atomic Ledger;no overwrite128-run'],current_cli_accepts_only_128=True))
 print('FROZEN_SPEC',len(train),len(dev),sha(O/'K1_FOREST_TRAINING_RUNSPEC.json'))
if __name__=='__main__':main()
