"""Fixed v2 math identities; weights require training-set calibration evidence."""
import math
from query_minco.training.io import require,digest
PREFIX="query_minco_pg2_guided_v2_"
BASELINE="query_minco_pg1_task_complete_v1"
METHODS={PREFIX+"safe_v1":("safe","rank_none"),PREFIX+"guidance_v1":("guidance","rank_none"),
 PREFIX+"control_v1":("control","rank_none"),PREFIX+"full_v1":("full","rank_pg2_feasible_goal_quality"),
 PREFIX+"yopo_rank_v1":("full","rank_yopo_style"),PREFIX+"no_rank_v1":("full","rank_none")}
FULL=PREFIX+"full_v1"
COMPONENTS=("imitation","clearance","retreat","corridor","braking","original_task","dense_guidance",
 "dynamic_limit","domain","lateral","acceleration","jerk","time","ranking","validity")
FIXED=dict(imitation=1.,original_task=1.,dynamic_limit=.1,domain=1.)
BUDGETS=dict(clearance=.3,retreat=.1,corridor=.3,braking=.1,dense_guidance=.1,lateral=.05,
 acceleration=.05,jerk=.05,time=.02,ranking=.05,validity=.05)
def recipe(method=FULL):
 require(method in METHODS,"V2_METHOD")
 arm,ranking=METHODS[method]
 return dict(schema="pg2_guided_v2_recipe_v1",method_id=method,arm=arm,ranking=ranking,
  radius=.12,margin=.03,points_per_segment=64,dynamic_limits=[3.,8.,100.],
  eps_m=.05,tau=.025/.15,braking_acceleration=4.,braking_horizon_s=.1,braking_cap_m=.15,
  lateral_acceleration=4.,low_speed_epsilon=.1,chord_corridor_scale=.1,
  projection=False,flatness_enabled=False,dense_scale=1.,
  geometry_scope="TRAIN_OR_OFFLINE_ONLY",weights_status="REQUIRES_TRAINING_SET_CALIBRATION")
def validate(cfg):
 require(cfg==recipe(cfg.get("method_id")),"V2_FIXED_MATH_RECIPE")
 return cfg
def weights_for_method(weights,method):
 cfg=recipe(method);w=dict(weights)
 require(set(w)==set(COMPONENTS) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in w.values()),"V2_WEIGHTS")
 for k,v in FIXED.items():require(w[k]==v,"V2_FIXED_ORIGINAL_WEIGHT")
 if cfg["arm"]=="safe":w["dense_guidance"]=0.
 if cfg["arm"] in ("safe","guidance"):
  for k in ("lateral","acceleration","jerk","time"):w[k]=0.
 if cfg["ranking"]=="rank_none":w["ranking"]=w["validity"]=0.
 if cfg["ranking"]=="rank_yopo_style":w["validity"]=0.
 return w
def score_rows_allowed(method):
 if method==BASELINE:return ()
 cfg=recipe(method)
 return (19,20) if cfg["ranking"]=="rank_pg2_feasible_goal_quality" else ((19,) if cfg["ranking"]=="rank_yopo_style" else ())
def draft(binding,code,calibration,method,run_id,device="cuda"):
 require(method in METHODS or method==BASELINE,"V2_DRAFT_METHOD")
 require(device in ("cpu","cuda"),"V2_DEVICE")
 resources=(dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,
  threads=1,disk_cap_gib=4) if device=="cpu" else
  dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,gpu_mib=6144,disk_cap_gib=4))
 return dict(schema="pg2_v2_pilot32_draft_v1",method=method,run_id=run_id,bindings=binding,code=code,
  calibration=calibration,seed=42,profile="pg1_pilot32_v1",batch_size=8,precision="FP32",device=device,
  optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  schedule="cosine_2e-5_to_1e-6_endpoint_inclusive",updates=32,checkpoints=[0,8,16,32],
  score_contract=dict(component_id="pg2_v2_score_row_policy_v1",allowed_rows=list(score_rows_allowed(method))),
  resources=resources,
  status="NOT_QUALIFIED",approved=False,authorization=None,preflight=None,cpu0=None,output_root=None)
