"""Independent PG2 identities and draft contracts; never authorizes updates."""
import copy
import math
from query_minco.training.io import require, digest

PREFIX = "query_minco_pg2_guided_"
METHODS = {
 PREFIX+"safe_v1": (1.,0.,0.,.1,0.),
 PREFIX+"safe_guidance_v1": (1.,1.,0.,.1,0.),
 PREFIX+"safe_guidance_control_v1": (1.,1.,.001,.1,0.),
 PREFIX+"full_v1": (1.,1.,.001,.1,.1),
}
FULL = PREFIX+"full_v1"

def recipe(method=FULL):
 require(method in METHODS,"PG2_METHOD")
 s,g,c,f,r=METHODS[method]
 return dict(schema="pg2_guided_recipe_v1",method_id=method,
  scales=dict(imitation=1.,safe=s,guidance=g,control=c,feasibility=f,ranking=r),
  radius=.12,margin=.03,points_per_segment=64,risk_eps_m=.05,lse_temperature=.025/.15,
  braking_buffer=True,braking_acceleration_mps2=4.,clearance_scale=1.,retreat_scale=1.,corridor_scale=1.,
  position_reference_m=1.,quality_scales=dict(acceleration=1.,jerk=1.,time=1.),
  dynamic_limits=[3.,8.,100.],lateral_acceleration_mps2=4.,curvature_scale=1.,terminal_stop_scale=1.,
  flatness_enabled=False,tilt_max_rad=math.pi/4,thrust_acceleration_min=2.,
  thrust_acceleration_max=20.,body_rate_max_radps=3.,domain_scale=10.,ranking_scale=1.,validity_scale=1.,projection=False,
  weights_status="RESEARCH_CANDIDATE_NOT_TUNED",geometry_scope="TRAIN_OR_OFFLINE_ONLY",
  control_model="kinematic_quality_not_controller_tracking")

def validate_recipe(cfg):
 require(cfg.get("method_id") in METHODS,"PG2_METHOD")
 expected=recipe(cfg["method_id"])
 require(set(cfg)==set(expected),"PG2_RECIPE_FIELDS")
 for k in ("schema","radius","margin","points_per_segment","dynamic_limits","geometry_scope","control_model"):
  require(cfg[k]==expected[k],"PG2_FROZEN_CONTRACT_"+k)
 for key in ("scales","quality_scales"):
  require(set(cfg[key])==set(expected[key]) and all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in cfg[key].values()),"PG2_SCALES")
 for key in ("braking_buffer","flatness_enabled","projection"): require(type(cfg[key]) is bool,"PG2_SWITCH")
 for key in ("risk_eps_m","lse_temperature","braking_acceleration_mps2","position_reference_m","lateral_acceleration_mps2","tilt_max_rad","thrust_acceleration_min","thrust_acceleration_max","body_rate_max_radps"):
  require(type(cfg[key]) in (int,float) and math.isfinite(cfg[key]) and cfg[key]>0,"PG2_POSITIVE_"+key)
 require(0<cfg["tilt_max_rad"]<math.pi/2 and cfg["thrust_acceleration_min"]<cfg["thrust_acceleration_max"],"PG2_FLATNESS_CONFIG")
 for k in ("clearance_scale","retreat_scale","corridor_scale","domain_scale","curvature_scale","terminal_stop_scale","ranking_scale","validity_scale"): require(type(cfg[k]) in (int,float) and math.isfinite(cfg[k]) and cfg[k]>=0,"PG2_NONNEGATIVE")
 return cfg

def draft_spec(binding,code,initialization,method,run_id):
 cfg=recipe(method)
 require(run_id.startswith("A42_") and all(c.isalnum() or c=="_" for c in run_id),"PG2_RUN_ID")
 return dict(schema="pg2_guided_pilot32_DRAFT_v1",method=method,recipe=cfg,recipe_sha256=digest(cfg),
  run_id=run_id,bindings=copy.deepcopy(binding),code=copy.deepcopy(code),initialization=copy.deepcopy(initialization),
  seed=42,batch_size=8,device="cpu",precision="FP32",threads=1,
  optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  schedule="cosine_2e-5_to_1e-6_endpoint_inclusive",updates=32,checkpoints=[0,8,16,32],
  gradient_guard=dict(ratio=10.,consecutive=2),
  resources=dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,disk_cap_gib=4),
  status="NOT_QUALIFIED_FOR_TRAINING",authorization=None,approved=False,
  blockers=["clean frozen source identity","complete CPU0 evidence","scoped per-run authorization","new objective adapter in shared run_engine"])
