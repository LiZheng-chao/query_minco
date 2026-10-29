"""Proposed research constants; weights calibrated on training data only."""
from query_minco.training.io import require,digest
PREFIX="query_minco_pg3_feasible_mode_v1"
METHODS={PREFIX+"_mode":("mode",),PREFIX+"_route":("mode","bottleneck","convex_corridor"),PREFIX+"_control":("mode","bottleneck","convex_corridor","qa","qj","qt","lateral")}
EXTRA=("mode","bottleneck","convex_corridor","qa","qj","qt","lateral","diversity")
BUDGETS=dict(mode=.5,bottleneck=.3,convex_corridor=.3,qa=.025,qj=.025,qt=.01,lateral=.025,diversity=.05)
def recipe():
 return dict(schema="pg3_feasible_mode_recipe_v1",queries=6,raw_dim=21,capacity=2,rotation=True,
  sigma_W=1.,sigma_P=1.,sigma_V=1.,alpha_W=1.,alpha_P=1.,alpha_V=.1,huber_delta=1.,
  alpha_A=0.,alpha_duration=0.,sigma_A=1.,sigma_duration=1.,sigma_perp=1.,sigma_parallel=1.,
  beta_perp=1.,beta_parallel=.1,sigma_plane=.15,radius=.12,margin=.03,proof_pad=1e-6,
  corridor_max_depth=4,bottleneck_range_m=.5,a_ref=8.,j_ref=100.,T_ref=2.2,lateral_ref=4.,
  quality_gamma=0.,quality_c_gate=.03,quality_tau=.03,diversity_fraction=.5,
  dynamic_limits=[3.,8.,100.],projection=False,ranking=False)
def effective(weights,method):
 require(method in METHODS,"PG3_METHOD")
 return {k:float(weights[k]) if k in METHODS[method] else 0. for k in EXTRA}
