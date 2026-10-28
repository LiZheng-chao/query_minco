"""R0 exact old sum; Rtail adds only the existing R1 physical-time Tail."""
import torch
from query_minco.training.pg1.objective import forward as old_forward
from query_minco.training.pg1_v2_r1.losses import bottleneck_safety_loss
from .spec import recipe
def tail_from_samples(positions,weights,boxes,certified_empty):
 return bottleneck_safety_loss(positions,weights,boxes,geometry_bound=[True]*len(boxes),
  certified_empty=certified_empty,temperature=.025,tail_fraction=.1,radius=.12,margin=.03,eps=.05)['losses']['tail']
def compose(base_loss,tail_mean,scale):
 if scale not in (0.,.0025):raise ValueError('R12_FIXED_LAMBDA')
 return base_loss if scale==0 else base_loss+scale*tail_mean
def forward(model,samples,sidecar,method):
 cfg=recipe(method)
 z=old_forward(model,samples,sidecar,'query_minco_pg1_task_complete_v1')
 fields=sidecar.load(samples)
 tail=tail_from_samples(z['positions'],z['time_weights'],[f.boxes.to(z['raw'].device) for f in fields],[f.row['empty_certified'] for f in fields])
 return dict(z,base_loss=z['loss'],base_recipe=z['recipe'],recipe=cfg,
  terms=dict(z['terms'],tail=tail.mean()),per_query=dict(z['per_query'],tail=tail),
  loss=compose(z['loss'],tail.mean(),cfg['tail_scale']))
