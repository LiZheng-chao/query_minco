"""Fail-closed native sensor certificate manifest. No GT/fixture certificate injection."""
import hashlib,json,numpy as np
from query_minco.inference.observed_space import DepthHistory,DepthFrame
from query_minco.inference.shadow import K
from query_minco.inference.online_selector import select
class NativeVolume:
 def __init__(self):self.history=DepthHistory();self.bounds=None
 def add(self,stamp,depth,origin,rotation):self.history.add(DepthFrame(stamp,depth,origin,rotation,'B0'),K)
 def certificate(self,coefficients_world,durations,now):
  z=self.history.volume(now).certify(coefficients_world,durations,now,required_until=now+float(np.sum(durations)))
  inputs=hashlib.sha256()
  for f in self.history.fresh(now):
   inputs.update(np.array([f.stamp],dtype='<f8').tobytes());inputs.update(f.depth.tobytes());inputs.update(f.origin.tobytes());inputs.update(f.rotation.tobytes())
  z.update(version='native_sensor_volume_manifest_v1',input_history_sha256=inputs.hexdigest(),r_body_m=.12,base_margin_m=.03,extra_bounds='UNQUALIFIED: RMS/P95 are not worst-case',perception_model='NATIVE_POINT_RAYS_ONLY',footprint_bound_available=False,braking_space='UNPROVEN',online_permission=False)
  assert z['status']=='UNKNOWN' and not z['selected_permission'];return z
 def certify_external(self,*args,**kwargs):raise ValueError('NO_GT_FIXTURE_OR_EXTERNAL_VOLUME_INJECTION')
def unchanged_selector(depth,state,out,age):
 z=select(depth,state,out,age);assert z['selected_query'] is None;return z
