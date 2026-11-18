"""Per-ray support/missing-return details for the saved blocking cells."""
from pathlib import Path
import sys,json,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from experiments.b0_sensor_observability_v1.reference import *
def main():
 exact=json.loads((O/'B0_EXACT_RAY_REFERENCE.json').read_text());voxels=exact['voxel_diagnosis']['voxels'];centers=np.array([v['world_center'] for v in voxels]);old=json.loads((BASE/'capture_ordered/INPUTS.json').read_text())['records'];new=json.loads((O/'virtual_views/INPUTS.json').read_text())['records'];rows=[]
 for scope,records in [('BASELINE',old[:1]),('VIRTUAL',new)]:
  for record in records:
   dep=np.load(record['path'])['depth'];g=project(centers,record['pose']);modes={mode:evaluate(dep,g,mode) for mode in ['current','uncapped_square','conservative_rectangle']};cells=[]
   for k,v in enumerate(voxels):
    checks={}
    for mode,(mask,reasons) in modes.items():
     bounds=g['rectangle'][k] if mode=='conservative_rectangle' else g['square'][k];left,right,top,bottom=bounds;detail=dict(free=bool(mask[k]),reasons=reasons[k],pixel_window=bounds.tolist())
     if left>=0 and right<160 and top>=0 and bottom<90 and g['z'][k]-g['r'][k]>=.04:
      w=dep[top:bottom+1,left:right+1];bad=~(np.isfinite(w)&(w>=.04)&(w<20));counts=dict(zero=int((w==0).sum()),finite_below_near=int((np.isfinite(w)&(w<.04)).sum()),at_or_above_range=int((np.isfinite(w)&(w>=20)).sum()),nonfinite=int((~np.isfinite(w)).sum()),valid=int((~bad).sum()),pixels=w.size)
      detail['depth_window_counts']=counts
      if bad.any():
       v0,u0=np.argwhere(bad)[0];value=float(w[v0,u0]);detail['first_invalid_ray']=dict(pixel=[int(left+u0),int(top+v0)],depth_value=value if np.isfinite(value) else str(value),optical_ray=[(left+u0-80)/80,(top+v0-45)/80,1])
     checks[mode]=detail
    cells.append(dict(world_center=v['world_center'],optical=g['optical'][k].tolist(),projection=g['projection'][k].tolist(),checks=checks))
   rows.append(dict(scope=scope,frame=record['index'],stamp=record['stamp'],pose=record['pose'],depth_population=dict(total=dep.size,valid=int((np.isfinite(dep)&(dep>=.04)&(dep<20)).sum()),zero=int((dep==0).sum()),at_or_above_range=int((np.isfinite(dep)&(dep>=20)).sum()),nonfinite=int((~np.isfinite(dep)).sum())),cells=cells))
 # Execute the unchanged frozen A*/MINCO on uncapped single-view snapshots.
 from experiments.b0_sensor_observability_v1.multiview import astar_check
 item=old[0];now=item['stamp'];dep=np.load(item['path'])['depth'];ledger=ObservedMap(CFG);ledger.add(dep,item['pose'],now,now,now);sensor=ledger.snapshot(now);grid=(sensor.origin+(np.indices(CFG.shape).transpose(1,2,3,0)+.5)*.1).reshape(-1,3);g=project(grid,item['pose'])
 single_astar={mode:astar_check(make_snapshot(sensor,evaluate(dep,g,mode)[0],now),now) for mode in ['current','uncapped_square','conservative_rectangle']}
 write('RAY_SUPPORT_DETAILS.json',dict(scope='OFFLINE_NO_GT_NO_FREE_FOR_NO_RETURN',baseline_content_identical_across_frames=exact['unique_depth_contents']==1,single_view_astar=single_astar,rows=rows,exit_code=0));print('DEPTH_POPULATIONS',[(x['scope'],x['frame'],x['depth_population']) for x in rows])
if __name__=='__main__':main()
