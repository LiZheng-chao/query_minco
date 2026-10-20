#!/usr/bin/env python3
"""F0 read-only depth/voxel-GT cross-check, never a continuous collision certificate.

Input: existing paired forest observation + separately regenerated native forest
occupancy voxel binary. Uses no ROS, no network inference, no huge pointcloud.

Exports a small JSON audit alongside the NEW regenerated map output directory.
Requires near-zero roll/pitch/yaw and zero camera pitch for initial smoke test;
other poses require full rotation handling in a future version.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def symmetric_index(v,n):
    if n <= 1: raise ValueError('Invalid mirror dimension')
    v %= 2*n-2
    return (2*n-2-v) if v>=n else v


def map_query(pt, voxel, origin, res, dim):
    ix,iy,iz=(int(math.floor((float(pt[j])-float(origin[j]))/res)) for j in range(3))
    ix=symmetric_index(ix,dim[0]); iy=symmetric_index(iy,dim[1])
    if iz>=dim[2]: return 0
    if iz<=0: return 1
    return int(voxel[ix,iy,iz])


def raycast(pt0, u, v, camera, voxel, origin, res, dim):
    # Matches native CUDA cameraRaycastKernel: optical image rays map to internal FLU.
    fx,fy,cx,cy=(float(camera[t]) for t in ['fx','fy','cx','cy'])
    maxd=float(camera['max_depth_dist'])
    ratio_y=-(u-cx)/fx
    ratio_z=-(v-cy)/fy
    dx=np.float32(0.5*res)
    step=np.asarray([dx, np.float32(ratio_y*dx),np.float32(ratio_z*dx)],dtype=np.float32)
    pos=np.asarray(pt0,dtype=np.float32).copy()
    point_x=np.float32(0)
    max_steps=int(np.ceil(maxd / float(dx)))+3
    for _ in range(max_steps):
        point_x=np.float32(point_x+dx)
        pos+=step
        if map_query(pos,voxel,origin,res,dim):
            # In native renderer the depth is x-coordinate of voxel center, not
            # raw ray distance. Also account for original world-origin translation.
            idx=int(math.floor((float(pos[0])-float(origin[0]))/res))
            return float(np.float32((idx+0.5)*res+float(origin[0])-float(pt0[0])))
        if float(point_x)>=maxd: return maxd
    raise RuntimeError('Raycast unexpectedly exceeded maximum depth iterations')


def selftest():
    o=np.asarray([0.,-1.,0.]); n=(100,30,100)
    a=np.zeros(n,dtype=np.uint8)
    a[45,10,20]=1
    assert map_query((4.55,0.05,2.05),a,o,0.1,n)==1
    assert map_query((4.55,0.05,2.05+0.6),a,o,0.1,n)==0
    assert map_query((4.55,0.05,-0.1),a,o,0.1,n)==1
    assert map_query((4.55,0.05,30),a,o,0.1,n)==0
    assert symmetric_index(-1,10)==1
    assert symmetric_index(10,10)==8
    print('PASS synthetic voxel/index/query tests')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-dir',type=Path,required=True)
    p.add_argument('--observation-npz',type=Path,required=True)
    p.add_argument('--observation-metadata',type=Path,required=True)
    p.add_argument('--expected-config-sha256',required=True)
    p.add_argument('--samples',type=int,default=320)
    p.add_argument('--expected-point-count',type=int,required=True,
                   help='PointCloud2 width observed from the running forest simulator')
    p.add_argument('--selftest',action='store_true')
    args=p.parse_args()
    if args.selftest: selftest()
    d=args.map_dir.resolve(); src=args.observation_npz.resolve(); mj=args.observation_metadata.resolve()
    with open(d/'occupancy_metadata.json') as f: m=json.load(f)
    with open(mj) as f: j=json.load(f)
    if m['schema']!='F0_OFFLINE_NATIVE_FOREST_VOXEL_REPLICA_V1': raise ValueError('Wrong replica metadata schema')
    if j['schema']!='F0_PREFLIGHT_FOREST_PAIRED_OBSERVATION_V2': raise ValueError('Wrong observation metadata schema')
    if int(j['forest_config']['seed'])!=int(m['seed']): raise ValueError('Seed mismatch')
    if int(m['point_count'])!=args.expected_point_count: raise ValueError('Point count differs from actual live ROS /mock_map; STOP')
    source_root=Path(j['root'])
    for short,rel in (('maps_cpp_current_file','Simulator/src/src/maps.cpp'),
                      ('sensor_cuda_current_file','Simulator/src/src/sensor_simulator.cu')):
        if sha(source_root/rel)!=j['source_hashes'][short]['sha256']:
            raise ValueError('Source code hash mismatch for '+rel)
    if float(j['forest_config']['resolution'])!=float(m['resolution_m']):
        # Small float rounding allowed; parser float vs C++ float precision
        if abs(float(j['forest_config']['resolution'])-float(m['resolution_m']))>1e-6:
            raise ValueError('Resolution mismatch')
    if sha(Path(m['config_path']))!=args.expected_config_sha256: raise ValueError('Current YAML SHA differs from expected')
    if j['active_config_sha256']!=args.expected_config_sha256: raise ValueError('Captured YAML SHA differs')
    if Path(m['config_path']).resolve()!=Path(j['active_config_path_from_ros']).resolve(): raise ValueError('Different config path')
    if sha(src)!=j['observation_pair_npz_sha256']: raise ValueError('Captured observation changed since recording')
    if sha(Path(m['tree_path']))!=j['source_hashes']['tree_ply']['sha256']: raise ValueError('Tree PLY changed')
    if any(abs(float(s['odometry']['orientation_xyzw'][k])-([0,0,0,1][k]))>1e-6
           for s in j['samples'] for k in range(4)):
        raise ValueError('v1 validator requires identity body orientation')
    if abs(float(j['active_camera_config']['pitch']))>1e-6:
        raise ValueError('v1 validator requires zero camera pitch')
    cam=j['active_camera_config']
    shape=tuple(map(int,m['shape_xyz'])); res=float(m['resolution_m']); origin=np.asarray(m['origin_m'],dtype=np.float64)
    binary=d/m['voxel_file']
    if binary.stat().st_size!=math.prod(shape): raise ValueError('Voxel file has unexpected byte count')
    voxel=np.memmap(binary,dtype=np.uint8,mode='r',shape=shape,order='C')
    if np.count_nonzero(voxel)!=m['occupied_voxels']: raise ValueError('Occupancy voxel count mismatch')
    with np.load(src,allow_pickle=False) as data:
        raw=data['depth_m_first']; raw2=data['depth_m_second']
        if not np.array_equal(raw,raw2): raise ValueError('Paired depth frames differ')
        if raw.shape!=(int(cam['image_height']),int(cam['image_width'])):
            raise ValueError('Observation size mismatch')
        depth=raw.copy()
    pos=j['samples'][0]['odometry']['position_m']
    H,W=depth.shape
    all_pixels=H*W
    rng=np.random.default_rng(20261020)
    choices=rng.choice(all_pixels,size=min(max(1,args.samples),all_pixels),replace=False)
    # Add a fixed spatial grid, including central direction.
    grid=[int(v)*W+int(u) for v in np.linspace(0,H-1,9) for u in np.linspace(0,W-1,11)]
    center=int(round(float(cam['cy'])))*W+int(round(float(cam['cx'])))
    choices=sorted(set(map(int,choices))|set(grid)|{center})
    preds=np.array([raycast(pos, idx%W,idx//W,cam,voxel,origin,res,shape)
                    for idx in choices],dtype=np.float64)
    targets=depth.reshape(-1)[choices].astype(np.float64)
    error=np.abs(preds-targets)
    dmax=float(cam['max_depth_dist'])
    hit=(targets>0)&(targets<dmax)
    predicted_hit=(preds>0)&(preds<dmax)
    agreement=float(np.mean(hit==predicted_hit))
    p95=float(np.percentile(error,95))
    maxerr=float(error.max())
    gt_p95_criteria=0.15
    gt_error_fraction_criteria=0.01
    gt_bad_fraction=float(np.mean(error>0.20))
    ok=agreement>=0.99 and p95<=gt_p95_criteria and gt_bad_fraction<=gt_error_fraction_criteria
    result={
        'schema':'F0_NATIVE_FOREST_GT_DEPTH_RAY_AUDIT_V1',
        'is_formal_f0_result':False,
        'identical_to_live_map_proven':False,
        'source_observation_sha256':sha(src),
        'regenerated_voxel_sha256':sha(binary),
        'source_config_sha256':args.expected_config_sha256,
        'grid_shape_xyz':list(shape),'origin_m':list(map(float,origin)),
        'occupied_voxels':int(m['occupied_voxels']),
        'tested_pixel_count':len(choices),'hit_class_agreement':agreement,
        'depth_abs_error_p95_m':p95, 'depth_abs_error_max_m':maxerr,
        'fraction_abs_error_gt_0p20m':gt_bad_fraction,
        'pre_registered_smoke_criteria':{
            'hit_class_agreement_at_least':0.99,'depth_abs_error_p95_m_at_most':gt_p95_criteria,
            'fraction_gt_0p20m_at_most':gt_error_fraction_criteria},
        'preflight_ray_agreement':'PASS' if ok else 'BLOCKED',
        'caveat':'Depth-ray smoke only, not entire-map identity, map-free online access, ESDF, body swept-volume safety, or continuous collision proof.'
    }
    outfile=d/'F0_GT_DEPTH_RAY_AUDIT.json'
    if outfile.exists(): raise FileExistsError('Refusing overwrite '+str(outfile))
    with open(outfile,'w') as f: json.dump(result,f,indent=2)
    print('Voxels: %s, occupied: %s'%(shape,m['occupied_voxels']))
    print('Ray samples=%d; hit class agreement=%.4f'%(len(choices),agreement))
    print('Depth absolute error p95=%.4f m; max=%.4f m; >0.20m=%.2f%%'%(p95,maxerr,100*gt_bad_fraction))
    print('F0 geometry-ray smoke:',result['preflight_ray_agreement'])
    print('Audit:',outfile)
    if not ok: raise SystemExit(3)

if __name__=='__main__': main()
