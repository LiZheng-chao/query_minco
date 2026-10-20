#!/usr/bin/env python3
"""F0-only camera contract prep and offline coordinate sanity check.

Writes ONLY under query_minco/configs/ and query_minco/outputs/forest_f0_v1/.
It NEVER modifies Simulator/, starts ROS, publishes odom/control, or evaluates a model.
Not a GT map/collision/occlusion certificate.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TARGET = {
    'fx': '100.0', 'fy': '100.0', 'cx': '79.5', 'cy': '63.5',
    'image_width': '160', 'image_height': '128',
    'max_depth_dist': '20.0', 'normalize_depth': 'false',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def yaml_simple_scalar(text, key):
    pat = r'(?m)^' + re.escape(key) + r':\s*([^#\r\n]+)'
    m = re.search(pat, text)
    if not m:
        raise ValueError('Missing root-level YAML field: ' + key)
    return m.group(1).strip().strip('\"\'')


def camera_section(text):
    # Require one root camera: block and avoid accidentally editing stereo.camera.
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r'^camera:\s*(?:#.*)?$', line.rstrip('\r\n'))]
    if len(starts) != 1:
        raise ValueError('Expected exactly one top-level camera: stanza')
    begin = starts[0] + 1
    end = next((i for i in range(begin, len(lines))
                if re.match(r'^[A-Za-z_][A-Za-z0-9_-]*:', lines[i])), len(lines))
    return lines, begin, end


def parse_camera(text):
    lines, a, b = camera_section(text)
    block = ''.join(lines[a:b])
    values = {}
    for k in list(TARGET) + ['pitch']:
        matches = re.findall(r'(?m)^  ' + re.escape(k) + r':\s*([^#\r\n]+)', block)
        if len(matches) != 1:
            raise ValueError('Expected one camera.' + k + ' field')
        values[k] = matches[0].strip().strip('\"\'')
    return values


def make_copy(original, root):
    lines, begin, end = camera_section(original)
    fields = set()
    for i in range(begin, end):
        m = re.match(r'^(  )([A-Za-z_][A-Za-z0-9_]*)(:\s*)([^#\r\n]*)(.*?)(\r?\n?)$', lines[i])
        if m and m.group(2) in TARGET:
            key = m.group(2)
            if key in fields:
                raise ValueError('Duplicate camera field: ' + key)
            fields.add(key)
            lines[i] = m.group(1) + key + m.group(3) + TARGET[key] + m.group(5) + m.group(6)
    if fields != set(TARGET):
        raise ValueError('Missing camera fields: ' + repr(sorted(set(TARGET)-fields)))
    # Make tree_file absolute, so running from a different working directory cannot silently change geometry.
    tree = root / 'Simulator/src/pointcloud/tree.ply'
    if not tree.is_file():
        raise ValueError('Tree PLY missing: ' + str(tree))
    hits = []
    for i, line in enumerate(lines):
        if re.match(r'^tree_file:', line):
            hits.append(i)
            suffix = ('\r\n' if line.endswith('\r\n') else '\n') if line.endswith('\n') else ''
            lines[i] = 'tree_file: "' + str(tree) + '"' + suffix
    if len(hits) != 1:
        raise ValueError('Expected exactly one top-level tree_file setting')
    updated = ''.join(lines)
    if yaml_simple_scalar(updated, 'maze_type') != '5' or yaml_simple_scalar(updated, 'random_map').lower() != 'true':
        raise ValueError('Refusing to prepare a config that is not random forest maze_type=5')
    got = parse_camera(updated)
    for k, value in TARGET.items():
        if got[k].lower() != value.lower():
            raise ValueError('Camera field did not update: ' + k + '=' + got[k])
    return updated


def quat_to_mat(q):
    x,y,z,w = [float(v) for v in q]
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or norm < 1e-10:
        raise ValueError('Invalid odometry quaternion')
    x,y,z,w = [v/norm for v in (x,y,z,w)]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ],dtype=np.float64)


def old_observation_check(root, src_config, obs_dir):
    if obs_dir is None:
        base = root/'query_minco/outputs/forest_f0_v1/preflight_observations'
        dirs = [d for d in base.glob('obs_*') if d.is_dir() and (d/'metadata.json').is_file() and (d/'observation.npz').is_file()] if base.exists() else []
        if not dirs:
            return {'status':'SKIPPED', 'reason':'No prior observation directory found'}
        obs_dir = sorted(dirs)[-1]
    obs_dir = obs_dir.expanduser().resolve()
    meta = json.loads((obs_dir/'metadata.json').read_text(encoding='utf-8'))
    data = np.load(obs_dir/'observation.npz', allow_pickle=False)
    depth = data['depth_m']
    hit = data['range_hit_mask'].astype(bool)
    maxmask = data['max_range_mask'].astype(bool)
    cm = parse_camera(src_config)
    H,W = depth.shape
    if [H,W] != [int(cm['image_height']),int(cm['image_width'])]:
        raise ValueError('Observed image shape does not match original config, cannot audit')
    if list(meta['camera_info']['K_row_major']) != [float(cm['fx']),0.,float(cm['cx']),0.,float(cm['fy']),float(cm['cy']),0.,0.,1.]:
        raise ValueError('Observed CameraInfo K mismatch with ORIGINAL config')
    if meta['depth']['frame_id'] != 'camera_link' or meta['depth']['encoding'] != '32FC1':
        raise ValueError('Unexpected depth frame or encoding')
    if meta['simulator_config_sha256'] != sha(root/'Simulator/src/config/config.yaml'):
        raise ValueError('Original config hash differs from recorded observation; stop and investigate')
    if not np.array_equal(hit,np.isfinite(depth)&(depth>0)&(depth<float(cm['max_depth_dist']))):
        raise ValueError('Invalid saved range_hit_mask')
    if not np.array_equal(maxmask,np.isfinite(depth)&(depth>=float(cm['max_depth_dist']))):
        raise ValueError('Invalid saved max_range_mask')
    pose = np.array(meta['odometry']['position_m'], dtype=np.float64)
    Rwb = quat_to_mat(meta['odometry']['orientation_xyzw'])
    pitch = math.radians(float(cm['pitch']))
    c,s = math.cos(pitch),math.sin(pitch)
    Rbc = np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    fx,fy,cx,cy = [float(cm[k]) for k in ('fx','fy','cx','cy')]
    sample_coords = [(W//2,H//2),(0,0),(W-1,0),(0,H-1),(W-1,H-1), (W//4,H//2), (3*W//4,H//2)]
    examples=[]
    for u,v in sample_coords:
        d = float(depth[v,u]);
        if not math.isfinite(d) or d <= 0 or d >= float(cm['max_depth_dist']):
            continue
        optical=np.array([(u-cx)*d/fx, (v-cy)*d/fy, d],dtype=np.float64)
        internal=np.array([optical[2],-optical[0],-optical[1]],dtype=np.float64)
        world=pose + Rwb @ (Rbc @ internal)
        recovered_internal=Rbc.T @ (Rwb.T @ (world-pose))
        recovered_u=cx-fx*recovered_internal[1]/recovered_internal[0]
        recovered_v=cy-fy*recovered_internal[2]/recovered_internal[0]
        err=math.hypot(recovered_u-u,recovered_v-v)
        if err>1e-5:
            raise ValueError('Coordinate roundtrip error at (%d,%d): %.4g'%(u,v,err))
        examples.append({'pixel_uv':[u,v],'depth_axial_m':d, 'point_body_forward_left_up_m':internal.tolist(),'point_world_m':world.tolist(),'reprojection_error_px':err})
    return {'status':'PASS_math_only_NOT_GT','observation_dir':str(obs_dir), 'observation_npz_sha256':sha(obs_dir/'observation.npz'),
        'timestamp_delta_s':meta['depth_minus_odom_time_s'], 'depth_valid_hit_ratio':float(hit.mean()),
        'camera_frame':meta['depth']['frame_id'], 'observed_intrinsics':[fx,fy,cx,cy],
        'reference_pose_world_m':pose.tolist(), 'samples':examples,
        'interpretation':'Assumes source-code x-forward ray convention and odometry camera pose; performs algebraic project/backproject test only. Does NOT validate world map GT, raycast accuracy, occlusion, or collision.'}


def exclusive_or_same(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError('File exists with different content; refusing to overwrite: '+str(path))
        return 'EXISTING_IDENTICAL'
    with path.open('xb') as f:
        f.write(raw)
    return 'CREATED'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--observation',type=Path,help='Directory containing observation.npz and metadata.json; defaults to newest')
    args=p.parse_args()
    root=args.root.expanduser().resolve()
    source=root/'Simulator/src/config/config.yaml'
    if not source.is_file():
        raise SystemExit('Cannot find source config: '+str(source))
    original=source.read_text(encoding='utf-8')
    audit=old_observation_check(root,original,args.observation)
    new_text=make_copy(original,root)
    dest=root/'query_minco/configs/forest_f0_camera_128x160_v1.yaml'
    target_status=exclusive_or_same(dest,new_text.encode('utf-8'))
    source_sha=sha(source)
    report={
        'schema':'F0_CAMERA_CONFIG_AND_COORDINATE_MATH_AUDIT_V1',
        'formal_F0_result':False,
        'source_config_path':str(source),'source_config_sha256':source_sha,
        'derived_config_path':str(dest),'derived_config_sha256':sha(dest),
        'source_camera':parse_camera(original),'proposed_camera':parse_camera(new_text),
        'tree_file_absolute':str(root/'Simulator/src/pointcloud/tree.ply'),
        'tree_file_sha256':sha(root/'Simulator/src/pointcloud/tree.ply'),
        'forest_seed':yaml_simple_scalar(original,'seed'),
        'old_observation_geometry_math_audit':audit,
        'safety_notes':[
            'No changes were made to Simulator/ or any existing ROS launch.',
            'New config does not take effect until the simulator is restarted with _config_file:=absolute path.',
            'Old f0_capture_static_observation.py assumes source Simulator/config.yaml and would MISLABEL a capture using derived config. Do not reuse it without a --config/active-config metadata fix.',
            'Offline projection roundtrip is NOT an independent map GT or collision check.',
            'Do not use the static odom->camera_link TF with +0.05m viz-only translation to reconstruct true rendered depth camera pose.',
            'Existing simulator binary contains old YOPO_exec_gap paths and must be independently rebuilt and audited before formal F0 reporting.'
        ]
    }
    out=root/'query_minco/outputs/forest_f0_v1/F0_CAMERA_PREP_AUDIT_V1.json'
    out_status=exclusive_or_same(out,(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Camera config: '+str(dest)+' ('+target_status+')')
    print('Camera config SHA256: '+sha(dest))
    print('Audit report: '+str(out)+' ('+out_status+')')
    print('Old observation geometry math check:',audit['status'])
    if audit.get('samples'):
        for s in audit['samples'][:3]:
            print('  pixel=%s z=%.4fm body_FLU=%s world=%s'%(s['pixel_uv'],s['depth_axial_m'], np.round(s['point_body_forward_left_up_m'],3).tolist(),np.round(s['point_world_m'],3).tolist()))
    print('NEXT: Restart only the existing Simulator ROS node with _config_file:='+str(dest))
    print('IMPORTANT: Do not yet use the old capture script for derived-config data; it hardcodes the original config in its metadata.')

if __name__=='__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        sys.exit('F0 STOP: '+str(exc))
