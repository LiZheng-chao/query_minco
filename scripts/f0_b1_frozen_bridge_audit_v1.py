#!/usr/bin/env python3
"""Read-only F0 Frozen bridge audit for Query-MINCO B1 step5760.

Locates four frozen B1 checkpoints and seals, hashes them, verifies the
forest paired observation and its camera contract, and records unresolved
semantic gates before any model inference. No torch.load / training / ROS.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_load(p):
    with p.open(encoding='utf-8') as f:
        return json.load(f)


def relative_or_absolute(p, root):
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def checkpoint_candidates(directory):
    extensions = {'.pt', '.pth', '.ckpt', '.safetensors'}
    files = []
    if directory.is_dir():
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix.lower() in extensions and '5760' in p.name.lower():
                files.append(p)
    return sorted(files)


def source_probe(root):
    names = ('model.py', 'depth_encoder.py', 'minco.py', 'trajectory.py',
             'dataset.py', 'planner.py', 'network.py', 'losses.py', 'inference.py')
    basedir = root / 'query_minco'
    result = []
    if basedir.exists():
        for name in names:
            p = basedir / name
            if p.is_file():
                result.append({'path': relative_or_absolute(p, root), 'sha256': sha256(p)})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--pair', type=Path, required=True,
                    help='Directory containing observation_pair.npz and metadata.json')
    ap.add_argument('--continuous-audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    pair = args.pair.resolve()
    f0_audit_path = args.continuous_audit.resolve()
    output = args.output or (root/'query_minco/outputs/forest_f0_v1/frozen_bridge/F0_B1_FROZEN_BRIDGE_AUDIT_V1.json')
    output = output.resolve()
    if not (root / 'query_minco').is_dir():
        ap.error('Not a Query-MINCO root: '+str(root))
    import numpy as np
    observations = pair / 'observation_pair.npz'
    pair_metadata = pair / 'metadata.json'
    result = {'schema': 'F0_B1_FROZEN_BRIDGE_AUDIT_V1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'purpose': 'Read-only frozen checkpoint and forest-input evidence collection',
              'model_forward_performed': False,
              'checkpoint_deserialization_performed': False,
              'training_or_ros_commands_performed': False,
              'root': str(root),
              'gates': {},
              'runs': {},
              'observation': {},
              'model_sources': source_probe(root),
              'remaining_requirements': [
                  'Confirm input channel-2 validity mask semantics from existing B1 preprocessing implementation',
                  'Confirm depth clip/normalization and camera intrinsics from existing B1 model data loader',
                  'Confirm 12D state ordering, B0 frame, gravity sign, and dynamics units',
                  'Prerecord start pose, Near/Far Goal and velocity/acceleration tasks; do not use GT to select network outputs',
                  'Verify original MINCO decoding and exact power-to-Bezier conversion using continuity / derivative roundtrips',
                  'Audit active simulator executable provenance for formal F0',
                  'Do not claim forest coverage without reference-feasible task denominator'
              ]}
    try:
        git_head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=10)
        if git_head.returncode == 0:
            result['git_head'] = git_head.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    base = root/'query_minco/outputs/p2c2c1b1_train_v1/runs'
    all_ok = True
    for run in ('A42', 'A43', 'A44', 'B42'):
        run_dir = base / run
        seal = run_dir / 'checkpoint_step5760_SEALED.json'
        candidates = checkpoint_candidates(run_dir)
        entry = {'run_dir': relative_or_absolute(run_dir, root),
                 'run_directory_present': run_dir.is_dir(),
                 'seal_exists': seal.is_file(),
                 'seal_sha256': sha256(seal) if seal.is_file() else None,
                 'seal_top_level_keys': [],
                 'step5760_checkpoint_candidates': [],
                 'status': 'BLOCKED'}
        if seal.is_file():
            try:
                contents = json_load(seal)
                entry['seal_top_level_keys'] = list(contents.keys()) if isinstance(contents, dict) else []
            except Exception as e:
                entry['seal_parse_error'] = str(e)
        for p in candidates:
            entry['step5760_checkpoint_candidates'].append({
                'path': relative_or_absolute(p, root), 'size_bytes': p.stat().st_size,
                'sha256': sha256(p)})
        entry['status'] = ('FILES_PRESENT_REVIEW_SEAL_CONTENT'
                           if len(candidates)==1 and seal.is_file() and 'seal_parse_error' not in entry
                           else 'BLOCKED_MISSING_OR_AMBIGUOUS_CHECKPOINT')
        if entry['status'].startswith('BLOCKED'):
            all_ok = False
        result['runs'][run] = entry
        print(f'{run}: {entry["status"]}, checkpoint candidates={len(candidates)}, seal={entry["seal_exists"]}')
        if len(candidates)==1:
            print('    checkpoint: '+entry['step5760_checkpoint_candidates'][0]['path'])
    result['gates']['four_checkpoint_files_located'] = all_ok
    result['gates']['checkpoint_architecture_loaded_and_matched'] = False
    if not observations.is_file() or not pair_metadata.is_file():
        raise FileNotFoundError('Missing observation pair files: '+str(pair))
    meta = json_load(pair_metadata)
    with np.load(observations, allow_pickle=False) as pack:
        required = ['depth_m_first', 'depth_m_second', 'range_hit_mask_first', 'max_range_mask_first']
        if any(n not in pack.files for n in required):
            raise ValueError('Missing NPZ keys: '+','.join([n for n in required if n not in pack.files]))
        a = pack['depth_m_first'].astype(np.float32,copy=False)
        b = pack['depth_m_second'].astype(np.float32,copy=False)
        hit = pack['range_hit_mask_first'].astype(bool)
        far = pack['max_range_mask_first'].astype(bool)
        valid_pos = np.isfinite(a) & (a > 0)
        valid_hit = valid_pos & (a < 20.)
        if a.shape!=(128,160) or b.shape!=(128,160) or a.dtype!=np.float32:
            raise ValueError('Invalid paired forest depth shape/dtype')
        if not np.array_equal(hit, valid_hit) or np.any(hit & far):
            raise ValueError('Depth range-hit masks do not match values')
        if not np.all(np.isfinite(a)):
            raise ValueError('Nonfinite depth; separate policy audit required')
        if not np.array_equal(a,b):
            print('NOTE: paired depth differs; record before inference')
        sha_npz=sha256(observations)
        cam=meta.get('active_camera_config', {})
        intrinsics_ok = (cam.get('image_height')==128 and cam.get('image_width')==160 and
                         cam.get('fx')==100. and cam.get('fy')==100. and
                         cam.get('cx')==79.5 and cam.get('cy')==63.5 and
                         cam.get('normalize_depth') is False and cam.get('max_depth_dist')==20.)
        result['observation'] = {
            'paired_npz': str(observations), 'paired_npz_sha256': sha_npz,
            'npz_sha_matches_metadata': sha_npz==meta.get('observation_pair_npz_sha256'),
            'active_config_sha256': meta.get('active_config_sha256'),
            'shape_hw': list(a.shape), 'depth_dtype': str(a.dtype),
            'camera_intrinsics_and_range_match_expected': intrinsics_ok,
            'two_depth_frames_bitwise_equal': bool(np.array_equal(a,b)),
            'finite_positive_ratio': float(valid_pos.mean()),
            'range_hit_ratio': float(valid_hit.mean()),
            'max_range_ratio': float(far.mean()),
            'potential_mask_semantics_disagreement_ratio': float(np.logical_xor(valid_pos,valid_hit).mean()),
            'state_from_odom': meta.get('samples', [{}])[0].get('odometry', {}),
            'goal_is_not_present_in_observation': True,
            'network_mask_policy_confirmed': False}
    result['gates']['forest_input_camera_evidence_ok'] = (intrinsics_ok and result['observation']['npz_sha_matches_metadata'])
    ray = json_load(f0_audit_path)
    classes = {k:v.get('result',{}).get('status') for k,v in ray.get('controls',{}).items()}
    expected={'short_clear_line':'CLEAR_CERTIFIED', 'known_collision_line':'BODY_COLLISION',
              'short_clear_quintic_bezier':'CLEAR_CERTIFIED', 'out_of_map_line':'UNKNOWN'}
    ray_ok = (ray.get('schema')=='F0_CONTINUOUS_SWEPT_VOXEL_PREFLIGHT_V1' and
              classes == expected and ray.get('map_sha256') == '2c939ec328f56d7016061ea9ff0eb0a2ef4f458926bedfc2109bd3b128e29d35')
    result['continuous_geometry']={'audit_sha256':sha256(f0_audit_path), 'classifications':classes,
                                   'expected_preflight_controls_match':ray_ok,
                                   'limitations':'Artificial controls only; not a candidate MINCO certificate.'}
    result['gates']['geometry_controls_match'] = ray_ok
    result['gates']['ready_for_formal_frozen_forest_coverage'] = False
    result['decision'] = ('READY_FOR_READONLY_INTERFACE_REVIEW' if all_ok and intrinsics_ok and ray_ok
                          else 'BLOCKED_LOCATE_OR_CHECK_FILES')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Forest camera={result["gates"]["forest_input_camera_evidence_ok"]}, geometry controls={ray_ok}')
    print('Two competing validity masks (NOT selected): finite-positive ratio=%.4f vs range-hit ratio=%.4f' %
          (result['observation']['finite_positive_ratio'],result['observation']['range_hit_ratio']))
    print('GATE_STATUS='+result['decision'])
    print('MODEL_INFERENCE=NOT_RUN')
    print('AUDIT='+str(output))
    return 0 if result['decision']=='READY_FOR_READONLY_INTERFACE_REVIEW' else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('AUDIT_FAILED: '+str(exc),file=sys.stderr)
        sys.exit(3)
