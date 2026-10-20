#!/usr/bin/env python3
"""Query-MINCO F0: read-only, one-forest-frame, four-sealed-B1-model RAW forward smoke.

IMPORTANT:
* This is NOT an F0 coverage, feasibility, or safety measurement.
* Never trains, performs ROS actions, selects a candidate, or decodes MINCO.
* Uses B1 archived samples to validate a depth/mask rule and obtain the
  canonical B0 gravity for a level stationary pose, rather than guessing it.
* This experiment FIXES goal_B0=(2,0,0) m for an interface smoke only; it does
  not claim a GT-feasible or generalizable task.
* Only use with checkpoints from your own verified training pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUNS = ('A42', 'A43', 'A44', 'B42')
OBS_SHA = '967c30329d66a8588456b7fd1d3c0d05f1a4510baa96585be58e3d6ae0860dcf'
CAM_CFG_SHA = '45f3184f236cc11c0739fe689f84c4f1b0e58f1ada8ef11fd18ba25734e5bf81'
GEOM_SHA = '2c939ec328f56d7016061ea9ff0eb0a2ef4f458926bedfc2109bd3b128e29d35'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def state_digest(model):
    """Same state SHA order as p2c2c1b1_preflight.common.state_sha."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_b1_training_evidence(root, limit=16):
    """Read actual B1-source teacher archives, not fabricated network inputs."""
    dataset = root / 'query_minco/outputs/p2c2c1a_teacher_v1/dataset'
    manifest = read_json(dataset / 'manifest.json')
    rows = manifest.get('rows')
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError('Expected B1 teacher manifest rows (>=16), found: ' + repr(type(rows)))

    valid_total = invalid_total = 0
    maxerr_depth = 0.0
    invalid_depth_nonzero = 0
    states = []
    examples = []
    for row in rows[:limit]:
        src = dataset / row['npz']
        digest = sha256_file(src)
        if row.get('npz_sha256') != digest:
            raise ValueError('B1 dataset NPZ seal mismatch: ' + str(src))
        with np.load(src, allow_pickle=False) as z:
            for key in ('depth_input', 'depth_m', 'valid_mask', 'state'):
                if key not in z:
                    raise ValueError('Missing archived B1 dataset key: ' + key)
            depth_input = np.asarray(z['depth_input'])
            depth_m = np.asarray(z['depth_m'])
            valid_mask = np.asarray(z['valid_mask'])
            state = np.asarray(z['state'])
        if (depth_input.shape != (2, 128, 160) or depth_m.shape != (128, 160)
                or valid_mask.shape != (128, 160) or state.shape != (12,)):
            raise ValueError('B1 archived sample shape mismatch: ' + str(src))
        if not (np.isfinite(depth_input).all() and np.isfinite(depth_m).all() and np.isfinite(state).all()):
            raise ValueError('Nonfinite values in B1 archived input')
        if not np.array_equal(depth_input[1].astype(bool), valid_mask.astype(bool)):
            raise ValueError('B1 archived depth channel 2 differs from valid_mask')
        diff = float(np.max(np.abs(depth_input[0].astype(np.float64) * 20.0 - depth_m)))
        if diff > 2e-5:
            raise ValueError('B1 archived depth/20 normalization mismatch: ' + str(diff))
        maxerr_depth = max(maxerr_depth, diff)
        invalid = ~valid_mask.astype(bool)
        # Frozen dataset's documented invalid convention is depth_m=0.
        invalid_depth_nonzero += int(np.count_nonzero(depth_m[invalid]))
        valid_total += int(np.count_nonzero(~invalid))
        invalid_total += int(np.count_nonzero(invalid))
        states.append(state.astype(np.float64))
        examples.append({'id': row.get('id'), 'depth_sha256': hashlib.sha256(depth_input.tobytes()).hexdigest(),
                         'valid_fraction': float(np.mean(~invalid))})
    if invalid_depth_nonzero:
        raise ValueError('B1 archived invalid depth is not zero; STOP and inspect training source')
    states = np.stack(states)
    grav = states[:, 9:12]
    norms = np.linalg.norm(grav, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5):
        raise ValueError('B1 archived gravity vectors are not unit length')
    # For a level B0 frame, require unambiguous canonical gravity in training.
    g = np.array([0., 0., -1.]) if np.median(grav[:, 2]) < 0 else np.array([0., 0., 1.])
    matching_level = [i for i, gv in enumerate(grav) if np.linalg.norm(gv - g) < 1e-4]
    if not matching_level:
        raise ValueError('Could not establish level-B0 training gravity; do not guess its sign')
    return {
        'manifest': str((dataset / 'manifest.json').relative_to(root)),
        'manifest_sha256': sha256_file(dataset / 'manifest.json'),
        'checked_samples': len(examples),
        'valid_pixels_checked': valid_total,
        'invalid_pixels_checked': invalid_total,
        'invalid_depth_nonzero': invalid_depth_nonzero,
        'depth_over_20_max_abs_err_m': maxerr_depth,
        'unique_level_gravity_convention_B0': g.tolist(),
        'matching_level_gravity_count': len(matching_level),
        'sample_evidence': examples[:4],
    }, g.astype(np.float32)


def prepare_forest_input(pair, g, goal):
    meta = read_json(pair / 'metadata.json')
    obs_file = pair / 'observation_pair.npz'
    if sha256_file(obs_file) != OBS_SHA:
        raise ValueError('Forest observation SHA256 differs from prior sealed sample')
    if meta.get('observation_pair_npz_sha256') != OBS_SHA:
        raise ValueError('Forest pair metadata NPZ SHA mismatch')
    if meta.get('active_config_sha256') != CAM_CFG_SHA:
        raise ValueError('Forest active camera config SHA mismatch')
    cam = meta['active_camera_config']
    expected = {'fx': 100.0, 'fy': 100.0, 'cx': 79.5, 'cy': 63.5,
                'image_width': 160, 'image_height': 128, 'max_depth_dist': 20.0,
                'normalize_depth': False}
    if any(cam.get(k) != v for k, v in expected.items()):
        raise ValueError('Camera intrinsics, shape, max range, or metric depth differ')
    if not meta.get('samples'):
        raise ValueError('Paired observation lacks ROS odometry')
    odom = meta['samples'][0]['odometry']
    q = np.asarray(odom['orientation_xyzw'], dtype=np.float64)
    if not np.allclose(q, [0,0,0,1], atol=1e-5):
        raise ValueError('This smoke assumes a LEVEL, IDENTITY-ORIENTATION B0 pose; orientation differs')
    pos = np.asarray(odom['position_m'], dtype=np.float64)
    v0 = np.asarray(odom['linear_velocity_mps'], dtype=np.float64)
    if not (np.isfinite(pos).all() and np.isfinite(v0).all() and np.linalg.norm(v0) < 0.001):
        raise ValueError('Expected a known, stationary forest pose')
    with np.load(obs_file, allow_pickle=False) as z:
        depth = np.asarray(z['depth_m_first'], dtype=np.float32)
        ray_hit = np.asarray(z['range_hit_mask_first'], dtype=np.uint8)
        max_range = np.asarray(z['max_range_mask_first'], dtype=np.uint8)
    if depth.shape != (128, 160) or ray_hit.shape != depth.shape or max_range.shape != depth.shape:
        raise ValueError('Forest observation array shape mismatch')
    is_hit = np.isfinite(depth) & (depth > 0.0) & (depth < 20.0)
    no_hit = np.isfinite(depth) & (depth >= 20.0)
    if not np.array_equal(ray_hit.astype(bool), is_hit) or not np.array_equal(max_range.astype(bool), no_hit):
        raise ValueError('Capture range_hit/miss masks differ from metric depth policy')
    if not np.all(np.isfinite(depth)):
        # For this recorded pair finite ratio was 1.0; fail rather than silently remap.
        raise ValueError('Forest raw depth is nonfinite; unsupported by current smoke')
    inp = np.zeros((2, 128, 160), dtype=np.float32)
    inp[0, is_hit] = depth[is_hit] / np.float32(20.0)
    inp[1] = is_hit.astype(np.float32)
    state = np.zeros((12,), dtype=np.float32)
    state[:3] = v0.astype(np.float32)  # B0=world for identity pose.
    state[3:6] = 0.0                # Assumed due to stationary state; no acceleration sensor.
    state[6:9] = np.asarray(goal, dtype=np.float32)
    state[9:12] = g
    if not (np.isfinite(inp).all() and np.isfinite(state).all()):
        raise ValueError('Nonfinite network input')
    if not np.all(inp[0, ~is_hit] == 0):
        raise ValueError('Invalid pixels were not zeroed')
    return inp, state, {
        'paired_observation_sha256': OBS_SHA,
        'camera_config_sha256': CAM_CFG_SHA,
        'world_start_position_m': pos.tolist(),
        'world_start_orientation_xyzw': q.tolist(),
        'velocity_B0_mps': state[:3].tolist(),
        'acceleration_B0_mps2': state[3:6].tolist(),
        'acceleration_source': 'ASSUMED_ZERO_FOR_STATIC_SMOKE_NOT_MEASURED',
        'goal_B0_m': state[6:9].tolist(),
        'gravity_B0_unit': g.tolist(),
        'depth_conversion': 'hit: depth_m/20; no_hit_at_20m: (depth=0, valid_mask=0)',
        'range_hit_fraction': float(np.mean(is_hit)),
        'max_range_fraction': float(np.mean(no_hit)),
        'depth_input_sha256': hashlib.sha256(inp.tobytes()).hexdigest(),
        'state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
        'network_sees_map_GT': False,
    }


def import_checkpoint_helper(root):
    p = root / 'query_minco/scripts/f0_b1_checkpoint_architecture_audit_v2.py'
    if not p.exists():
        raise ValueError('Missing V2 safe checkpoint helper: ' + str(p))
    spec = importlib.util.spec_from_file_location('f0_ckpt_v2_helper', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--pair', type=Path, required=True)
    ap.add_argument('--goal', type=float, nargs=3, default=(2.0, 0.0, 0.0),
                    help='Declared goal_B0 in metres; default is a single Near-Goal smoke only')
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    pair = args.pair.expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = root / 'query_minco/outputs/forest_f0_v1/frozen_forward_smoke_v1' / ('smoke_' + stamp)
    else:
        output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite a prior smoke: ' + str(output))
    if not np.isfinite(args.goal).all() or np.linalg.norm(args.goal) > 3.4 or np.linalg.norm(args.goal) < 0.5:
        raise ValueError('This smoke requires a predeclared Near Goal with norm 0.5...3.4 metres')

    sys.path.insert(0, str(root))
    import torch
    from query_minco.planner import QueryMincoPlanner
    print('PYTHON:', sys.executable, 'TORCH:', torch.__version__, flush=True)

    # Verify independent sealed V2 audit before loading checkpoint.
    prior = read_json(root / 'query_minco/outputs/forest_f0_v1/frozen_bridge/F0_B1_ARCHITECTURE_LOAD_AUDIT_V2.json')
    if prior.get('gate') != 'READY_FOR_INPUT_CONTRACT_REVIEW' or not prior.get('all_four_architecture_match') or not prior.get('training_sources_all_frozen_hashes_match'):
        raise ValueError('F0 previous architecture audit is not PASS')
    freeze = read_json(root / 'query_minco/outputs/p2c2c1b1_train_v1/TRAINING_SOURCE_FREEZE.json')
    for relative, expected in freeze['training_sources'].items():
        if sha256_file(root / relative) != expected:
            raise ValueError('Frozen B1 source changed since training: ' + relative)

    training, gravity = read_b1_training_evidence(root)
    print('B1_ARCHIVED_INPUT_RULE=PASS samples=%d invalid_pixels=%d level_gravity=%s' % (
        training['checked_samples'], training['invalid_pixels_checked'], training['unique_level_gravity_convention_B0']), flush=True)

    inp, state, forest = prepare_forest_input(pair, gravity, args.goal)
    print('FOREST_INPUT=PASS depth=(1,2,128,160) state=(1,12) hit=%.4f far=%.4f' % (
        forest['range_hit_fraction'], forest['max_range_fraction']), flush=True)
    print('STATE_B0=', state.tolist(), flush=True)

    helper = import_checkpoint_helper(root)
    model = QueryMincoPlanner().cpu()
    if sum(p.numel() for p in model.parameters()) != 470461 or len(model.state_dict()) != 87:
        raise ValueError('QueryMincoPlanner no longer matches the frozen architecture')

    depth_tensor = torch.from_numpy(inp.copy()).unsqueeze(0)
    state_tensor = torch.from_numpy(state.copy()).unsqueeze(0)
    outputs = {}
    reports = {}
    for name in RUNS:
        directory = root / 'query_minco/outputs/p2c2c1b1_train_v1/runs' / name
        ckpt = directory / 'checkpoint_step5760.pt'
        seal = read_json(directory / 'checkpoint_step5760_SEALED.json')
        digest = sha256_file(ckpt)
        if (digest != helper.EXPECTED_B1_SHA256[name] or seal.get('sha256') != digest
                or seal.get('step') != 5760 or seal.get('sealed_before_evaluation') is not True):
            raise ValueError('Checkpoint SHA256/seal mismatch: ' + name)
        payload, allowed = helper.load_restricted_numpy_rng_checkpoint(torch, ckpt)
        if not isinstance(payload, dict) or 'state_dict' not in payload:
            raise ValueError('Unknown checkpoint format: ' + name)
        model.load_state_dict(payload['state_dict'], strict=True)
        model.eval()
        weight_hash_before = state_digest(model)
        if weight_hash_before != seal.get('weights_sha256'):
            raise ValueError('State weights hash mismatches B1 sealed record: ' + name)
        with torch.inference_mode():
            raw1 = model(depth_tensor, state_tensor)
            raw2 = model(depth_tensor, state_tensor)
        if (raw1.shape != (1, 6, 21) or raw1.dtype != torch.float32
                or not torch.isfinite(raw1).all()):
            raise ValueError('Model forward output fails shape/dtype/finiteness: ' + name)
        if not torch.equal(raw1, raw2):
            raise ValueError('Identical CPU eval forward produced different output: ' + name)
        if state_digest(model) != weight_hash_before:
            raise ValueError('Model weights mutated during smoke: ' + name)
        arr = raw1[0].detach().cpu().numpy()
        outputs[name] = arr
        reports[name] = {
            'checkpoint_sha256': digest, 'weights_sha256': weight_hash_before,
            'checkpoint_unsafe_numpy_globals': allowed,
            'raw_shape': list(arr.shape), 'raw_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
            'raw_range_min': float(arr.min()), 'raw_range_max': float(arr.max()),
            'cpu_eval_repeat_bitwise_equal': True, 'model_weights_unchanged': True,
            'trajectory_decoded': False,
            'raw_21d_spec': 'q1,q2,q3=0:9; vT=9:12; aT=12:15; T_logit=15; dt_logits=16:19; untrained_score=19; untrained_feas=20',
        }
        print(f'{name}: RAW_FORWARD_PASS raw=(1,6,21) stable=True sha256={reports[name]["raw_sha256"][:16]}', flush=True)

    output.mkdir(parents=True, exist_ok=False)
    for name in RUNS:
        with (output / (name + '_raw_6x21.npy')).open('xb') as f:
            np.save(f, outputs[name], allow_pickle=False)
    with (output / 'forest_input.npz').open('xb') as f:
        np.savez_compressed(f, depth_input=inp, state=state)
    document = {
        'schema': 'F0_B1_FOREST_RAW_FORWARD_SMOKE_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'is_formal_f0_result': False, 'is_oracle_coverage': False,
        'map_GT_used_in_model_input': False,
        'model_inference_run': True,
        'trajectory_minco_decode_run': False,
        'collision_verification_run': False,
        'candidate_selector_run': False,
        'training_run': False, 'ros_control_run': False,
        'repo_root': str(root),
        'input_archived_B1_evidence': training,
        'forest_observation_and_input': forest,
        'checkpoint_runs': reports,
        'goal_contract_label': 'Near-interface-smoke-only; reference feasibility not established',
        'caveat': 'Raw-network compatibility only; no physical trajectory/Goal/coverage claims and no selector calibration.',
        'decision': 'RAW_FORWARD_SMOKE_PASS',
    }
    with (output / 'F0_B1_FOREST_RAW_FORWARD_SMOKE_V1.json').open('x', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('F0_B1_RAW_FORWARD_SMOKE=PASS', flush=True)
    print('AUDIT=' + str(output / 'F0_B1_FOREST_RAW_FORWARD_SMOKE_V1.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('F0_B1_RAW_FORWARD_SMOKE=BLOCKED', type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
