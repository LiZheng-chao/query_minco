#!/usr/bin/env python3
"""F0: certify six frozen Query-MINCO forest candidates against offline voxel map.

Read-only source files.  No model inference, training, controller, ROS, S1, or
online candidate selection.  Uses previously validated conservative checker.

This is ONLY a single pre-registered input's offline-voxel geometry diagnostic,
not original tree-mesh safety, entire-live-map identity, dynamics certificate,
reference-feasible task count, or forest Oracle Coverage.

Examples:
  python3 query_minco/scripts/f0_b1_forest_continuous_geometry_audit_v1.py --self-test
  python3 query_minco/scripts/f0_b1_forest_continuous_geometry_audit_v1.py \\
      --root . --decode-dir query_minco/outputs/forest_f0_v1/frozen_minco_decode_v1/decode_20261020T073437Z \\
      --metadata query_minco/outputs/forest_f0_v1/voxel_replicas/seed3_20261020T054914Z/occupancy_metadata.json
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from math import comb
from pathlib import Path
import sys

import numpy as np

# This file MUST be in the same directory as the independently preflighted checker.
from f0_continuous_swept_voxel_v1 import MAP_SHA, classify, load_world, synthetic_tests

RUNS = ('A42', 'A43', 'A44', 'B42')
AUDIT_NAME = 'F0_B1_FOREST_MINCO_DECODE_AUDIT_V1.json'
NPZ_NAME = 'F0_B1_FOREST_MINCO_TRAJECTORIES_V1.npz'
OUTPUT_NAME = 'F0_B1_FOREST_CONTINUOUS_GEOMETRY_AUDIT_V1.json'


def digest(p):
    h = sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_shape(a, shape, name):
    a = np.asarray(a)
    require(a.shape == shape and np.issubdtype(a.dtype, np.number) and np.isfinite(a).all(),
            f'{name}: expected finite array {shape}, got {a.shape} dtype={a.dtype}')
    return a


def quat_matrix(xyzw):
    q = finite_shape(xyzw, (4,), 'start quaternion').astype(np.float64)
    n = float(np.linalg.norm(q))
    require(abs(n-1.0) < 1e-5, f'Unnormalized B0-to-world quaternion: norm={n}')
    x, y, z, w = q / n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)


def power_to_bezier(coeff_3x6, T):
    c = np.asarray(coeff_3x6, dtype=np.float64)
    require(c.shape == (3, 6) and T > 0, 'Invalid power-basis segment')
    a = c * np.power(float(T), np.arange(6))[None, :]
    return np.stack([sum((comb(i,j)/comb(5,j))*a[:,j] for j in range(i+1))
                     for i in range(6)])


def frame_self_test():
    mat = quat_matrix([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    require(np.allclose(mat @ [1,0,0], [0,1,0], atol=1e-12), 'Frame rotation self-test')
    require(np.allclose(quat_matrix([0,0,0,1]), np.eye(3)), 'Identity quaternion self-test')
    print('B0_TO_WORLD_FRAME_SELF_TEST=PASS')


def run(args):
    frame_self_test()
    if args.self_test:
        synthetic_tests()
        print('F0_GEOMETRY_BRIDGE_SELF_TEST=PASS')
        return
    require(args.decode_dir is not None and args.metadata is not None,
            'Both --decode-dir and --metadata required')
    root = args.root.resolve()
    require((root / 'query_minco/planner.py').is_file(), 'Unexpected project root')
    decode = args.decode_dir.resolve()
    audit_path = decode / AUDIT_NAME
    npz_path = decode / NPZ_NAME
    prior = json.loads(audit_path.read_text(encoding='utf-8'))
    require(prior['schema'] == 'F0_B1_FOREST_MINCO_DECODE_SMOKE_V1' and
            prior['decision'] == 'MINCO_DECODE_SMOKE_PASS' and
            prior['collision_checker_run'] is False and
            prior['formal_forest_coverage'] is False and
            tuple(prior['runs'].keys()) == RUNS, 'Unexpected upstream decode audit')
    require(prior['offline_map_expected_sha256_for_next_step'] == MAP_SHA,
            'Decoder points to unexpected map')
    require(prior['goal_B0_m'] == [2.0, 0.0, 0.0],
            'This smoke is locked to the previously specified Near Goal (2,0,0)')
    world = load_world(args.metadata.resolve(), MAP_SHA)
    require(world.hash == MAP_SHA, 'Map identity differs')
    with np.load(npz_path, allow_pickle=False) as z:
        require(set(z.files) == {
            'run_names', 'coeffs_B0_power', 'durations_s', 'bezier_B0',
            'state_B0', 'start_world_m', 'start_world_q_xyzw'
        }, 'Unexpected trajectory NPZ keys')
        names = z['run_names'].tolist()
        coeffs = finite_shape(z['coeffs_B0_power'], (4,6,3,3,6), 'coeffs')
        durations = finite_shape(z['durations_s'], (4,6,3), 'durations')
        bez = finite_shape(z['bezier_B0'], (4,6,3,6,3), 'bezier')
        state = finite_shape(z['state_B0'], (12,), 'state')
        pos = finite_shape(z['start_world_m'], (3,), 'start position').astype('float64')
        quat = finite_shape(z['start_world_q_xyzw'], (4,), 'start orientation')
    require(tuple(names) == RUNS, f'Unexpected run ordering: {names}')
    require(np.array_equal(state[6:9], [2.,0.,0.]) and
            np.allclose(state[9:12], [0,0,-1], atol=1e-7),
            'Goal and gravity input differ from preregistered frozen smoke')
    require(np.allclose(pos, prior['start_position_world_m'], rtol=0, atol=1e-9),
            'Start position differs from decode audit')
    require(np.allclose(quat, prior['start_orientation_world_xyzw'], rtol=0, atol=1e-9),
            'Quaternion differs from decode audit')
    require(np.allclose(quat, [0,0,0,1], rtol=0, atol=1e-8),
            'This V1 bridge is restricted to the original level, identity-orientation capture')
    R = quat_matrix(quat)
    require(np.all(durations > 0.12 - 1e-5) and
            np.all((durations.sum(2) >= .8-1e-5) & (durations.sum(2) <= 2.2+1e-5)),
            'Segment/total duration outside B1 bound')

    runs, counts = {}, Counter()
    all_rows = []
    for i, name in enumerate(RUNS):
        existing = prior['runs'][name]
        for arr, key in [(coeffs[i], 'coeffs_sha256'),
                         (durations[i], 'durations_sha256'), (bez[i], 'bezier_sha256')]:
            require(sha256(arr.tobytes()).hexdigest() == existing[key],
                    f'{name} tampered {key}')
        sample_statuses = []
        rows = []
        for k in range(6):
            for seg in range(3):
                recalculated = power_to_bezier(coeffs[i,k,seg], float(durations[i,k,seg]))
                require(np.allclose(recalculated, bez[i,k,seg], rtol=0, atol=1e-9),
                        f'{name} query{k} segment{seg} failed power-to-Bezier recomputation')
            cp_world = [cp @ R.T + pos for cp in bez[i,k]]
            join_gap = max(float(np.linalg.norm(cp_world[seg][-1]-cp_world[seg+1][0]))
                           for seg in (0,1))
            require(join_gap <= 2e-3, f'{name} query{k}: C0 gap {join_gap} > 2mm')
            require(float(np.linalg.norm(cp_world[0][0]-pos)) <= 1e-4,
                    f'{name} query{k}: start position mismatch')
            geom = classify(world, cp_world, args.body_radius, args.margin,
                            args.max_depth, args.max_nodes)
            status = geom['status']
            counts[status] += 1
            sample_statuses.append(status)
            old = existing['candidates'][k]
            endpoint = (cp_world[-1][-1] - pos) @ R
            require(np.allclose(endpoint, old['pT_B0_m'], rtol=0, atol=5e-5),
                    f'{name} query{k}: endpoint differs from decode audit')
            goal_error = float(np.linalg.norm(endpoint - state[6:9]))
            require(abs(goal_error - float(old['near_goal_endpoint_error_m'])) < 5e-5,
                    f'{name} query{k}: goal error differs from decode audit')
            arrival = bool(old['near_goal_arrival_diagnostic_only'])
            row = {
                'query': k,
                'status': status,
                'continuous_voxel_geometry': geom,
                'max_segment_C0_gap_m': join_gap,
                'goal_B0_error_m': goal_error,
                'terminal_speed_mps': float(old['near_goal_terminal_speed_mps']),
                'near_arrival_diagnostic_only': arrival,
                'near_arrival_and_geometry_clear_diagnostic_only': bool(arrival and status == 'CLEAR_CERTIFIED'),
                'sampled_acceleration_violation_witness': bool(old['sampled_acceleration_violation_witness']),
                'sampled_speed_violation_witness': bool(old['sampled_speed_violation_witness']),
                'sampled_jerk_violation_witness': bool(old['sampled_jerk_violation_witness']),
                'sampled_dynamics_are_not_continuous_certificates': True,
                'full_physical_contract_certified': False,
            }
            rows.append(row)
            all_rows.append({'model': name, **row})
        c = Counter(sample_statuses)
        runs[name] = {
            'counts': {key: c[key] for key in ('CLEAR_CERTIFIED','MARGIN_ONLY','BODY_COLLISION','UNKNOWN')},
            'near_arrival_count_diagnostic_only': sum(row['near_arrival_diagnostic_only'] for row in rows),
            'near_arrival_and_geometry_clear_count_diagnostic_only': sum(row['near_arrival_and_geometry_clear_diagnostic_only'] for row in rows),
            'candidates': rows,
        }
        print(f'{name}: CLEAR_CERTIFIED={c["CLEAR_CERTIFIED"]} MARGIN_ONLY={c["MARGIN_ONLY"]} '
              f'BODY_COLLISION={c["BODY_COLLISION"]} UNKNOWN={c["UNKNOWN"]} '
              f'near_arrival_and_geometry_clear={runs[name]["near_arrival_and_geometry_clear_count_diagnostic_only"]}/6',
              flush=True)
    # An independently constructed straight centerline is only a GT GEOMETRY
    # diagnostic; it is never used as a candidate ranking or a physical teacher.
    line_end = pos + R @ np.asarray(state[6:9], dtype=np.float64)
    straight_ref = classify(world, [np.stack([pos, line_end])], args.body_radius, args.margin,
                            args.max_depth, args.max_nodes)
    print('STRAIGHT_LINE_GEOMETRY_REFERENCE=' + straight_ref['status'])
    summary = {key: counts[key] for key in ('CLEAR_CERTIFIED','MARGIN_ONLY','BODY_COLLISION','UNKNOWN')}
    require(sum(summary.values()) == 24, 'Exactly 24 candidates required')
    result = {
        'schema': 'F0_B1_FOREST_CONTINUOUS_GEOMETRY_AUDIT_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'single_task_diagnostic_only': True,
        'is_formal_forest_coverage': False,
        'reference_physically_feasible_proven': False,
        'collision_checker_against_offline_solid_voxel_proxy': True,
        'live_map_identity_proven': False,
        'swept_body_observability_verified': False,
        'continuous_dynamics_certified': False,
        'model_forward_rerun': False,
        'network_training_or_selection_or_ros': False,
        'map_sha256': world.hash,
        'map_metadata_path': str(args.metadata.resolve()),
        'trajectory_npz_sha256': digest(npz_path),
        'decoder_audit_sha256': digest(audit_path),
        'geometry_checker_source_sha256': digest(Path(__file__).with_name('f0_continuous_swept_voxel_v1.py')),
        'start_world_m': pos.tolist(),
        'goal_B0_m': state[6:9].tolist(),
        'body_radius_m': args.body_radius,
        'extra_margin_m': args.margin,
        'max_depth': args.max_depth,
        'max_nodes': args.max_nodes,
        'total_candidates': 24,
        'total_status_counts': summary,
        'models': runs,
        'straight_line_goal_geometry_reference': straight_ref,
        'not_full_success_reason': 'No complete continuous dynamics, observation support, physically verified reference denominator, live-map proof or multi-task statistics',
        'decision': 'CANDIDATE_CONTINUOUS_VOXEL_GEOMETRY_SMOKE_PASS'
    }
    output_path = (args.output.resolve() if args.output else
                   (root / 'query_minco/outputs/forest_f0_v1/frozen_swept_candidates_v1' /
                    decode.name / OUTPUT_NAME))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('ALL_24_CONTINUOUS_VOXEL_GEOMETRY=' + str(summary))
    print('F0_CANDIDATE_GEOMETRY_SMOKE=PASS')
    print('AUDIT=' + str(output_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--decode-dir', type=Path)
    ap.add_argument('--metadata', type=Path)
    ap.add_argument('--body-radius', type=float, default=.12)
    ap.add_argument('--margin', type=float, default=.03)
    ap.add_argument('--max-depth', type=int, default=16)
    ap.add_argument('--max-nodes', type=int, default=50000)
    ap.add_argument('--output', type=Path, help='New JSON audit path; existing file is never overwritten')
    args = ap.parse_args()
    require(args.body_radius == .12 and args.margin == .03,
            'F0 smoke fixed to registered body radius .12 and extra margin .03')
    run(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('F0_CANDIDATE_GEOMETRY_SMOKE=BLOCKED ' + type(exc).__name__+': '+str(exc), file=sys.stderr)
        sys.exit(1)
