#!/usr/bin/env python3
"""F0 frozen Query-MINCO: turn *archived* 6x21 raw predictions into MINCO polynomials.

Read-only wrt baseline/model/checkpoints. No ROS, training, selectors, or GT selection.
This script only validates the original B1 trajectory postprocessor and a
power-basis -> Bernstein/Bezier bridge. It does NOT certify collision safety,
continuous dynamical feasibility, observability, or F0 oracle coverage.

Run --self-test first. Then --root . --raw-dir <raw-forward-smoke-output>.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import sys
import numpy as np

RUNS = ('A42', 'A43', 'A44', 'B42')
RAW_AUDIT_NAME = 'F0_B1_FOREST_RAW_FORWARD_SMOKE_V1.json'
MAP_SHA = '2c939ec328f56d7016061ea9ff0eb0a2ef4f458926bedfc2109bd3b128e29d35'
EXPECTED_GOAL = np.array([2., 0., 0.], dtype=np.float32)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(2 << 20), b''):
            d.update(data)
    return d.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def power_to_bezier(coeff_3x6, duration_s):
    """p(t)=sum_k c[:,k]*t**k on [0,T] -> control points (6,3) in s=t/T.

    Exact Bernstein basis change in real arithmetic:
      b_i = sum_(j<=i) [C(i,j)/C(5,j)] c_j T**j.
    """
    c = np.asarray(coeff_3x6, dtype=np.float64)
    T = float(duration_s)
    if c.shape != (3, 6) or not np.isfinite(c).all() or not (np.isfinite(T) and T > 0):
        raise ValueError('Expected finite power coefficients [3,6] and positive duration')
    a = c * T ** np.arange(6, dtype=np.float64)[None, :]
    b = np.empty((6, 3), dtype=np.float64)
    for i in range(6):
        b[i, :] = sum((comb(i, j) / comb(5, j)) * a[:, j] for j in range(i+1))
    return b


def de_casteljau(points, u):
    p = np.asarray(points, dtype=np.float64).copy()
    for _ in range(len(p)-1):
        p = (1-u)*p[:-1] + u*p[1:]
    return p[0]


def power_derivative(c, t, derivative=0):
    """Value of position / velocity / acceleration / jerk for local segment time t."""
    c = np.asarray(c, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    for k in range(derivative, 6):
        factor = 1
        for j in range(derivative):
            factor *= k - j
        v += c[:, k] * factor * (t ** (k-derivative))
    return v


def self_test():
    # Order, scaling and control-point endpoint invariants, not a real B1 forward.
    c = np.array([[0., 1., 0.2, -0.1, 0.01, 0.003],
                  [0.1, 0.2, -0.3, 0.5, 0., 0.],
                  [1., -0.4, 0.15, 0.01, -0.02, 0.001]], dtype=np.float64)
    for T in (0.12, 0.37, 1.7, 2.2):
        bez = power_to_bezier(c, T)
        for u in np.linspace(0., 1., 53):
            p = power_derivative(c, T*u)
            b = de_casteljau(bez, u)
            if not np.allclose(p, b, rtol=0, atol=1e-11):
                raise AssertionError((T, u, p, b))
        if not np.allclose(bez[0], c[:, 0], rtol=0, atol=1e-12):
            raise AssertionError('Bezier endpoint 0')
        if not np.allclose(bez[-1], power_derivative(c, T), rtol=0, atol=1e-11):
            raise AssertionError('Bezier endpoint 1')
    print('POWER_TO_BEZIER_SYNTHETIC=PASS', flush=True)


def assert_finite_shape(array, expected, what):
    z = np.asarray(array)
    if z.shape != expected or not np.isfinite(z).all():
        raise ValueError('%s bad shape or nonfinite: %s expected %s' % (what, z.shape, expected))
    return z


def local_eval(coeff, durations):
    """Numerical checks independent of TrajectoryPostprocessor's samples."""
    assert_finite_shape(coeff, (6, 3, 3, 6), 'coeffs')
    assert_finite_shape(durations, (6, 3), 'durations')
    if np.any(durations <= 0):
        raise ValueError('Nonpositive MINCO segment duration')
    bez = np.empty((6, 3, 6, 3), dtype=np.float64)
    all_runs = []
    max_roundtrip = max_p_jump = max_v_jump = max_a_jump = 0.
    for q in range(6):
        C = np.asarray(coeff[q], dtype=np.float64)
        dt = np.asarray(durations[q], dtype=np.float64)
        qbez = np.empty((3, 6, 3), dtype=np.float64)
        per_seg = []
        for s in range(3):
            qbez[s] = power_to_bezier(C[s], dt[s])
            for u in np.linspace(0., 1., 13):
                p = power_derivative(C[s], float(dt[s]*u))
                b = de_casteljau(qbez[s], float(u))
                max_roundtrip = max(max_roundtrip, float(np.linalg.norm(p-b, ord=np.inf)))
            ts = np.linspace(0., float(dt[s]), 257)
            mx = [max(float(np.linalg.norm(power_derivative(C[s], float(t), derivative=k))) for t in ts)
                  for k in (1, 2, 3)]
            per_seg.append(mx)
            if s:
                max_p_jump = max(max_p_jump, float(np.linalg.norm(
                    power_derivative(C[s-1], dt[s-1]) - power_derivative(C[s], 0))))
                max_v_jump = max(max_v_jump, float(np.linalg.norm(
                    power_derivative(C[s-1], dt[s-1], 1) - power_derivative(C[s], 0, 1))))
                max_a_jump = max(max_a_jump, float(np.linalg.norm(
                    power_derivative(C[s-1], dt[s-1], 2) - power_derivative(C[s], 0, 2))))
        bez[q] = qbez
        sampled = np.max(per_seg, axis=0)
        all_runs.append({
            'query': q, 'segment_durations_s': dt.tolist(), 'total_duration_s': float(sum(dt)),
            'p0_B0_m': power_derivative(C[0], 0).tolist(),
            'v0_B0_mps': power_derivative(C[0], 0, 1).tolist(),
            'a0_B0_mps2': power_derivative(C[0], 0, 2).tolist(),
            'pT_B0_m': power_derivative(C[-1], dt[-1]).tolist(),
            'vT_B0_mps': power_derivative(C[-1], dt[-1], 1).tolist(),
            'aT_B0_mps2': power_derivative(C[-1], dt[-1], 2).tolist(),
            'sampled_max_speed_mps': float(sampled[0]),
            'sampled_max_acceleration_mps2': float(sampled[1]),
            'sampled_max_jerk_mps3': float(sampled[2]),
            'sampled_speed_violation_witness': bool(sampled[0] > 3.0+1e-7),
            'sampled_acceleration_violation_witness': bool(sampled[1] > 8.0+1e-7),
            'sampled_jerk_violation_witness': bool(sampled[2] > 100.0+1e-7),
        })
    return bez, all_runs, {
        'power_bezier_roundtrip_max_axis_error_m': max_roundtrip,
        'segment_position_jump_max_m': max_p_jump,
        'segment_velocity_jump_max_mps': max_v_jump,
        'segment_acceleration_jump_max_mps2': max_a_jump,
        'sampling_dynamics_is_not_continuous_certificate': True,
        'samples_per_segment_for_dynamics': 257,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true', help='Run power->Bezier unit tests, no project needed')
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--raw-dir', type=Path, help='Output directory of the F0 raw-forward smoke')
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return
    if args.raw_dir is None:
        ap.error('--raw-dir is required unless --self-test is supplied')
    root = args.root.resolve()
    source = args.raw_dir.resolve()
    if not (root / 'query_minco/planner.py').is_file():
        raise ValueError('Unexpected project root')
    raw_audit_file = source / RAW_AUDIT_NAME
    prior = json_read(raw_audit_file)
    if (prior.get('decision') != 'RAW_FORWARD_SMOKE_PASS' or
            prior.get('trajectory_minco_decode_run') is not False or
            prior.get('is_formal_f0_result') is not False or
            set(prior['checkpoint_runs']) != set(RUNS)):
        raise ValueError('Prior raw forward audit is not the required F0 PASS')
    obs = prior['forest_observation_and_input']
    if (obs['paired_observation_sha256'] != '967c30329d66a8588456b7fd1d3c0d05f1a4510baa96585be58e3d6ae0860dcf'
            or obs['camera_config_sha256'] != '45f3184f236cc11c0739fe689f84c4f1b0e58f1ada8ef11fd18ba25734e5bf81'):
        raise ValueError('Unexpected forest input source: STOP')
    with np.load(source / 'forest_input.npz', allow_pickle=False) as zin:
        dep = assert_finite_shape(zin['depth_input'], (2, 128, 160), 'forest depth')
        state = assert_finite_shape(zin['state'], (12,), 'forest state')
    if (hashlib.sha256(dep.tobytes()).hexdigest() != obs['depth_input_sha256'] or
            hashlib.sha256(state.tobytes()).hexdigest() != obs['state_sha256']):
        raise ValueError('Forest input does not match recorded audit sha')
    if not np.array_equal(state[6:9], EXPECTED_GOAL):
        raise ValueError('Unexpected goal, this script only handles the fixed (2,0,0) Near smoke')
    if not np.allclose(state[9:12], [0, 0, -1], rtol=0, atol=1e-7):
        raise ValueError('Unexpected gravity convention')
    from_path = root / 'query_minco/trajectory'
    sys.path.insert(0, str(root))
    import torch
    from query_minco.trajectory import TrajectoryPostprocessor
    decoder = TrajectoryPostprocessor()
    runs = {}
    coeffs_all, durations_all, bezier_all = [], [], []
    for name in RUNS:
        raw_path = source / f'{name}_raw_6x21.npy'
        raw = np.load(raw_path, allow_pickle=False)
        assert_finite_shape(raw, (6, 21), f'{name} raw')
        raw_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        if raw_sha != prior['checkpoint_runs'][name]['raw_sha256']:
            raise ValueError('%s raw SHA differs from frozen-forward audit' % name)
        # Reuse original (not rewritten) B1 PhysicalParameterDecoder + MINCO code.
        with torch.inference_mode():
            out = decoder(torch.from_numpy(raw.astype(np.float32, copy=False)).unsqueeze(0),
                          torch.from_numpy(state.astype(np.float32, copy=False)).unsqueeze(0),
                          num_samples=513)
        c = out.trajectory.coeffs.detach().cpu().numpy()
        dt = out.parameters.durations.detach().cpu().numpy()
        assert_finite_shape(c, (1, 6, 3, 3, 6), f'{name} coeffs')
        assert_finite_shape(dt, (1, 6, 3), f'{name} durations')
        bez, candidates, checks = local_eval(c[0], dt[0])
        q0 = np.asarray([q['p0_B0_m'] for q in candidates])
        v0 = np.asarray([q['v0_B0_mps'] for q in candidates])
        a0 = np.asarray([q['a0_B0_mps2'] for q in candidates])
        checks['max_start_position_error_m'] = float(np.max(np.linalg.norm(q0, axis=1)))
        checks['max_start_velocity_error_mps'] = float(np.max(np.linalg.norm(v0-state[:3], axis=1)))
        checks['max_start_acceleration_error_mps2'] = float(np.max(np.linalg.norm(a0-state[3:6], axis=1)))
        checks['duration_window_pass'] = bool(np.all((dt[0] >= 0.12 - 1e-5)) and
                                             np.all((dt[0].sum(axis=1) >= 0.8 - 1e-5)) and
                                             np.all((dt[0].sum(axis=1) <= 2.2 + 1e-5)))
        checks['power_bezier_pass'] = checks['power_bezier_roundtrip_max_axis_error_m'] <= 1e-5
        checks['minco_pva_continuity_pass'] = (checks['segment_position_jump_max_m'] <= 2e-3 and
                checks['segment_velocity_jump_max_mps'] <= 2e-3 and
                checks['segment_acceleration_jump_max_mps2'] <= 2e-2)
        checks['initial_pva_pass'] = (checks['max_start_position_error_m'] <= 1e-4 and
                checks['max_start_velocity_error_mps'] <= 2e-3 and
                checks['max_start_acceleration_error_mps2'] <= 2e-2)
        for q in candidates:
            q['near_goal_endpoint_error_m'] = float(np.linalg.norm(np.asarray(q['pT_B0_m']) - state[6:9]))
            q['near_goal_terminal_speed_mps'] = float(np.linalg.norm(np.asarray(q['vT_B0_mps'])))
            q['near_goal_arrival_diagnostic_only'] = (q['near_goal_endpoint_error_m'] <= 0.15 and
                                                      q['near_goal_terminal_speed_mps'] <= 0.25)
        runs[name] = {
            'raw_sha256': raw_sha,
            'coeffs_sha256': hashlib.sha256(c[0].tobytes()).hexdigest(),
            'durations_sha256': hashlib.sha256(dt[0].tobytes()).hexdigest(),
            'bezier_sha256': hashlib.sha256(bez.tobytes()).hexdigest(),
            'invariants': checks,
            'candidates': candidates,
            'full_goal_and_geometry_feasibility_verified': False,
        }
        coeffs_all.append(c[0]); durations_all.append(dt[0]); bezier_all.append(bez)
        passed = all(checks[k] for k in ('duration_window_pass','power_bezier_pass',
                                        'minco_pva_continuity_pass','initial_pva_pass'))
        print('%s: %s T=[%.3f,%.3f] C2jump=[%.2g,%.2g,%.2g] goal_err=[%.3f,%.3f] sample_speed_max=%.3f' % (
            name, 'MINCO_DECODE_PASS' if passed else 'MINCO_DECODE_REVIEW',
            float(dt[0].sum(axis=1).min()), float(dt[0].sum(axis=1).max()),
            checks['segment_position_jump_max_m'],checks['segment_velocity_jump_max_mps'],
            checks['segment_acceleration_jump_max_mps2'],
            min(q['near_goal_endpoint_error_m'] for q in candidates),
            max(q['near_goal_endpoint_error_m'] for q in candidates),
            max(q['sampled_max_speed_mps'] for q in candidates)
        ), flush=True)
    source_dir = root / 'query_minco/trajectory'
    source_hashes = {}
    if source_dir.is_dir():
        for p in sorted(source_dir.rglob('*.py')):
            source_hashes[str(p.relative_to(root))] = sha(p)
    else:
        for p in (root/'query_minco/trajectory.py',):
            if p.is_file():source_hashes[str(p.relative_to(root))] = sha(p)
    if not source_hashes:
        raise ValueError('Could not audit original trajectory postprocessor source hashes')
    complete = all(all(runs[name]['invariants'][k] for k in (
        'duration_window_pass','power_bezier_pass','minco_pva_continuity_pass','initial_pva_pass')) for name in RUNS)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir.resolve() if args.output_dir else (root /
            'query_minco/outputs/forest_f0_v1/frozen_minco_decode_v1' / ('decode_' + stamp))
    if output.exists():
        raise FileExistsError('Refusing overwrite: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'F0_B1_FOREST_MINCO_TRAJECTORIES_V1.npz').open('xb') as f:
        np.savez_compressed(f, run_names=np.asarray(RUNS),
            coeffs_B0_power=np.asarray(coeffs_all), durations_s=np.asarray(durations_all),
            bezier_B0=np.asarray(bezier_all), state_B0=state,
            start_world_m=np.asarray(obs['world_start_position_m'], dtype=np.float64),
            start_world_q_xyzw=np.asarray(obs['world_start_orientation_xyzw'], dtype=np.float64))
    report = {
        'schema': 'F0_B1_FOREST_MINCO_DECODE_SMOKE_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'formal_forest_coverage': False, 'collision_checker_run': False,
        'network_forward_rerun': False, 'training_performed': False,
        'ros_commands_issued': False, 'selector_performed': False,
        'forest_input_raw_audit_sha256': sha(raw_audit_file),
        'forest_input_observation_sha256': obs['paired_observation_sha256'],
        'offline_map_expected_sha256_for_next_step': MAP_SHA,
        'goal_B0_m': state[6:9].tolist(),
        'start_position_world_m': obs['world_start_position_m'],
        'start_orientation_world_xyzw': obs['world_start_orientation_xyzw'],
        'initial_acceleration_assumed_zero': True,
        'trajectory_basis': 'per-segment ascending-time power coeffs [run,query,segment,xyz,power]; Bernstein controls in normalized local time',
        'trajectory_source_file_sha256': source_hashes,
        'candidate_array_file': 'F0_B1_FOREST_MINCO_TRAJECTORIES_V1.npz',
        'runs': runs,
        'decision': 'MINCO_DECODE_SMOKE_PASS' if complete else 'BLOCKED_MINCO_NUMERICAL_INVARIANTS',
        'limitations': 'One stationary Near task; no ground-truth reference feasibility; sampled dynamics maxima are NOT continuous certifications; no collision/observability checks.'
    }
    with (output/'F0_B1_FOREST_MINCO_DECODE_AUDIT_V1.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print('F0_B1_MINCO_DECODE_SMOKE=%s' % ('PASS' if complete else 'BLOCKED'), flush=True)
    print('AUDIT='+str(output/'F0_B1_FOREST_MINCO_DECODE_AUDIT_V1.json'), flush=True)
    print('TRAJECTORIES='+str(output/'F0_B1_FOREST_MINCO_TRAJECTORIES_V1.npz'), flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('F0_B1_MINCO_DECODE_SMOKE=BLOCKED '+type(exc).__name__+': '+str(exc),
              file=sys.stderr, flush=True)
        sys.exit(1)
