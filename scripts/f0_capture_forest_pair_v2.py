#!/usr/bin/env python3
"""F0 preflight: record two synchronized, static YOPO-MINCO forest observations.

Records the configuration path actually selected by the running ROS simulator
(private _config_file parameter), not the original/default YAML path.

Effects: publishes NO messages; does NO training/inference; never subscribes
/mock_map; creates ONLY a new timestamped directory under query_minco/outputs/.
This is a sensor-input smoke test, NOT a map GT, safety, or formal F0 result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import yaml


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def time_s(stamp):
    return stamp.secs + stamp.nsecs * 1e-9


def vec3(v):
    return [float(v.x), float(v.y), float(v.z)]


def quat4(v):
    return [float(v.x), float(v.y), float(v.z), float(v.w)]


def image_to_numpy(msg):
    if msg.encoding != '32FC1':
        raise ValueError('Expected 32FC1 depth, got ' + msg.encoding)
    if msg.width <= 0 or msg.height <= 0 or msg.step < 4 * msg.width:
        raise ValueError('Malformed image width/height/step')
    if len(msg.data) < msg.step * msg.height:
        raise ValueError('Truncated image data')
    dt = np.dtype('>f4' if msg.is_bigendian else '<f4')
    arr = np.ndarray((msg.height, msg.width), dtype=dt,
                     buffer=msg.data, strides=(msg.step, 4))
    return arr.astype(np.float32, copy=True)


def parse_config(path):
    with path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError('YAML top level is not a mapping')
    cam = raw.get('camera')
    if not isinstance(cam, dict):
        raise ValueError('Missing camera mapping')
    return raw, cam


def same_float(actual, expected, label, tolerance=1e-6):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise ValueError('{} mismatch: ROS={}, running config={}'.format(label, actual, expected))


def validate_contract(cfg, cam, info, msg, expected_h, expected_w):
    if cfg.get('random_map') is not True or int(cfg.get('maze_type', -1)) != 5:
        raise ValueError('Active ROS config is not random forest (random_map:true, maze_type:5)')
    if bool(cam.get('normalize_depth')):
        raise ValueError('Active simulator normalizes depth; expected raw meters')
    if (int(cam['image_width']), int(cam['image_height'])) != (expected_w, expected_h):
        raise ValueError('Active YAML camera dimensions do not match expected dimensions')
    for thing, width, height in (('CameraInfo', info.width, info.height),
                                 ('Image', msg.width, msg.height)):
        if (width, height) != (expected_w, expected_h):
            raise ValueError('{} size {}x{} differs from config expected {}x{}'.format(
                thing, width, height, expected_w, expected_h))
    k = list(info.K)
    if len(k) != 9:
        raise ValueError('CameraInfo K is not 3x3')
    for val, key, pos in (('fx', 'fx', 0), ('fy', 'fy', 4),
                          ('cx', 'cx', 2), ('cy', 'cy', 5)):
        same_float(k[pos], cam[key], 'CameraInfo K.' + val)
    if msg.header.frame_id != info.header.frame_id or msg.header.frame_id != 'camera_link':
        raise ValueError('Unexpected depth/CameraInfo frame IDs: {!r} / {!r}'.format(
            msg.header.frame_id, info.header.frame_id))
    if abs(k[1]) > 1e-7 or abs(k[3]) > 1e-7 or abs(k[6]) > 1e-7 or abs(k[7]) > 1e-7 or abs(k[8]-1) > 1e-7:
        raise ValueError('CameraInfo K is not zero-skew pinhole calibration')
    if int(cfg.get('seed', -1)) < 0:
        raise ValueError('Missing/nonstandard map seed')
    maxd = float(cam['max_depth_dist'])
    if not math.isfinite(maxd) or maxd <= 0:
        raise ValueError('Bad max_depth_dist')
    return maxd


def normalize_quat(q):
    arr = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(arr)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('Invalid odom quaternion')
    return arr / n


def odom_as_dict(msg):
    return {
        'timestamp_s': time_s(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'child_frame_id': msg.child_frame_id,
        'position_m': vec3(msg.pose.pose.position),
        'orientation_xyzw': quat4(msg.pose.pose.orientation),
        'linear_velocity_mps': vec3(msg.twist.twist.linear),
        'angular_velocity_rps': vec3(msg.twist.twist.angular),
    }


def calc_stats(a, b, maxd):
    valid_a = np.isfinite(a) & (a > 0.0) & (a < maxd)
    valid_b = np.isfinite(b) & (b > 0.0) & (b < maxd)
    far_a = np.isfinite(a) & (a >= maxd)
    far_b = np.isfinite(b) & (b >= maxd)
    common = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[common].astype(np.float64)-b[common].astype(np.float64))
    return {
        'finite_ratio_first': float(np.isfinite(a).mean()),
        'finite_ratio_second': float(np.isfinite(b).mean()),
        'range_hit_ratio_first': float(valid_a.mean()),
        'range_hit_ratio_second': float(valid_b.mean()),
        'max_range_ratio_first': float(far_a.mean()),
        'max_range_ratio_second': float(far_b.mean()),
        'range_hit_mask_disagreement_ratio': float(np.mean(valid_a != valid_b)),
        'max_range_mask_disagreement_ratio': float(np.mean(far_a != far_b)),
        'abs_depth_difference_m_max': float(error.max()) if error.size else None,
        'abs_depth_difference_m_p95': float(np.percentile(error, 95)) if error.size else None,
        'changed_over_10cm_ratio': float(np.mean(error > .10)) if error.size else None,
        'first_depth_m_min': float(a[np.isfinite(a)].min()) if np.isfinite(a).any() else None,
        'first_depth_m_max': float(a[np.isfinite(a)].max()) if np.isfinite(a).any() else None,
    }, valid_a, far_a


def git_head(root):
    x = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True, check=False)
    return x.stdout.strip() if x.returncode == 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('~/reach_planner').expanduser())
    p.add_argument('--expected-config-sha256', required=True,
                   help='Pinned SHA256 of the independently prepared F0 YAML')
    p.add_argument('--timeout', type=float, default=25.0)
    p.add_argument('--gap-seconds', type=float, default=2.0)
    p.add_argument('--simulator-node', default='/sensor_simulator_node')
    args = p.parse_args()
    if args.gap_seconds < 0.5 or args.timeout <= 0:
        raise ValueError('Require gap-seconds >= 0.5 and timeout > 0')
    expected_sha = args.expected_config_sha256.lower()
    if len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
        raise ValueError('--expected-config-sha256 requires a 64-hex SHA256 value')

    root = args.root.expanduser().resolve()
    if not (root / 'Simulator/src/config/config.yaml').is_file():
        raise ValueError('Not a recognized reach_planner root: ' + str(root))

    import rospy
    import message_filters
    from sensor_msgs.msg import CameraInfo, Image
    from nav_msgs.msg import Odometry

    rospy.init_node('f0_forest_pair_capture_v2', anonymous=True, disable_signals=True)
    param = args.simulator_node.rstrip('/') + '/config_file'
    active_config = rospy.get_param(param, None)
    if not isinstance(active_config, str) or not active_config:
        raise ValueError('No ROS private parameter ' + param +
                         '; cannot prove simulator uses intended YAML. STOP.')
    config_path = Path(active_config).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError('ROS-selected YAML does not exist: ' + str(config_path))
    try:
        config_path.relative_to(root / 'query_minco/configs')
    except ValueError:
        raise ValueError('Active YAML outside query_minco/configs: ' + str(config_path))
    before_sha = sha256(config_path)
    if before_sha != expected_sha:
        raise ValueError('Active YAML SHA256 mismatch: expected {}, observed {}'.format(
            expected_sha, before_sha))
    cfg, cam = parse_config(config_path)
    h, w = int(cam['image_height']), int(cam['image_width'])
    if (h, w) != (128, 160):
        raise ValueError('Expected Query-MINCO camera 128x160; got {}x{}'.format(h, w))
    for key, val in [('fx', 100), ('fy', 100), ('cx', 79.5), ('cy', 63.5)]:
        same_float(cam[key], val, 'Active YAML camera.'+key)
    print('Active ROS config: ' + str(config_path), flush=True)
    print('Active config SHA256: ' + before_sha, flush=True)
    print('Waiting for CameraInfo...', flush=True)
    info = rospy.wait_for_message('/camera_info', CameraInfo, timeout=args.timeout)

    samples = []
    lock = threading.Lock()
    completed = threading.Event()
    subscribers = [
        message_filters.Subscriber('/depth_image', Image, queue_size=12),
        message_filters.Subscriber('/sim/odom', Odometry, queue_size=36),
    ]
    sync = message_filters.ApproximateTimeSynchronizer(
        subscribers, queue_size=36, slop=0.04, allow_headerless=False)

    def receive(image, odom):
        if completed.is_set():
            return
        image_stamp = time_s(image.header.stamp)
        odom_stamp = time_s(odom.header.stamp)
        if abs(image_stamp-odom_stamp) > 0.04:
            return
        with lock:
            if not samples or (image_stamp-samples[0][0]) >= args.gap_seconds:
                samples.append((image_stamp, image, odom))
                if len(samples) == 2:
                    completed.set()

    sync.registerCallback(receive)
    print('Waiting for two synchronized depth+odom pairs, gap >= {:.1f}s...'.format(args.gap_seconds), flush=True)
    try:
        if not completed.wait(timeout=args.timeout + args.gap_seconds + 3.0):
            raise ValueError('Timeout waiting for two synchronized depth+odom pairs')
    finally:
        for sub in subscribers:
            sub.unregister()
    with lock:
        two = list(samples[:2])
    t1, im1, od1 = two[0]
    t2, im2, od2 = two[1]
    maxd = validate_contract(cfg, cam, info, im1, h, w)
    validate_contract(cfg, cam, info, im2, h, w)
    d1, d2 = image_to_numpy(im1), image_to_numpy(im2)
    metrics, range_mask, far_mask = calc_stats(d1, d2, maxd)
    o1, o2 = odom_as_dict(od1), odom_as_dict(od2)
    dp = float(np.linalg.norm(np.asarray(o1['position_m']) - np.asarray(o2['position_m'])))
    q1, q2 = normalize_quat(o1['orientation_xyzw']), normalize_quat(o2['orientation_xyzw'])
    da = float(2 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))
    speed1 = float(np.linalg.norm(o1['linear_velocity_mps']))
    speed2 = float(np.linalg.norm(o2['linear_velocity_mps']))

    # Recording is allowed for either outcome; audit status is strictly descriptive.
    stable_pose = dp <= .01 and da <= math.radians(.5) and max(speed1, speed2) <= .05
    stable_depth = (metrics['changed_over_10cm_ratio'] is not None and
                    metrics['changed_over_10cm_ratio'] <= .01 and
                    metrics['range_hit_mask_disagreement_ratio'] <= .01)

    tree_value = Path(str(cfg['tree_file'])).expanduser()
    tree = (tree_value if tree_value.is_absolute() else root / 'Simulator' / tree_value).resolve()
    if not tree.is_file():
        raise ValueError('Referenced tree PLY not found: ' + str(tree))
    if sha256(config_path) != before_sha:
        raise ValueError('Active YAML changed during capture: STOP')
    binary = root / 'Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda'
    other = {
        'tree_ply': tree,
        'simulator_binary_expected_not_proven_live_exe': binary,
        'maps_cpp_current_file': root/'Simulator/src/src/maps.cpp',
        'sensor_cuda_current_file': root/'Simulator/src/src/sensor_simulator.cu',
        'original_simulator_yaml': root/'Simulator/src/config/config.yaml',
    }
    hashes = {k: {'path': str(v), 'sha256': sha256(v)} if v.is_file() else None
              for k, v in other.items()}
    meta = {
        'schema': 'F0_PREFLIGHT_FOREST_PAIRED_OBSERVATION_V2',
        'formal_F0_result': False,
        'map_GT_captured': False,
        'model_inference_or_training': False,
        'map_replay_across_restarts_verified': False,
        'actual_live_binary_identity_verified': False,
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'git_head': git_head(root),
        'simulator_node_private_config_param': param,
        'active_config_path_from_ros': str(config_path),
        'active_config_sha256': before_sha,
        'source_hashes': hashes,
        'forest_config': {k: cfg.get(k) for k in
                          ('random_map', 'maze_type', 'seed', 'tree_file', 'tree_dist', 'resolution',
                           'x_length', 'y_length', 'z_length')},
        'active_camera_config': {k: cam.get(k) for k in
                                 ('fx', 'fy', 'cx', 'cy', 'image_width', 'image_height',
                                  'max_depth_dist', 'normalize_depth', 'pitch')},
        'camera_info': {
            'timestamp_s': time_s(info.header.stamp),
            'frame_id': info.header.frame_id,
            'K': list(info.K),
            'D': list(info.D),
            'width': info.width,
            'height': info.height,
        },
        'samples': [
            {'depth_timestamp_s': t1, 'depth_frame_id': im1.header.frame_id,
             'depth_encoding': im1.encoding, 'depth_odom_delta_s': t1-o1['timestamp_s'],
             'camera_info_age_s_at_depth': t1-time_s(info.header.stamp), 'odometry': o1},
            {'depth_timestamp_s': t2, 'depth_frame_id': im2.header.frame_id,
             'depth_encoding': im2.encoding, 'depth_odom_delta_s': t2-o2['timestamp_s'],
             'camera_info_age_s_at_depth': t2-time_s(info.header.stamp), 'odometry': o2},
        ],
        'temporal_comparison': {
            'time_gap_s': t2-t1,
            'position_delta_m': dp,
            'orientation_delta_deg': math.degrees(da),
            'linear_speed_1_mps': speed1,
            'linear_speed_2_mps': speed2,
            **metrics,
            'stable_pose_preflight': stable_pose,
            'stable_depth_preflight': stable_depth,
            'repeatability_scope': 'Within one running map instance, NOT evidence of reproducibility across restarts',
        },
        'mask_semantics': {
            'range_hit_mask': 'Finite and (0<depth<max_depth). This is a ray hit, NOT verified free-space visibility or a safety certificate.',
            'max_range_mask': 'Finite and depth>=max_depth, not treated as verified free.',
            'network_validity_mask_contract_verified': False,
        },
        'camera_pose_warning': 'Render origin uses odometry position and Rwb*Rbc; static /odom -> /camera_link +0.05 m is visualization-only, not the rendering origin.',
        'next_gates': ['map occupancy GT / collision API', 'between-restart map identity',
                       'check V1 validity-mask semantics', 'frozen checkpoint interface',
                       'live binary/source provenance'],
    }
    dest_base = root / 'query_minco/outputs/forest_f0_v1/preflight_observations_v2'
    dest = dest_base / ('pair_{}_{:09d}'.format(im1.header.stamp.secs, im1.header.stamp.nsecs))
    dest.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(dest/'observation_pair.npz',
                        depth_m_first=d1, depth_m_second=d2,
                        range_hit_mask_first=range_mask.astype(np.uint8),
                        max_range_mask_first=far_mask.astype(np.uint8))
    meta['observation_pair_npz_sha256'] = sha256(dest/'observation_pair.npz')
    (dest/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    print('OK: ' + str(dest), flush=True)
    print('shape={}x{}, fx={}, fy={}, cx={}, cy={}'.format(h,w,cam['fx'],cam['fy'],cam['cx'],cam['cy']), flush=True)
    print('two-frame gap: {:.3f}s, position drift: {:.6f}m, rotation drift: {:.6f}deg'.format(
        t2-t1, dp, math.degrees(da)), flush=True)
    print('range hit={:.2%}; >=20m={:.2%}; >10cm changed pixels={:.3%}'.format(
        metrics['range_hit_ratio_first'], metrics['max_range_ratio_first'],
        metrics['changed_over_10cm_ratio']), flush=True)
    print('PRECHECK pose={}, depth={}'.format('PASS' if stable_pose else 'REVIEW',
                                             'PASS' if stable_depth else 'REVIEW'), flush=True)
    print('No map GT, checkpoint evaluation, online selection, or flight commands performed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.exit('F0 PRECHECK STOP: {}: {}'.format(type(exc).__name__, exc))
