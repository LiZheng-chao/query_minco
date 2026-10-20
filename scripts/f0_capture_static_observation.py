#!/usr/bin/env python3
"""Capture one ROS-synchronized YOPO forest depth+odometry PRE-FLIGHT observation.

No map subscription, no flight commands, no network inference or training.
Outputs an observation record, NOT a ground-truth collision map or a formal F0 result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone

import numpy as np
import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def config_scalar(text, name):
    match = re.search(r'^\s*' + re.escape(name) + r':\s*([^#\r\n]+)', text, re.MULTILINE)
    return match.group(1).strip().strip('"\'') if match else None


def stamp_as_float(stamp):
    return stamp.secs + stamp.nsecs * 1e-9


def xyz(msg):
    return [float(msg.x), float(msg.y), float(msg.z)]


def xyzw(msg):
    return [float(msg.x), float(msg.y), float(msg.z), float(msg.w)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('~/reach_planner').expanduser())
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config = root / 'Simulator/src/config/config.yaml'
    binary = root / 'Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda'
    for p in (config, binary):
        if not p.is_file():
            raise SystemExit('Required local file not found: ' + str(p))
    config_text = config.read_text(encoding='utf-8')

    rospy.init_node('f0_static_observation_capture', anonymous=True, disable_signals=True)
    print('Waiting for camera_info...', flush=True)
    info = rospy.wait_for_message('/camera_info', CameraInfo, timeout=args.timeout)
    paired = {}
    event = threading.Event()
    subs = [
        message_filters.Subscriber('/depth_image', Image, queue_size=8),
        message_filters.Subscriber('/sim/odom', Odometry, queue_size=25),
    ]
    sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=30, slop=0.06)

    def on_pair(depth_msg, odom_msg):
        if event.is_set():
            return
        paired['depth'] = depth_msg
        paired['odom'] = odom_msg
        event.set()

    sync.registerCallback(on_pair)
    print('Waiting for synchronized depth + odometry...', flush=True)
    if not event.wait(args.timeout):
        raise SystemExit('ERROR: No depth+odom pair received within timeout; check topic timestamps.')
    for s in subs:
        s.unregister()

    image = paired['depth']
    odom = paired['odom']
    if image.encoding != '32FC1':
        raise SystemExit('Unexpected depth encoding: ' + image.encoding)
    if image.step < image.width * 4:
        raise SystemExit('Malformed depth image step.')
    if len(image.data) < image.step * image.height:
        raise SystemExit('Malformed depth image buffer.')
    dtype = np.dtype('>f4' if image.is_bigendian else '<f4')
    depth_m = np.ndarray((image.height, image.width), dtype=dtype,
                         buffer=image.data, strides=(image.step, 4)).copy().astype(np.float32)
    max_str = config_scalar(config_text, 'max_depth_dist')
    max_range = float(max_str) if max_str else 20.0
    finite = np.isfinite(depth_m)
    range_hit_mask = finite & (depth_m > 0) & (depth_m < max_range)
    max_range_mask = finite & (depth_m >= max_range)
    delta = stamp_as_float(image.header.stamp) - stamp_as_float(odom.header.stamp)
    if abs(delta) > 0.061:
        raise SystemExit('Unexpected depth/odom timestamp gap: {:.4f}s'.format(delta))

    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=False)
    snapshot_id = 'obs_{}_{:09d}'.format(image.header.stamp.secs, image.header.stamp.nsecs)
    output_dir = root / 'query_minco/outputs/forest_f0_v1/preflight_observations' / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / 'observation.npz',
                        depth_m=depth_m,
                        range_hit_mask=range_hit_mask.astype(np.uint8),
                        max_range_mask=max_range_mask.astype(np.uint8))
    metadata = {
        'schema': 'F0_PREFLIGHT_FOREST_OBSERVATION_V1',
        'formal_F0_result': False,
        'ground_truth_map_captured': False,
        'checkpoint_or_query_minco_evaluated': False,
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'git_head': head.stdout.strip() if head.returncode == 0 else None,
        'simulator_config_path': str(config),
        'simulator_config_sha256': sha256_file(config),
        'simulator_binary_path': str(binary),
        'simulator_binary_sha256': sha256_file(binary),
        'map_parameters_from_config': {
            name: config_scalar(config_text, name)
            for name in ('random_map', 'seed', 'maze_type', 'tree_file', 'tree_dist',
                         'resolution', 'x_length', 'y_length', 'z_length')
        },
        'depth': {
            'topic': '/depth_image',
            'timestamp': stamp_as_float(image.header.stamp),
            'frame_id': image.header.frame_id,
            'encoding': image.encoding,
            'width': image.width,
            'height': image.height,
            'step_bytes': image.step,
            'min_finite_m': float(depth_m[finite].min()) if finite.any() else None,
            'max_finite_m': float(depth_m[finite].max()) if finite.any() else None,
            'finite_ratio': float(finite.mean()),
            'range_hit_ratio': float(range_hit_mask.mean()),
            'max_range_ratio': float(max_range_mask.mean()),
            'range_hit_depth_percentiles_m': np.percentile(depth_m[range_hit_mask], [5, 50, 95]).tolist()
                                           if range_hit_mask.any() else None,
            'range_hit_mask_semantics': '0 < finite depth < configured max depth; NOT an occlusion or safety certificate',
        },
        'camera_info': {
            'topic': '/camera_info',
            'timestamp': stamp_as_float(info.header.stamp),
            'frame_id': info.header.frame_id,
            'width': info.width,
            'height': info.height,
            'K_row_major': list(info.K),
            'D': list(info.D),
            'distortion_model': info.distortion_model,
        },
        'odometry': {
            'topic': '/sim/odom',
            'timestamp': stamp_as_float(odom.header.stamp),
            'frame_id': odom.header.frame_id,
            'child_frame_id': odom.child_frame_id,
            'position_m': xyz(odom.pose.pose.position),
            'orientation_xyzw': xyzw(odom.pose.pose.orientation),
            'linear_velocity': xyz(odom.twist.twist.linear),
            'angular_velocity': xyz(odom.twist.twist.angular),
        },
        'depth_minus_odom_time_s': delta,
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('OK: ' + str(output_dir))
    print('frame={frame}, size={height}x{width}, range_hit={ratio:.2%}, max_range={maxratio:.2%}'.format(
        frame=image.header.frame_id, height=image.height, width=image.width,
        ratio=float(range_hit_mask.mean()), maxratio=float(max_range_mask.mean())))
    print('depth-minus-odom timestamp delta: {:.4f}s'.format(delta))
    print('odom position:', metadata['odometry']['position_m'])
    print('Files: observation.npz (small), metadata.json')
    print('NOTE: map GT, camera optical transform, and model input contract NOT yet validated.')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException as exc:
        sys.exit('ROS error: ' + str(exc))
