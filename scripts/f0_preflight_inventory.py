#!/usr/bin/env python3
"""F0-Preflight: read-only local YOPO-MINCO / Query-MINCO inventory.

Does not import project code, run a simulator, start ROS, load a checkpoint,
change model weights, or modify baseline project files. Writes one new JSON
report under query_minco/outputs/forest_f0_v1/. Python 3.8+ stdlib only.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

SKIP_DIRS = {'.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules',
             'build', 'devel', 'install', '.cache', '.idea', '.pytest_cache'}
SCANNED_EXTENSIONS = {'.py', '.cc', '.cpp', '.c', '.h', '.hpp', '.yaml', '.yml',
                      '.xml', '.launch', '.world', '.sdf', '.urdf', '.json',
                      '.md', '.txt', '.cfg', '.ini', '.sh', '.bag', '.db3', '.npz',
                      '.npy', '.pcd', '.ply', '.obj', '.stl', '.mesh'}
CHECKPOINT_EXTENSIONS = {'.pt', '.pth', '.ckpt', '.safetensors', '.pkl'}
CATEGORIES = {
    'forest_map_simulator': re.compile(r'forest|tree|wood|obstacle|map_generator|random_map|environment|world|cylinder', re.I),
    'depth_camera_sensor': re.compile(r'depth|camera|sensor|pointcloud|point_cloud|render|raycast|fisheye|stereo', re.I),
    'trajectory_and_evaluation': re.compile(r'minco|query|minco_adapter|traj|planner|collision|clearance|feasib|verif|teacher|selector|observab', re.I),
    'launch_and_configuration': re.compile(r'launch|\.yaml$|\.yml$|\.world$|\.sdf$|\.urdf$', re.I),
    'B1_and_F0_evidence': re.compile(r'(^|[/_.-])(b1|f0|step5760|a42|a43|a44|b42)([/_.-]|$)|p2c2c1b1', re.I),
}
MAX_FILES = 100000
MAX_MATCHES_PER_CATEGORY = 180
MAX_CHECKPOINTS = 160
MAX_HASHED_CODE = 80
MAX_HASHED_CHECKPOINTS = 20


def command(root, args, timeout=8):
    try:
        proc = subprocess.run(args, cwd=str(root), capture_output=True,
                              text=True, errors='replace', timeout=timeout,
                              check=False)
        return {'exit_code': proc.returncode,
                'stdout': proc.stdout[:12000].strip(),
                'stderr': proc.stderr[:4000].strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'),
                        help='reach_planner project root (default: current directory)')
    parser.add_argument('--output', type=Path, default=None,
                        help='report path; must be inside query_minco/outputs/forest_f0_v1')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir() or not (root / 'query_minco').is_dir():
        parser.error('Run from reach_planner root (must contain query_minco/).')
    allowed = (root / 'query_minco/outputs/forest_f0_v1').resolve()
    out = (args.output if args.output else
           Path('query_minco/outputs/forest_f0_v1/F0_LOCAL_INVENTORY.json'))
    out = (out if out.is_absolute() else root / out).resolve()
    if out == allowed or allowed not in out.parents:
        parser.error('Output must be inside query_minco/outputs/forest_f0_v1/.')
    if out.exists():
        parser.error('Output already exists. Use a new report file name; will not overwrite.')

    try:
        torch_version = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        torch_version = None

    report = {
        'tool': 'F0_PREFLIGHT_READ_ONLY_INVENTORY_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'baseline_writes': False,
        'model_forward_or_training_performed': False,
        'ros_or_simulator_started': False,
        'system': {
            'python': sys.version.splitlines()[0],
            'platform': platform.platform(),
            'torch_distribution_version': torch_version,
            'ROS_DISTRO': os.environ.get('ROS_DISTRO', 'not_set'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set'),
        },
        'git_root': {
            'top_level': command(root, ['git', 'rev-parse', '--show-toplevel']),
            'head': command(root, ['git', 'rev-parse', 'HEAD']),
            'branch': command(root, ['git', 'branch', '--show-current']),
            'status': command(root, ['git', 'status', '--short', '-uno']),
        },
        'top_level_directories': sorted(p.name for p in root.iterdir() if p.is_dir()),
        'candidate_files': {key: [] for key in CATEGORIES},
        'checkpoint_files': [],
        'scanned_file_count': 0,
        'scan_truncated': False,
        'file_sha256_selected': [],
        'warnings': [],
        'next_checks': [
            'Verify reproducible forest map seed and source GT geometry',
            'Verify depth definition, units, validity mask, camera intrinsics/extrinsics',
            'Verify B0 FLU state and optical-frame coordinate transform',
            'Verify all four B1 step5760 checkpoint formats and original evaluation contract',
            'Only after those checks run a deterministic 1-map static smoke test',
        ],
    }

    source_files = []
    checkpoints = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for filename in sorted(files):
            report['scanned_file_count'] += 1
            if report['scanned_file_count'] > MAX_FILES:
                report['scan_truncated'] = True
                break
            path = Path(base) / filename
            rel = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            if suffix in CHECKPOINT_EXTENSIONS:
                try:
                    st = path.stat()
                    checkpoints.append({'path': rel, 'size_bytes': st.st_size})
                except OSError:
                    pass
            if suffix not in SCANNED_EXTENSIONS:
                continue
            for category, matcher in CATEGORIES.items():
                if matcher.search(rel) and len(report['candidate_files'][category]) < MAX_MATCHES_PER_CATEGORY:
                    report['candidate_files'][category].append(rel)
                    if category in ('forest_map_simulator', 'depth_camera_sensor', 'trajectory_and_evaluation'):
                        source_files.append(path)
        if report['scan_truncated']:
            break
    report['checkpoint_files'] = checkpoints[:MAX_CHECKPOINTS]
    report['checkpoint_count_total'] = len(checkpoints)
    report['checkpoint_display_truncated'] = len(checkpoints) > MAX_CHECKPOINTS

    # Fingerprint a small, explicit set of likely entrypoints/configs, not the full baseline.
    ordered = sorted(set(source_files), key=lambda p: (
        0 if CATEGORIES['forest_map_simulator'].search(p.name) else 1,
        0 if CATEGORIES['depth_camera_sensor'].search(p.name) else 1,
        str(p)))
    for path in ordered[:MAX_HASHED_CODE]:
        try:
            st = path.stat()
            if 0 < st.st_size <= 8 * 1024 * 1024:
                report['file_sha256_selected'].append({
                    'path': path.relative_to(root).as_posix(),
                    'size_bytes': st.st_size,
                    'sha256': file_sha256(path)})
        except OSError as exc:
            report['warnings'].append('Cannot hash ' + str(path) + ': ' + str(exc))

    # Hash up to 20 checkpoint candidates likely to contain final B1 results.
    likely = [item for item in checkpoints if
              re.search(r'5760|(^|[/_.-])(b1|a42|a43|a44|b42)([/_.-]|$)|p2c2c1b1',
                        item['path'], re.I)]
    for item in sorted(likely, key=lambda x: x['path'])[:MAX_HASHED_CHECKPOINTS]:
        try:
            if item['size_bytes'] <= 512 * 1024 * 1024:
                item['sha256'] = file_sha256(root / item['path'])
            else:
                item['sha256'] = None
                item['sha256_status'] = 'skipped_over_512MiB'
        except OSError as exc:
            item['sha256_error'] = str(exc)
    if not report['candidate_files']['forest_map_simulator']:
        report['warnings'].append('No forest-looking FILE NAME found; search contents or ROS dependencies next.')
    if not checkpoints:
        report['warnings'].append('No checkpoint files found beneath project root; check external storage.')
    if not likely:
        report['warnings'].append('No checkpoint filename looked like B1/step5760. Check naming and off-tree storage.')

    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never overwrite existing runs.
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print('F0 read-only local inventory completed (no ROS, inference or training).')
    print('JSON report:', out)
    print('Files scanned:', report['scanned_file_count'])
    print('Checkpoint files:', len(checkpoints))
    for name, files in report['candidate_files'].items():
        print(f'  {name}: {len(files)} candidate paths recorded')
    for warning in report['warnings']:
        print('WARNING:', warning)
    print('Next: send F0_LOCAL_INVENTORY.json for review; no F0 metric is claimed.')


if __name__ == '__main__':
    main()
