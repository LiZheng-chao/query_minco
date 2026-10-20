#!/usr/bin/env python3
"""F0 geometry preflight: spherical body vs. solid occupancy voxel AABBs.

Only queries a STATIC offline voxel replica. Does not publish ROS topics, infer
models, provide continuous trajectory certificates, or verify the live map.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def query_nearest_voxel(grid, origin, resolution, point, search_radius):
    shape = np.asarray(grid.shape, dtype=np.int64)
    lo = np.maximum(np.floor((point - search_radius - origin) / resolution).astype(np.int64) - 1, 0)
    hi = np.minimum(np.ceil((point + search_radius - origin) / resolution).astype(np.int64) + 2, shape)
    sub = np.asarray(grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    indices = np.argwhere(sub == 1)
    if len(indices) == 0:
        return None
    indices += lo
    voxel_centers = origin + (indices.astype(np.float64) + 0.5) * resolution
    # Exact Euclidean point-to-solid-axis-aligned-box distance, not center distance.
    per_axis = np.maximum(np.abs(voxel_centers - point) - resolution / 2.0, 0.0)
    distances = np.linalg.norm(per_axis, axis=1)
    best = int(np.argmin(distances))
    return {"distance_m": float(distances[best]),
            "voxel_index_xyz": [int(x) for x in indices[best]],
            "voxel_center_world_m": voxel_centers[best].tolist(),
            "voxels_examined_in_window": int(len(indices))}


def main():
    ap = argparse.ArgumentParser(description="F0 starting body occupancy check (offline, no control)")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--voxel", type=Path, help="Override voxel path from metadata")
    ap.add_argument("--center", type=float, nargs=3, default=[0, 0, 2.007999913352256], metavar=("X", "Y", "Z"))
    ap.add_argument("--body-radius", type=float, default=0.12)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--search-radius", type=float, default=1.5)
    ap.add_argument("--expected-sha256", default="2c939ec328f56d7016061ea9ff0eb0a2ef4f458926bedfc2109bd3b128e29d35")
    ap.add_argument("--output", type=Path, help="Write audit JSON, refuse to overwrite existing file")
    args = ap.parse_args()
    if args.body_radius < 0 or args.margin < 0 or args.search_radius <= args.body_radius + args.margin:
        ap.error("Require nonnegative radii and search-radius > body-radius + margin")
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in meta["shape_xyz"])
    origin = np.asarray(meta["origin_m"], dtype=np.float64)
    resolution = float(meta["resolution_m"])
    if len(shape) != 3 or min(shape) <= 0 or not math.isfinite(resolution) or resolution <= 0:
        raise SystemExit("BLOCKED: invalid metadata dimensions or resolution")
    voxel = args.voxel or (args.metadata.parent / meta["voxel_file"])
    if voxel.stat().st_size != math.prod(shape):
        raise SystemExit("BLOCKED: voxel binary size does not equal shape product")
    fingerprint = sha256(voxel)
    if args.expected_sha256 and fingerprint != args.expected_sha256:
        raise SystemExit("BLOCKED: voxel SHA256 differs from expected map")
    grid = np.memmap(voxel, dtype="u1", mode="r", shape=shape, order="C")
    center = np.asarray(args.center, dtype=np.float64)
    if not np.all(np.isfinite(center)):
        raise SystemExit("BLOCKED: nonfinite position")
    total_radius = args.body_radius + args.margin
    world_min = origin.copy()
    world_max = origin + np.asarray(shape, dtype=np.float64) * resolution
    # Conservative physical boundary rule: do not silently use the CUDA renderer's
    # x/y mirrored map indices or its z-above-grid free convention for collision safety.
    inside = bool(np.all(center - total_radius >= world_min) and
                  np.all(center + total_radius <= world_max))
    best = query_nearest_voxel(grid, origin, resolution, center, args.search_radius)
    d = float("inf") if best is None else best["distance_m"]
    if not inside:
        status = "BLOCKED_OUTSIDE_MAP_BOUNDARY"
    elif d <= args.body_radius:
        status = "BODY_COLLISION"
    elif d <= total_radius:
        status = "MARGIN_ONLY_COLLISION"
    else:
        status = "CLEAR_AT_START_ONLY"
    audit = {
        "schema": "F0_START_BODY_CLEARANCE_V1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "geometry_semantics": "occupied 0.1m voxel is a SOLID AABB; sphere-to-box exact distance; static point only",
        "not_a_trajectory_certificate": True,
        "not_a_live_map_identity_proof": True,
        "metadata_path": str(args.metadata.resolve()),
        "voxel_sha256": fingerprint,
        "map_shape_xyz": list(shape),
        "map_origin_world_m": origin.tolist(),
        "resolution_m": resolution,
        "center_world_m": center.tolist(),
        "body_radius_m": args.body_radius,
        "extra_margin_m": args.margin,
        "total_check_radius_m": total_radius,
        "query_window_radius_m": args.search_radius,
        "world_map_bounds_xyz_m": {"min": world_min.tolist(), "max": world_max.tolist()},
        "sphere_fully_inside_stored_grid": inside,
        "closest_occupied_box_in_window": best,
        "nearest_occupied_box_distance_m": None if not math.isfinite(d) else d,
        "clearance_after_body_and_margin_m": None if not math.isfinite(d) else d - total_radius,
        "status": status,
        "caveat": "Clearance is against a conservative solid-voxel occupancy proxy, not exact tree mesh; x/y mirror not extended into physical planning domain; no continuous motion proof."
    }
    print("Map SHA256:", fingerprint)
    print("Center world m:", center.tolist())
    print("Body radius + margin:", args.body_radius, "+", args.margin, "=", total_radius, "m")
    print("Full sphere within stored map bounds:", inside)
    print("Nearest occupied voxel AABB distance (searched <= %.3fm):" % args.search_radius, "none" if best is None else "%.6f m" % d)
    print("Clearance after total check radius:", "at least %.3fm" % (args.search_radius-total_radius) if best is None else "%.6f m" % (d - total_radius))
    print("F0_START_BODY_STATUS=" + status)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("Audit:", args.output)
    if status != "CLEAR_AT_START_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
