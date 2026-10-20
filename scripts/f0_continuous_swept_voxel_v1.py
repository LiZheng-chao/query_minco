#!/usr/bin/env python3
"""F0 offline continuous swept sphere vs. solid voxel AABB *preflight*.

A Bézier curve lies in the convex hull of its control points. Recursive de
Casteljau subdivision plus conservative AABB occupancy queries therefore
certifies collision-free *continuous* swept spheres when every leaf is clear.
A collision is declared only with an actual point-on-curve AABB witness.
Unresolved intervals are UNKNOWN, never silently clear.

This is a certificate against the *offline solid-voxel proxy*, NOT a true tree
mesh certificate, online sensor observability, or closed-loop flight safety.
No ROS, no controller commands, no training, no writes to source code.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MAP_SHA = '2c939ec328f56d7016061ea9ff0eb0a2ef4f458926bedfc2109bd3b128e29d35'
EPS_BROAD_M = 1e-8  # outward numeric tolerance for safe broad phase
EPS_WITNESS_M = 1e-9  # require a strictly interior witness, no boundary roundoff claim


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class VoxelWorld:
    def __init__(self, voxels, origin, resolution, map_sha=None):
        self.grid = voxels
        self.origin = np.asarray(origin, dtype=np.float64)
        self.res = float(resolution)
        self.shape = np.asarray(voxels.shape, dtype=np.int64)
        assert self.origin.shape == (3,) and self.shape.shape == (3,)
        assert np.all(self.shape > 0) and math.isfinite(self.res) and self.res > 0
        self.lo = self.origin.copy()
        self.hi = self.origin + self.shape * self.res
        self.hash = map_sha
        self.queries = 0
        self.visited_voxels = 0

    def inside_envelope(self, box_lo, box_hi, radius):
        return bool(np.all(box_lo - radius - EPS_BROAD_M >= self.lo) and
                    np.all(box_hi + radius + EPS_BROAD_M <= self.hi))

    def occupancy_near_bbox(self, box_lo, box_hi, radius):
        """Conservative: True iff MAY intersect box swept with a radius-r sphere."""
        r = radius + EPS_BROAD_M
        imin = np.floor((box_lo - r - self.origin) / self.res).astype(np.int64) - 1
        imax = np.ceil((box_hi + r - self.origin) / self.res).astype(np.int64) + 2
        imin = np.maximum(imin, 0)
        imax = np.minimum(imax, self.shape)
        self.queries += 1
        if np.any(imin >= imax):
            return False
        self.visited_voxels += int(np.prod(imax - imin))
        occupied_local = np.argwhere(self.grid[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]] != 0)
        if occupied_local.size == 0:
            return False
        ii = occupied_local + imin
        v_lo = self.origin + ii * self.res
        v_hi = v_lo + self.res
        # Exact AABB-to-AABB Euclidean lower bound, rather than just overlap of
        # per-axis expanded index windows (which gives many false UNKNOWNs).
        axis_gap = np.maximum(np.maximum(v_lo - box_hi, box_lo - v_hi), 0.0)
        dist2 = np.einsum('ij,ij->i', axis_gap, axis_gap)
        return bool(np.any(dist2 <= r*r))

    def point_witness(self, xyz, r):
        """Exact point-to-solid-AABB distances to all nearby occupied cells."""
        p = np.asarray(xyz, dtype=np.float64)
        if np.any(p - r < self.lo) or np.any(p + r > self.hi):
            return None
        rr = r + self.res + 1e-7
        i0 = np.maximum(np.floor((p - rr - self.origin) / self.res).astype(np.int64), 0)
        i1 = np.minimum(np.ceil((p + rr - self.origin) / self.res).astype(np.int64) + 1, self.shape)
        sub = np.asarray(self.grid[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]])
        ii = np.argwhere(sub != 0)
        if ii.size == 0:
            return None
        ii += i0
        corners_lo = self.origin + ii * self.res
        corners_hi = corners_lo + self.res
        per_axis = np.maximum(np.maximum(corners_lo - p, p - corners_hi), 0)
        dsq = np.einsum('ij,ij->i', per_axis, per_axis)
        j = int(np.argmin(dsq))
        d = float(math.sqrt(float(dsq[j])))
        if d < r - EPS_WITNESS_M:
            return {'point_world_m': p.tolist(), 'voxel_index_xyz': ii[j].tolist(),
                    'point_to_voxel_aabb_m': d}
        return None


def split_bezier(cp):
    """de Casteljau at 1/2; each half describes the identical original curve."""
    rows = [cp]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1] + rows[-1][1:]) / 2.0)
    left = np.stack([row[0] for row in rows])
    right = np.stack([row[-1] for row in rows[::-1]])
    return left, right


def bezier_midpoint(cp):
    q = cp.copy()
    while len(q) > 1:
        q = (q[:-1] + q[1:]) * 0.5
    return q[0]


def check_radius(world, segments, radius, max_depth, max_nodes):
    unknown = set()
    nodes = 0
    for segment_id, cp in enumerate(segments):
        # stack: (control points, depth, param_start, param_end)
        stack = [(cp, 0, 0., 1.)]
        # Boundary endpoints are unresolved; cannot issue a clearance certificate.
        # First preserve endpoint collision witnesses, if any.
        for p, param in ((cp[0], 0.), (cp[-1], 1.)):
            witness = world.point_witness(p, radius)
            if witness is not None:
                witness.update({'segment_id': segment_id, 'segment_parameter_s': param})
                return {'status': 'WITNESS_COLLISION', 'witness': witness, 'nodes': nodes}
            if not world.inside_envelope(p, p, radius):
                return {'status':'UNKNOWN','reasons':['MAP_BOUNDARY_ENDPOINT'],
                        'nodes':nodes,'segment_id':segment_id}
        while stack:
            cur, depth, t0, t1 = stack.pop()
            if nodes >= max_nodes:
                unknown.add('MAX_NODE_BUDGET')
                return {'status': 'UNKNOWN', 'reasons': sorted(unknown), 'nodes': nodes,
                        'segment_id': segment_id}
            nodes += 1
            lo, hi = cur.min(axis=0), cur.max(axis=0)
            # An entire hull envelope may protrude while curve remains inside;
            # if so, subdivide. Only definitely clear after envelope is inside.
            within = world.inside_envelope(lo, hi, radius)
            possible_occupied = world.occupancy_near_bbox(lo, hi, radius)
            if within and not possible_occupied:
                continue  # CERTIFIED SAFE over the full continuous subcurve
            for point, param in ((cur[0], t0), (bezier_midpoint(cur), (t0+t1)/2), (cur[-1], t1)):
                witness = world.point_witness(point, radius)
                if witness is not None:
                    witness.update({'segment_id': segment_id, 'segment_parameter_s': param})
                    return {'status': 'WITNESS_COLLISION', 'witness': witness, 'nodes': nodes}
            if depth >= max_depth:
                unknown.add('MAP_BOUNDARY' if not within else 'MAX_SUBDIVISION_DEPTH')
                continue
            l, r = split_bezier(cur)
            tm = (t0 + t1) / 2.0
            stack.append((r, depth+1, tm, t1))
            stack.append((l, depth+1, t0, tm))
    if unknown:
        return {'status': 'UNKNOWN', 'reasons': sorted(unknown), 'nodes': nodes}
    return {'status': 'CERTIFIED_CLEAR', 'nodes': nodes}


def classify(world, segments, body_radius=0.12, margin=0.03, max_depth=16, max_nodes=50000):
    if body_radius <= 0 or margin < 0 or max_depth < 1 or max_nodes < 1:
        raise ValueError('Invalid radius/subdivision budget')
    world.queries = 0
    world.visited_voxels = 0
    body = check_radius(world, segments, body_radius, max_depth, max_nodes)
    if body['status'] == 'WITNESS_COLLISION':
        overall, checked = 'BODY_COLLISION', {'body': body}
    elif body['status'] == 'UNKNOWN':
        overall, checked = 'UNKNOWN', {'body': body}
    else:
        margin_check = check_radius(world, segments, body_radius+margin, max_depth, max_nodes)
        checked = {'body': body, 'body_plus_margin': margin_check}
        if margin_check['status'] == 'WITNESS_COLLISION':
            overall = 'MARGIN_ONLY'
        elif margin_check['status'] == 'CERTIFIED_CLEAR':
            overall = 'CLEAR_CERTIFIED'
        else:
            overall = 'UNKNOWN'
    return {'status': overall, 'checks': checked,
            'aabb_occupancy_queries': world.queries,
            'voxel_cells_scanned_upper_bound': world.visited_voxels}


def from_json_segments(data):
    if not isinstance(data, dict) or not isinstance(data.get('segments'), list) or not data['segments']:
        raise ValueError('Expected JSON object with nonempty segments list')
    segments = []
    for n, item in enumerate(data['segments']):
        tp = item.get('type')
        if tp == 'line':
            cp = np.asarray([item['start_world_m'], item['end_world_m']], dtype=np.float64)
        elif tp == 'bezier':
            cp = np.asarray(item['control_points_world_m'], dtype=np.float64)
        else:
            raise ValueError('Unsupported segment %d type %r' % (n, tp))
        if cp.ndim != 2 or cp.shape[1] != 3 or not 2 <= len(cp) <= 12 or not np.isfinite(cp).all():
            raise ValueError('Bad 3D control points at segment %d' % n)
        if len(segments) and not np.allclose(segments[-1][-1], cp[0], rtol=0, atol=1e-6):
            raise ValueError('Segment %d does not connect continuously to previous' % n)
        segments.append(cp)
    return segments


def load_world(metadata, expected_sha):
    meta = json.loads(metadata.read_text(encoding='utf-8'))
    dims = tuple(int(v) for v in meta['shape_xyz'])
    voxel_path = metadata.parent / meta['voxel_file']
    if voxel_path.stat().st_size != math.prod(dims):
        raise ValueError('Binary length does not match metadata shape')
    actual_sha = digest(voxel_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError('Map SHA256 mismatch: expected %s, actual %s' % (expected_sha, actual_sha))
    grid = np.memmap(voxel_path, dtype=np.uint8, mode='r', shape=dims, order='C')
    if np.any(np.asarray(meta['origin_m'],dtype=float)>1e6):
        raise ValueError('Invalid metadata origin')
    return VoxelWorld(grid, meta['origin_m'], meta['resolution_m'], actual_sha)


def synthetic_tests():
    grid = np.zeros((60, 60, 60), dtype='u1')
    origin = np.array([-1.5, -1.5, -1.5])
    res = 0.05
    # A solid obstacle centered on x=0.825, y=z=0.025 (voxel x=[.8,.85]).
    grid[46,30,30] = 1
    w = VoxelWorld(grid, origin, res)
    p = [0., 0., 0.]
    def test(label, segments, want):
        got = classify(w, segments, max_depth=18, max_nodes=50000)
        print('%s: %s (nodes=%s)' % (label, got['status'], got['checks']['body']['nodes']))
        if got['status'] != want:
            raise AssertionError('%s: wanted %s got %s' % (label, want, got))
    test('CLEAR_LINE', [np.array([p,[.20,0,0]])], 'CLEAR_CERTIFIED')
    test('BODY_COLLISION_LINE', [np.array([p,[.82,.02,.02]])], 'BODY_COLLISION')
    # At endpoint x=.667, obstacle begins at x=.8; distance=.133 in (0.12,0.15).
    test('MARGIN_ONLY_LINE', [np.array([p,[.667,.02,.02]])], 'MARGIN_ONLY')
    test('BOUNDARY_UNKNOWN', [np.array([[0,-.5,0],[1.45,-.5,0]])], 'UNKNOWN')
    test('CLEAR_BEZIER', [np.array([[0,0,0],[.05,.02,0],[.12,.02,0],[.2,0,0]])], 'CLEAR_CERTIFIED')
    test('BODY_COLLISION_BEZIER', [np.array([[0,0,0],[1.35,0,0],[1.35,0,0],[0,0,0]])], 'BODY_COLLISION')
    # If controls include an obstacle but curve stays far, conservative subdivision
    # must eventually prove it safe or return UNKNOWN, never claim body collision.
    bow = [np.array([[0.,-.3,0],[1.,-.3,0],[1.,-.3,0],[0.,-.3,0]])]
    test('CURVED_BEZIER_CLEAR', bow, 'CLEAR_CERTIFIED')
    print('SYNTHETIC_TESTS=PASS')


def demo_forest(world, original_audit):
    audit = json.loads(original_audit.read_text(encoding='utf-8'))
    assert audit['voxel_sha256'] == world.hash, 'Start audit map does not match'
    p = np.asarray(audit['center_world_m'], dtype=float)
    near = np.asarray(audit['closest_occupied_box_in_window']['voxel_center_world_m'], dtype=float)
    cases = {
        'short_clear_line': {'segments': [{'type':'line', 'start_world_m':p.tolist(),
                                           'end_world_m':(p+[0.05,0,0]).tolist()}]},
        'known_collision_line': {'segments': [{'type':'line', 'start_world_m':p.tolist(),
                                               'end_world_m':near.tolist()}]},
        'short_clear_quintic_bezier': {'segments': [{'type':'bezier',
            'control_points_world_m':[(p + k*np.array([0.01,0,0])).tolist() for k in range(6)]}]},
        'out_of_map_line': {'segments': [{'type':'line', 'start_world_m':p.tolist(),
                                         'end_world_m':(world.hi+[1,0,0]).tolist()}]},
    }
    for name, data in cases.items():
        result = classify(world, from_json_segments(data))
        print('%s: %s' % (name, result['status']))
        data['result'] = result
    assert cases['short_clear_line']['result']['status'] == 'CLEAR_CERTIFIED'
    assert cases['known_collision_line']['result']['status'] == 'BODY_COLLISION'
    assert cases['short_clear_quintic_bezier']['result']['status'] == 'CLEAR_CERTIFIED'
    assert cases['out_of_map_line']['result']['status'] == 'UNKNOWN'
    print('FOREST_NEGATIVE_POSITIVE_CONTROLS=PASS')
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--metadata', type=Path)
    ap.add_argument('--trajectory-json', type=Path, help='World XYZ lines / Bézier control points')
    ap.add_argument('--start-audit', type=Path, help='Run the four known forest controls')
    ap.add_argument('--expected-map-sha', default=MAP_SHA)
    ap.add_argument('--body-radius', type=float, default=.12)
    ap.add_argument('--margin', type=float, default=.03)
    ap.add_argument('--max-depth', type=int, default=16)
    ap.add_argument('--max-nodes', type=int, default=50000)
    ap.add_argument('--output', type=Path, help='Write new JSON audit; refuse overwrite')
    args = ap.parse_args()
    if args.self_test:
        synthetic_tests()
        if not args.metadata:
            return
    if not args.metadata:
        ap.error('--metadata is required unless only --self-test is used')
    world = load_world(args.metadata, args.expected_map_sha)
    result = {'schema':'F0_CONTINUOUS_SWEPT_VOXEL_PREFLIGHT_V1',
              'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'is_formal_f0_result':False, 'offline_voxel_proxy_only':True,
              'not_original_tree_mesh_certificate':True,
              'map_sha256':world.hash,'map_metadata_path':str(args.metadata.resolve()),
              'body_radius_m':args.body_radius,'extra_margin_m':args.margin,
              'classification_definitions':{
                  'BODY_COLLISION':'An actual curve point overlaps a solid voxel AABB within body radius',
                  'MARGIN_ONLY':'Whole curve certified free at body radius but has a witness inside margin radius',
                  'CLEAR_CERTIFIED':'Entire continuous Bézier/line curve certified free of voxels within total radius',
                  'UNKNOWN':'Subdivision, budget, or physical map boundary prevented proof; NOT safe'},
              'limits':'No map live-executable proof, no sensors/FOV, no dynamical constraints, no network inference.'}
    if args.start_audit:
        result['controls'] = demo_forest(world, args.start_audit)
    if args.trajectory_json:
        payload = json.loads(args.trajectory_json.read_text(encoding='utf-8'))
        cps = from_json_segments(payload)
        result['trajectory_input_path'] = str(args.trajectory_json.resolve())
        result['trajectory_input_sha256'] = digest(args.trajectory_json)
        result['trajectory_segment_count'] = len(cps)
        result['trajectory'] = classify(world,cps,args.body_radius,args.margin,args.max_depth,args.max_nodes)
        print('F0_TRAJECTORY_STATUS=' + result['trajectory']['status'])
    if not args.start_audit and not args.trajectory_json:
        ap.error('Use --start-audit and/or --trajectory-json')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
            f.write('\n')
        print('AUDIT='+str(args.output))


if __name__=='__main__':
    main()
