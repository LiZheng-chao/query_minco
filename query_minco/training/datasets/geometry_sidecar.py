"""Versioned, read-only train geometry binding. Reference FrozenDataset is unchanged."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ..io import artifact, exact, require, digest, read
from .order import FrozenOrder


def geometry_digest(scene):
    return digest(dict(obstacles=scene['obstacles'], domain=scene['flight_center_domain_m']))


def validate_semantics(row):
    require(row['split'] == 'train', 'GEOMETRY_NONTRAIN')
    require(row['frame'] == 'fixed B0 FLU' and row['units'] == 'm_s', 'GEOMETRY_FRAME_UNITS')
    require(row['domain_semantics'] == 'center_bounds_not_occupied_solids', 'DOMAIN_SEMANTICS')
    require(type(row['empty_certified']) is bool, 'EMPTY_CERTIFICATE')
    require(row['body_radius_m'] == .12 and row['safety_margin_m'] == .03, 'BODY_CONTRACT')
    domain = np.asarray(row['flight_center_domain_m'])
    require(domain.dtype.kind in 'fi' and domain.shape == (2, 3) and
            np.isfinite(domain).all() and (domain[0] < domain[1]).all() and
            (domain[0] <= 0).all() and (domain[1] >= 0).all(), 'DOMAIN')
    require(isinstance(row['obstacles'], list), 'BOX_LIST')
    if not row['obstacles']:
        require(row['empty_certified'], 'UNAUTHENTICATED_EMPTY')
    else:
        require(not row['empty_certified'], 'EMPTY_CONTRADICTION')
        require(len({b['name'] for b in row['obstacles']}) == len(row['obstacles']), 'BOX_NAMES')
        for box in row['obstacles']:
            exact(box, ['name', 'min', 'max'], 'AABB')
            require(len(box['min']) == len(box['max']) == 3, 'AABB_SHAPE')
            a = np.asarray([box['min'], box['max']])
            require(a.dtype.kind in 'fi' and a.shape == (2, 3) and
                    np.isfinite(a).all() and (a[0] < a[1]).all(), 'AABB_EXTENTS')


class GeometrySidecar:
    """Must be constructed against the already qualified frozen train dataset."""
    def __init__(self, binding, dataset):
        self.path = artifact(binding)
        self.binding = copy.deepcopy(binding)
        self.dataset = dataset
        self.manifest = read(self.path)
        m = self.manifest
        exact(m, ['schema', 'qualification', 'dataset', 'orders', 'provenance', 'rows'], 'geometry_sidecar')
        require(m['schema'] == 'rsps_geometry_sidecar_v1' and
                m['qualification'] == 'TRAIN_GEOMETRY_ONLY_NOT_TEACHER_REGENERATION', 'SIDECAR_SCHEMA')
        require(artifact(m['dataset']) == dataset.path and
                dataset.manifest['qualification'] == 'TRAIN', 'SIDECAR_DATASET')
        require(m['provenance']['status'] == 'still_open' and
                m['provenance']['blocks_geometry_qualification'] is False, 'PROVENANCE_DECISION')
        require(len(m['rows']) == len(dataset.rows) == 384, 'SIDECAR_ROW_COUNT')
        self.rows = {}
        self._json_cache = {}
        for row in m['rows']:
            require(row['sample_id'] not in self.rows, 'SIDECAR_DUPLICATE_ID')
            self._validate_row(row)
            self.rows[row['sample_id']] = copy.deepcopy(row)
        require(set(self.rows) == set(dataset.rows), 'SIDECAR_ID_BIJECTION')
        require({x['seed'] for x in m['orders']} == {42, 43, 44} and len(m['orders']) == 3, 'THREE_SEEDS')
        self.orders = {}
        for order in m['orders']:
            require(order['artifact']['path'].endswith('C1_orthogonal_seed%d.npz' % order['seed']), 'SEED_ORDER')
            self.orders[order['seed']] = FrozenOrder(order['artifact'], dataset, 'b2c1_reference_v1')
        self._json_cache.clear()

    def _read(self, binding):
        p = artifact(binding)  # Rehash even when parsed JSON is cached.
        key = str(p)
        if key not in self._json_cache:
            self._json_cache[key] = read(p)
        return self._json_cache[key]

    def _validate_row(self, row):
        exact(row, ['sample_id', 'split', 'scene_id', 'family', 'sample', 'certificates',
                    'scene_source', 'scene_pointer', 'anchor', 'contract_source', 'frame',
                    'units', 'body_radius_m', 'safety_margin_m', 'obstacles',
                    'flight_center_domain_m', 'domain_semantics', 'empty_certified'], 'geometry_row')
        validate_semantics(row)
        sid = row['sample_id']
        require(sid in self.dataset.rows, 'GEOMETRY_UNKNOWN_TRAIN_ID')
        dr = self.dataset.rows[sid]
        require(dr['split'] == 'train' and dr['artifact'] == row['sample'] and
                dr['certificates'] == row['certificates'], 'SAMPLE_BINDING')
        self.dataset.check_artifact(row['sample'])
        for cert in row['certificates']:
            self.dataset.check_artifact(cert)
        source = self._read(row['scene_source'])
        parts = row['scene_pointer'].strip('/').split('/')
        scene = source
        try:
            for part in parts:
                scene = scene[int(part)] if isinstance(scene, list) else scene[part]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError('INVALID_SCENE_POINTER') from error
        require(scene['obstacles'] == row['obstacles'] and
                scene['flight_center_domain_m'] == row['flight_center_domain_m'] and
                scene['frame'] == row['frame'] and
                geometry_digest(scene) == row['scene_id'] and
                scene.get('family_id') == row['family'], 'SOURCE_SCENE_MISMATCH')
        anchor = self._read(row['anchor']['artifact'])
        if row['anchor']['kind'] == 'metadata':
            candidates = [x for x in anchor['rows'] if x['id'] == sid and x['split'] == 'train']
            require(len(candidates) == 1 and
                    candidates[0]['metadata_sha256'] == row['scene_source']['sha256'] and
                    candidates[0]['npz_sha256'] == row['sample']['sha256'], 'METADATA_ANCHOR')
            require(source['id'] == sid and source['split'] == 'train', 'SOURCE_NONTRAIN')
        elif row['anchor']['kind'] == 'task_plan':
            require(len(parts) == 2 and parts[1] == 'scene', 'PLAN_POINTER')
            entry = source[int(parts[0])]
            require(entry['id'] == sid and entry['block'] == 'train_orth' and
                    entry['split'] == 'train_supplement' and
                    entry['geometry_sha256'] == row['scene_id'], 'PLAN_NONTRAIN')
            ar = [x for x in anchor['rows'] if x['id'] == sid and x['block'] == 'train_orth']
            require(len(ar) == 1 and ar[0]['npz_sha256'] == row['sample']['sha256'] and
                    ar[0]['geometry_sha256'] == row['scene_id'], 'PLAN_DATASET_ANCHOR')
            seal = self._read(row['anchor']['seal'])
            require(seal['task_plan_sha'] == anchor['task_plan_sha256'] ==
                    row['scene_source']['sha256'], 'PLAN_SEAL')
        else:
            raise ValueError('UNKNOWN_ANCHOR')
        contract = self._read(row['contract_source'])
        require(contract['split'] == 'train' and geometry_digest(contract['scene']) == row['scene_id'] and
                contract['body_radius_m'] == row['body_radius_m'] and contract['margin_m'] == row['safety_margin_m'],
                'CONTRACT_SOURCE')
        camera = contract['camera']
        require(camera['optical_to_B0_rotation'] == [[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]] and
                camera['origin_B0'] == [0.,0.,0.], 'CAMERA_FRAME')
        if not row['obstacles']:
            require(scene.get('certified_empty') is True, 'SOURCE_EMPTY_CERTIFICATE')

    def load(self, samples):
        artifact(self.binding)
        result = []
        for sample in samples:
            require(sample.sample_id in self.rows, 'UNBOUND_SAMPLE')
            row = self.rows[sample.sample_id]
            self._validate_row(row)
            source = self._read(row['scene_source'])
            if row['anchor']['kind'] == 'task_plan':
                entry = source[int(row['scene_pointer'].split('/')[1])]
            else:
                entry = source
            require(np.array_equal(sample.state.detach().cpu().numpy(), np.asarray(entry['state'], np.float32)),
                    'SAMPLE_STATE_FRAME_MISMATCH')
            # Authenticate all model/teacher tensors, not only sample_id.
            original = self.dataset.load([sample.sample_id])[0]
            for key in ['state', 'depth_input', 'teacher_coeffs', 'teacher_durations']:
                require(torch.equal(getattr(sample, key).cpu(), getattr(original, key)), 'SAMPLE_TENSOR_BINDING')
            boxes = torch.tensor([[b['min'], b['max']] for b in row['obstacles']], dtype=torch.float32).reshape(-1,2,3)
            result.append(SimpleNamespace(boxes=boxes, row=copy.deepcopy(row)))
        self._json_cache.clear()
        return result
