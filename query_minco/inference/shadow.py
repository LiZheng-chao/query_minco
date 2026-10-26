"""CPU-only frozen pair and sensor contracts. No controller/selector/GT map."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from query_minco.inference.identity import state_sha
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'research_outputs/rsps_forest15m_v1/preparation001/configs/FROZEN_MODELS.json'
K = np.array([[100., 0, 79.5], [0, 100., 63.5], [0, 0, 1.]])

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rotation(q):
    q = np.asarray(q, dtype=float)
    require(q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1) < .001, 'ODOM_QUATERNION')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def decode_image(data, encoding, width, height, step, bigendian):
    require((height, width) == (128, 160), 'DEPTH_SIZE_NO_RESIZE')
    require(encoding in ('32FC1', '16UC1'), 'DEPTH_ENCODING')
    dt = np.dtype(('>' if bigendian else '<')+('f4' if encoding == '32FC1' else 'u2'))
    require(step >= width*dt.itemsize and len(data) == height*step, 'DEPTH_STRIDE')
    z = np.ndarray((height, width), dtype=dt, buffer=data, strides=(step, dt.itemsize)).astype(np.float32)
    if encoding == '16UC1':
        z *= .001
    valid = np.isfinite(z) & (z > 0) & (z < 20)
    return np.stack([np.where(valid, z/20, 0), valid.astype(np.float32)]).astype(np.float32)

def camera_check(width, height, intrinsic, distortion, rectification, projection):
    require((height, width) == (128, 160), 'CAMERA_SIZE')
    require(np.allclose(np.reshape(intrinsic, (3, 3)), K, atol=1e-6, rtol=0), 'CAMERA_K')
    require(np.isfinite(distortion).all() and np.all(np.asarray(distortion) == 0), 'CAMERA_DISTORTION')
    require(np.allclose(np.reshape(rectification, (3, 3)), np.eye(3), atol=1e-6, rtol=0), 'CAMERA_R')
    require(np.allclose(np.reshape(projection, (3, 4)), np.c_[K, np.zeros(3)], atol=1e-6, rtol=0), 'CAMERA_P')

def point_depth_label(depth, point):
    """Ray diagnostic only: neither occupied-volume test nor body certificate."""
    x, y, z = np.asarray(point, dtype=float)
    if x <= .04:
        return 'UNKNOWN_BEHIND_CAMERA'
    u, v = int(round(100*(-y)/x+79.5)), int(round(100*(-z)/x+63.5))
    if not (1 <= u < 159 and 1 <= v < 127):
        return 'UNKNOWN_OUTSIDE_FOV'
    patch = depth[:, v-1:v+2, u-1:u+2]
    if not np.all(patch[1] == 1):
        return 'UNKNOWN_NO_HIT'
    nearest = float(patch[0].min()*20)
    if abs(nearest-x) <= .15:
        return 'SURFACE_OR_MARGIN_NEAR_GOAL'
    if nearest < x-.15:
        return 'OCCLUDED_GOAL_OCCUPANCY_UNKNOWN'
    return 'RAY_FRONT_ONLY_BODY_UNKNOWN'

def make_state(position, R, velocity_world, acceleration_world, global_goal):
    require(np.isfinite(np.concatenate([position, R.reshape(-1), velocity_world, acceleration_world, global_goal])).all(), 'STATE_NONFINITE')
    delta = np.asarray(global_goal)-np.asarray(position)
    distance = float(np.linalg.norm(delta))
    require(distance >= 2.4, 'TERMINAL_LT_2P4_UNSUPPORTED_NO_BRAKE')
    goal = R.T@delta*(2.4/distance)
    require(goal[0] > 0, 'GOAL_BEHIND_UNSUPPORTED')
    # Largest observed near-goal lateral/vertical slope: .42/2.4.
    require(np.all(np.abs(goal[1:]/goal[0]) <= .175+1e-6), 'GOAL_OUTSIDE_OBSERVED_DIRECTION_ENVELOPE')
    state = np.concatenate([R.T@velocity_world, R.T@acceleration_world, goal, R.T@[0., 0., -1.]]).astype(np.float32)
    ood = []
    for ok, tag in [(np.allclose(state[:3], [.8, 0, 0], atol=.05), 'VELOCITY_NOT_F1_POINT8'),
                    (np.linalg.norm(state[3:6]) <= .05, 'ACCELERATION_NOT_TRAIN_ZERO'),
                    (np.allclose(state[9:], [0, 0, -1], atol=.01), 'ATTITUDE_NOT_TRAIN_LEVEL'),
                    (np.allclose(goal, [2.4, 0, 0], atol=.01), 'GOAL_NOT_F1_FORWARD')]:
        if not ok:
            ood.append(tag)
    return state, dict(global_distance_m=distance, local_goal_B0=goal.tolist(),
                       local_goal_W=(np.asarray(position)+R@goal).tolist(), ood=ood)

class FrozenPair:
    def __init__(self, binding_sha):
        require(sha(FROZEN) == binding_sha, 'FROZEN_BINDING_SHA')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.models = {}
        self.bindings = [b for b in json.loads(FROZEN.read_text()) if b['label'] in ('C1', 'E3')]
        require([b['label'] for b in self.bindings] == ['C1', 'E3'], 'PAIR_IDENTITY')
        for b in self.bindings:
            require(sha(b['checkpoint']['path']) == b['checkpoint']['sha256'], 'CHECKPOINT_SHA')
            blob = torch.load(b['checkpoint']['path'], map_location='cpu', weights_only=True)
            if b['label'] == 'C1':
                require(blob['step'] == blob['optimizer_updates_completed'] == 2880 and blob['seed'] == 42 and blob['arm'] == 'C1_orthogonal', 'C1_IDENTITY')
            else:
                require(blob['completed_optimizer_updates'] == blob['order_cursor'] == 32 and blob['spec']['seed'] == 42 and blob['spec']['training_method_id'] == b['method_id'], 'E3_IDENTITY')
            model = QueryMincoPlanner()
            model.load_state_dict(blob['state_dict'], strict=True)
            require(state_sha(model) == b['weights_sha256'], 'WEIGHTS_SHA')
            model.eval().requires_grad_(False)
            self.models[b['label']] = model
        self.post = TrajectoryPostprocessor()

    def forward(self, depth, state):
        InferenceAdapter.inputs(depth[None], state[None], None)
        outputs = {}
        with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
            s = torch.from_numpy(state[None].copy())
            for label, model in self.models.items():
                raw = model(torch.from_numpy(depth[None].copy()), s)
                out = self.post(raw, s, num_samples=65)
                record = InferenceAdapter.export(out, raw, out.samples)
                require(all(np.isfinite(x).all() for x in record.values()), 'NONFINITE_OUTPUT')
                require(record['solve_valid'].all(), 'MINCO_SOLVE_INVALID')
                require(record['raw'].shape == (1, 6, 21), 'SIX_CANDIDATES')
                outputs[label] = {k: v[0] for k, v in record.items()}
        return outputs
