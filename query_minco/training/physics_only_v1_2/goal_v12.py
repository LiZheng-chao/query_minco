"""Goal V1.2 training surrogate, distinct from frozen hard evaluation."""
import torch

FRAME = "fixed B0 FLU"
UNITS = "m_s"


def finite_tensor(x, name):
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + ": FP32/FP64 tensor required")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(name + ": nonfinite")


def phi2(z):
    """2*k^2*(sqrt(1+(z/k)^2)-1), k=2, rationalized near zero.

    Equivalent form 2*k*z^2/(hypot(z,k)+k) avoids cancellation and z^2 overflow.
    No detach or clipping of the differentiable violation.
    """
    finite_tensor(z, "phi2")
    if bool((z < 0).any()):
        raise ValueError("phi2 requires nonnegative violations")
    k = z.new_full((), 2.)
    value = (2 * k * (z / (torch.hypot(z, k) + k))) * z
    finite_tensor(value, "phi2 result")
    return value


def validate_goal_batch(goal, enabled, task_types, *, frame=FRAME, units=UNITS):
    finite_tensor(goal, "goal")
    if goal.ndim != 2 or goal.shape[1] != 3 or len(goal) == 0:
        raise ValueError("goal [B,3] required")
    if frame != FRAME or units != UNITS:
        raise ValueError("goal frame/units mismatch")
    if not isinstance(enabled, torch.Tensor) or enabled.dtype != torch.bool or (
            enabled.shape != (len(goal),) or enabled.device != goal.device):
        raise ValueError("legal goal mask bool [B] required")
    if not isinstance(task_types, (tuple, list)) or len(task_types) != len(goal):
        raise ValueError("explicit task_types per scene required")
    distance = goal.norm(dim=-1)
    for i, task in enumerate(task_types):
        if task == "normal":
            if not bool(distance[i] > 0):
                raise ValueError("zero goal requires explicit hover task")
        elif task == "hover":
            if not bool(distance[i] == 0):
                raise ValueError("this hover extension is zero-displacement only")
        else:
            raise ValueError("unknown task type")
    return distance


def goal_terms(endpoint, terminal_velocity, goal, enabled, task_types, *, frame=FRAME, units=UNITS):
    """All differentiable terms [B,K]; goal/mask provenance is a caller contract."""
    distance = validate_goal_batch(goal, enabled, task_types, frame=frame, units=units)
    if endpoint.ndim != 3 or endpoint.shape[1] not in (1, 6):
        raise ValueError("goal candidate axis must be 1 or 6")
    candidates = endpoint.shape[1]
    for x, name in ((endpoint, "endpoint"), (terminal_velocity, "terminal velocity")):
        finite_tensor(x, name)
        if x.shape != (len(goal), candidates, 3) or x.dtype != goal.dtype or x.device != goal.device:
            raise ValueError("goal prediction shape/dtype/device mismatch")
    err = endpoint - goal[:, None]
    remaining = err.norm(dim=-1)
    speed = terminal_velocity.norm(dim=-1)
    near = phi2((remaining - .15).clamp_min(0) / .15) + phi2((speed - .25).clamp_min(0) / .25)
    unit = goal / distance.clamp_min(1.)[:, None]
    h = (endpoint * unit[:, None]).sum(-1)
    lateral = (endpoint - h[..., None] * unit[:, None]).norm(dim=-1)
    reduction = distance[:, None] - remaining
    far = phi2((1. - reduction).clamp_min(0)) + phi2((1. - h).clamp_min(0))
    far = far + phi2((lateral - .8).clamp_min(0) / .8)
    u = ((distance - 3.2) / .4).clamp(0., 1.)
    blend = u.square() * (3 - 2 * u)
    task = (1 - blend[:, None]) * near + blend[:, None] * far
    progress = (torch.sqrt(err.square().sum(-1) + .05**2) - .05) / distance.clamp_min(1.)[:, None]
    mask = enabled.to(goal.dtype)[:, None]
    # Diagnostics retain autograd too; only outer logging detaches.
    return dict(G_N=near * mask, G_F=far * mask, blend=blend[:, None].expand(-1, candidates),
                G_task_train=task * mask, P=progress * mask)
