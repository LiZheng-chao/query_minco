import torch
from query_minco.trajectory import MincoAdapter, PhysicalParameterDecoder, TrajectorySampler

SEED = 701


def fixture(batch=1, device="cpu"):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    head = torch.zeros(batch, 6, 3, 3, device=device)
    head[..., 1, :] = torch.tensor([0.7, 0.1, -0.05], device=device)
    head[..., 2, :] = torch.tensor([0.1, -0.05, 0.03], device=device)
    tail = torch.zeros_like(head)
    tail[..., 0, :] = torch.tensor([3., 0.1, 0.0], device=device)
    tail[..., 1, :] = torch.tensor([0.5, 0.1, 0.0], device=device)
    tail[..., 2, :] = torch.tensor([0.1, 0., 0.], device=device)
    inner = torch.tensor([[1., .2, .1], [2., -.1, .2]], device=device).expand(batch, 6, -1, -1).clone()
    durations = torch.tensor([.5, .7, .9], device=device).expand(batch, 6, -1).clone()
    for i in range(6):
        inner[:, i, :, 1] += .01*i
        durations[:, i] += .01*i
    for b in range(batch):
        inner[b, ..., 0] += .03*b
        tail[b, ..., 0, 1] += .02*b
        head[b, ..., 1, 2] += .01*b
    return head, tail, inner, durations


def raw_fixture(device="cpu"):
    torch.manual_seed(SEED)
    raw = (torch.randn(1, 6, 21, device=device) * .1)
    state = torch.tensor([[.7, .1, -.05, .1, -.05, .03, 4., 1., .2, 0., 0., -1.]], device=device)
    return raw, state


def trajectory_loss(raw, state):
    params = PhysicalParameterDecoder()(raw, state)
    solution = MincoAdapter()(params.head_pva, params.tail_pva, params.inner_pts, params.durations)
    samples = TrajectorySampler()(solution, num_samples=33)
    u = torch.linspace(0, 1, 33, device=raw.device)
    target = torch.stack((3*u, .3*u, -.1*u), dim=-1)
    return (samples.p-target).square().mean() + .07*(samples.v-.2).square().mean()
