import torch
from query_minco import QueryMincoPlanner

SEED = 42


def model_and_inputs(batch=2, device="cpu", dropout=0.0):
    from query_minco import QueryMincoConfig
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = QueryMincoPlanner(QueryMincoConfig(dropout=dropout)).to(device)
    depth = torch.rand(batch, 2, 128, 160, device=device)
    depth[:, 1] = torch.randint(0, 2, (batch, 128, 160), device=device).float()
    state = torch.randn(batch, 12, device=device)
    state[:, 9:12] = torch.nn.functional.normalize(state[:, 9:12], dim=1)
    return model, depth, state


def gradient_summary(model):
    groups = {
        "DepthEncoder": list(model.depth_encoder.parameters()),
        "StateEncoder": list(model.state_encoder.parameters()),
        "learned_queries": [model.query_decoder.learned_queries],
        "QueryDecoder": [p for n, p in model.query_decoder.named_parameters() if n != "learned_queries"],
        "TrajectoryHead": list(model.trajectory_head.parameters()),
    }
    result = {}
    for name, params in groups.items():
        grads = [p.grad for p in params]
        result[name] = {
            "parameter_tensors": len(params),
            "with_gradient": sum(g is not None for g in grads),
            "all_finite": all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
            "nonzero_tensors": sum(g is not None and bool(g.abs().sum() > 0) for g in grads),
            "l2_norm": sum(float(g.float().square().sum()) for g in grads if g is not None) ** 0.5,
        }
    return result
