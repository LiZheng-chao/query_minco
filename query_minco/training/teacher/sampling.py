import torch
from query_minco.trajectory import evaluate_piece
from query_minco.trajectory.cumulative_time import cumulative_three

def teacher_evaluate(coeffs,durations,times,order):
    # coeffs[3,3,6], durations[3], times[K,S] -> [K,S,3].
    boundaries=cumulative_three(durations)
    idx=(times[...,None]>=boundaries[:2]).sum(-1)
    starts=torch.cat((torch.zeros_like(boundaries[:1]),boundaries[:2]))
    local=times-starts[idx]
    return evaluate_piece(coeffs[idx],local[...,None],order).squeeze(-2)
