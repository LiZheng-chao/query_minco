"""Shared raw-logit head; no physical decoding or trajectory construction."""
from torch import nn
from ._validation import check_tensor


class TrajectoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 21))

    def forward(self, queries):
        check_tensor(queries, "queries", (6, 128))
        return self.mlp(queries)
