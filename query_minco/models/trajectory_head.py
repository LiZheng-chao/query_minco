"""Shared raw-logit head; no physical decoding or trajectory construction."""
from torch import nn
from ._validation import check_tensor


class TrajectoryHead(nn.Module):
    def __init__(self, *, num_queries=6):
        super().__init__()
        if type(num_queries) is not int or num_queries not in (1, 6):
            raise ValueError("num_queries must be 1 or 6")
        self.num_queries = num_queries
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 21))

    def forward(self, queries):
        check_tensor(queries, "queries", (self.num_queries, 128))
        return self.mlp(queries)
