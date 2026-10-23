"""State order: velocity, acceleration, local goal, gravity direction (xyz)."""
from torch import nn
from ._validation import check_tensor


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(12, 64), nn.SiLU(),
                                 nn.Linear(64, 128), nn.LayerNorm(128))

    def forward(self, state):
        check_tensor(state, "state", (12,))
        return self.mlp(state).unsqueeze(1)
