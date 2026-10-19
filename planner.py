"""Tensor-only P1 network; raw output is not a verified physical trajectory."""
import torch
from torch import nn
from .config import QueryMincoConfig
from .depth_encoder import DepthEncoder
from .state_encoder import StateEncoder
from .query_decoder import QueryDecoder
from .trajectory_head import TrajectoryHead
from ._validation import check_tensor, check_module_input


class QueryMincoPlanner(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = QueryMincoConfig()
        if not isinstance(config, QueryMincoConfig):
            raise TypeError("config must be QueryMincoConfig")
        self.config = config
        self.depth_encoder = DepthEncoder()
        self.state_encoder = StateEncoder()
        self.query_decoder = QueryDecoder(config.dropout)
        self.trajectory_head = TrajectoryHead()

    def forward(self, depth_input, state, return_diagnostics=False):
        """Return raw [B,6,21], or (raw, diagnostics) when requested.

        Raw: q1/q2/q3=0:9, vT=9:12, aT=12:15, total-time logit=15,
        segment logits=16:19, lower-is-better score=19, feasibility logit=20.
        No bounds, MINCO, verification or selection are applied.
        """
        if type(return_diagnostics) is not bool:
            raise TypeError("return_diagnostics must be bool")
        check_tensor(depth_input, "depth_input", (2, 128, 160), finite=True)
        check_tensor(state, "state", (12,), finite=True)
        if depth_input.shape[0] != state.shape[0]:
            raise ValueError("depth_input and state batch sizes must match")
        if depth_input.device != state.device:
            raise ValueError("depth_input and state must be on the same device")
        if depth_input.dtype != state.dtype:
            raise TypeError("depth_input and state must have the same dtype")
        check_module_input(depth_input, next(self.parameters()), "depth_input")
        image_tokens = self.depth_encoder(depth_input)
        state_token = self.state_encoder(state)
        context = torch.cat((image_tokens, state_token), dim=1)
        decoded = self.query_decoder(context, state_token[:, 0], return_diagnostics)
        if return_diagnostics:
            decoded, attention = decoded
        raw_params = self.trajectory_head(decoded)
        if return_diagnostics:
            return raw_params, {
                "image_tokens": image_tokens, "state_token": state_token,
                "context": context, "decoder_tokens": decoded,
                "cross_attention": attention, "visual_grid": (8, 10),
                "flatten_order": "row-major: token_index = row * 10 + column",
                "state_token_index": 80}
        return raw_params
