"""Six state-conditioned learnable queries; no fixed spatial anchors."""
import torch
from torch import nn
from ._validation import check_tensor


class QueryDecoderBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(128, 4, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(128, 4, dropout=dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(128) for _ in range(3))
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(256, 128))

    def forward(self, queries, context, return_diagnostics=False):
        update, _ = self.self_attention(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + self.residual_dropout(update))
        update, weights = self.cross_attention(
            queries, context, context, need_weights=return_diagnostics, average_attn_weights=False)
        queries = self.norm2(queries + self.residual_dropout(update))
        queries = self.norm3(queries + self.residual_dropout(self.ffn(queries)))
        return queries, weights


class QueryDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.learned_queries = nn.Parameter(torch.empty(6, 128))
        nn.init.normal_(self.learned_queries, std=0.02)
        self.layers = nn.ModuleList([QueryDecoderBlock(dropout) for _ in range(2)])

    def forward(self, context, state_embed, return_diagnostics=False):
        check_tensor(context, "context", (81, 128))
        check_tensor(state_embed, "state_embed", (128,))
        if context.shape[0] != state_embed.shape[0]:
            raise ValueError("context and state_embed batch sizes must match")
        queries = self.learned_queries.unsqueeze(0) + state_embed.unsqueeze(1)
        weights = [] if return_diagnostics else None
        for layer in self.layers:
            queries, attention = layer(queries, context, return_diagnostics)
            if return_diagnostics:
                weights.append(attention)
        if return_diagnostics:
            return queries, torch.stack(weights, dim=1)
        return queries
