"""Independent fixed P1 architecture; never imports YOPO cfg."""
from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class QueryMincoConfig:
    depth_channels: int = 2
    depth_height: int = 128
    depth_width: int = 160
    state_dim: int = 12
    embed_dim: int = 128
    num_queries: int = 6
    decoder_layers: int = 2
    attention_heads: int = 4
    ffn_dim: int = 256
    output_dim: int = 21
    dropout: float = 0.0

    def __post_init__(self):
        for field, expected in zip(fields(self), (2, 128, 160, 12, 128, 6, 2, 4, 256, 21)):
            value = getattr(self, field.name)
            if type(value) is not int or value != expected:
                raise ValueError("P1 fixes {}={} (got {!r})".format(field.name, expected, value))
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a finite number in [0, 1)")
        if not isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
