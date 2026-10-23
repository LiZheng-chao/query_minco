"""Lightweight CNN; row-major 8x10 tokens with fixed 2D sine/cosine PE."""
import math
import torch
from torch import nn
from ._validation import check_tensor


def conv_norm_act(in_channels, out_channels, kernel_size, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=kernel_size // 2, groups=groups, bias=False),
        nn.GroupNorm(8, out_channels), nn.SiLU())


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = conv_norm_act(in_channels, in_channels, 3, stride, in_channels)
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                        nn.GroupNorm(8, out_channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.GroupNorm(8, out_channels))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.pointwise(self.depthwise(x)) + self.skip(x))


def image_position_encoding():
    """Token index = row*10+column; channels = y sin/cos then x sin/cos."""
    y, x = torch.meshgrid(torch.arange(8, dtype=torch.float32),
                          torch.arange(10, dtype=torch.float32), indexing="ij")
    frequency = torch.exp(-math.log(10000.0) * torch.arange(32) / 32)
    yp, xp = y.reshape(-1, 1) * frequency, x.reshape(-1, 1) * frequency
    return torch.cat((yp.sin(), yp.cos(), xp.sin(), xp.cos()), dim=-1)[None]


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_norm_act(2, 24, 3, 2)
        self.d2 = DepthwiseResidualBlock(24, 32, 2)
        self.d3 = DepthwiseResidualBlock(32, 64, 2)
        self.d4 = DepthwiseResidualBlock(64, 96, 2)
        self.d5 = DepthwiseResidualBlock(96, 96, 1)
        self.d6 = conv_norm_act(96, 128, 1)
        self.register_buffer("position_encoding", image_position_encoding())

    def forward(self, depth_input):
        check_tensor(depth_input, "depth_input", (2, 128, 160))
        x = self.d6(self.d5(self.d4(self.d3(self.d2(self.d1(depth_input))))))
        return x.flatten(2).transpose(1, 2) + self.position_encoding.to(dtype=x.dtype)
