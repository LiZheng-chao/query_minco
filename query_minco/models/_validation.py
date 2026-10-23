"""Explicit contracts; no implicit resizing or repair."""
import torch


def check_tensor(value, name, trailing_shape, finite=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim != len(trailing_shape) + 1 or tuple(value.shape[1:]) != tuple(trailing_shape):
        raise ValueError("{} must have shape [B, {}], got {}".format(
            name, ", ".join(map(str, trailing_shape)), tuple(value.shape)))
    if value.shape[0] < 1:
        raise ValueError("{} batch must be nonempty".format(name))
    if not value.is_floating_point():
        raise TypeError("{} must have a floating point dtype".format(name))
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError("{} contains NaN or Inf".format(name))


def check_module_input(value, parameter, name):
    if value.device != parameter.device:
        raise ValueError("{} and model must be on the same device".format(name))
    if value.dtype != parameter.dtype:
        raise TypeError("{} dtype must match model dtype; use autocast for AMP".format(name))
