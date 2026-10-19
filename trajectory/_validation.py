import torch

FLOAT_INPUTS = (torch.float16, torch.bfloat16, torch.float32)


def tensor(value, name, trailing_shape, fp32=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a Tensor".format(name))
    if value.ndim != len(trailing_shape) + 1 or tuple(value.shape[1:]) != trailing_shape:
        raise ValueError("{} must have shape [B,{}], got {}".format(
            name, ",".join(map(str, trailing_shape)), tuple(value.shape)))
    if value.shape[0] < 1:
        raise ValueError("{} requires a nonempty batch".format(name))
    allowed = (torch.float32,) if fp32 else FLOAT_INPUTS
    if value.dtype not in allowed:
        raise TypeError("{} dtype must be {}".format(name, "float32" if fp32 else "float16/bfloat16/float32"))
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError("{} device must be CPU or CUDA".format(name))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} contains NaN or Inf".format(name))


def matching(values):
    first = values[0]
    if any(v.shape[0] != first.shape[0] for v in values):
        raise ValueError("batch sizes must match")
    if any(v.device != first.device for v in values):
        raise ValueError("all inputs must be on the same device")
