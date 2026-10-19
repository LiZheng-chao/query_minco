"""Read-only original checkpoint/network smoke; never starts ROS or saves a model."""
from pathlib import Path
import sys

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "YOPO"))
    from config.config import cfg
    from policy.yopo_network import YopoNetwork

    torch.set_num_threads(1)
    cfg["train"] = False  # In-memory inference flag only; no YAML write.
    model = YopoNetwork().eval()
    checkpoint = root / "YOPO/saved/YOPO_1/epoch50.pth"
    loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    print("CHECKPOINT_STRICT", loaded)
    # Original StateTransform creates lattice constants on CUDA when visible.
    # Its existing CPU preparation method avoids changing those original internals.
    import numpy as np
    prepared = model.state_transform.prepare_input_cpu(np.zeros((2, 9), dtype=np.float32))
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        model = model.to(device)
        depth = torch.zeros(2, 1, cfg["image_height"], cfg["image_width"], device=device)
        obs = torch.from_numpy(prepared).to(device)
        with torch.inference_mode():
            output = model(depth, obs)
        shapes = [list(x.shape) for x in output]
        finite = [bool(torch.isfinite(x).all()) for x in output]
        assert shapes == [[2, 14, 3, 5], [2, 3, 5], [2, 20, 3, 5]]
        assert all(finite)
        print("BASELINE", device, "SHAPES", shapes, "FINITE", finite)
    print("NO_TRAINING_NO_ROS_NO_CHECKPOINT_WRITE")


if __name__ == "__main__":
    main()
