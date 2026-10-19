"""Independent explicit math pipeline; leaves QueryMincoPlanner.forward unchanged."""
from .parameter_decoder import PhysicalParameterDecoder
from .minco_adapter import MincoAdapter
from .sampler import TrajectorySampler
from .types import TrajectoryOutput


class TrajectoryPostprocessor:
    def __init__(self, config=None):
        self.decoder = PhysicalParameterDecoder(config)
        self.adapter = MincoAdapter()
        self.sampler = TrajectorySampler()

    def __call__(self, raw, state, num_samples=64):
        parameters = self.decoder(raw, state)
        trajectory = self.adapter(parameters.head_pva, parameters.tail_pva,
                                  parameters.inner_pts, parameters.durations)
        samples = self.sampler(trajectory, num_samples=num_samples)
        return TrajectoryOutput(parameters, trajectory, samples)
