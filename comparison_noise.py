"""Deterministic noise replay for paired controller comparisons."""

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True, init=False)
class ReplayGaussianNoise:
    """Call-order-independent, piecewise-constant Gaussian noise replay.

    One sample is generated for each ``sample_period`` interval.  Looking up a
    sample is a pure function of time, so adaptive ODE solvers and diagnostic
    calls cannot change the future noise seen by a simulation.
    """

    seed: int
    sample_period: float
    t_end: float
    scale: tuple[float, float, float]
    identifier: str
    _samples: np.ndarray = field(repr=False)

    def __init__(
        self,
        seed: int,
        sample_period: float,
        t_end: float,
        scale: np.ndarray | tuple[float, float, float] = (0.2, 0.2, 0.1),
    ) -> None:
        if not np.isfinite(sample_period) or sample_period <= 0.0:
            raise ValueError("sample_period must be positive")
        if not np.isfinite(t_end) or t_end < 0.0:
            raise ValueError("t_end must be non-negative")
        noise_scale = np.asarray(scale, dtype=float)
        if (
            noise_scale.shape != (3,)
            or not np.all(np.isfinite(noise_scale))
            or np.any(noise_scale < 0.0)
        ):
            raise ValueError("scale must be a non-negative length-3 vector")

        replay_seed = int(seed)
        replay_period = float(sample_period)
        replay_end = float(t_end)
        count = max(1, int(np.ceil(replay_end / replay_period)) + 2)
        rng = np.random.default_rng(replay_seed)
        samples = rng.standard_normal((count, 3)) * noise_scale
        samples.setflags(write=False)
        digest = hashlib.sha256()
        digest.update(np.asarray([replay_seed, count], dtype=np.int64).tobytes())
        digest.update(np.asarray([replay_period, replay_end], dtype=np.float64).tobytes())
        digest.update(noise_scale.tobytes())
        digest.update(samples.tobytes())
        object.__setattr__(self, "seed", replay_seed)
        object.__setattr__(self, "sample_period", replay_period)
        object.__setattr__(self, "t_end", replay_end)
        object.__setattr__(self, "scale", tuple(float(value) for value in noise_scale))
        object.__setattr__(self, "_samples", samples)
        object.__setattr__(self, "identifier", digest.hexdigest()[:16])

    def __call__(self, t: float) -> np.ndarray:
        if not np.isfinite(t):
            raise ValueError("noise lookup time must be finite")
        lookup_time = float(t)
        tolerance = 1e-12 * max(1.0, self.t_end)
        if lookup_time < -tolerance or lookup_time > self.t_end + tolerance:
            raise ValueError(f"noise lookup time {lookup_time} outside [0, {self.t_end}]")
        lookup_time = min(max(0.0, lookup_time), self.t_end)
        index = int(np.floor(lookup_time / self.sample_period + 1e-12))
        index = min(index, len(self._samples) - 1)
        return self._samples[index].copy()


