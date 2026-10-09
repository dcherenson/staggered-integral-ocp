"""Causal high-gain disturbance observer for the DOB-DT-MPC comparator.

The observer is independent of the DT-MPC controller.  It estimates an
additive acceleration residual from sampled velocity and nominal acceleration
data, then exposes the paper's Eq. 37 estimation-error margin.  The controller
receives only the resulting scalar disturbance bound; no observer estimate is
injected into its model or optimizer.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _vector(values: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite length-{length} vector")
    return result


def equation_37_margin(
    initial_error_bound: float,
    derivative_bound: float,
    gain: float,
    time: float,
) -> float:
    """Return the Eq. 37 bound on observer estimation error."""
    values = {
        "initial_error_bound": initial_error_bound,
        "derivative_bound": derivative_bound,
        "gain": gain,
        "time": time,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Eq. 37 inputs must be finite")
    if initial_error_bound < 0.0 or derivative_bound < 0.0 or gain <= 0.0 or time < 0.0:
        raise ValueError("Eq. 37 inputs have invalid signs")
    steady_squared = (derivative_bound / gain) ** 2
    squared_margin = (
        (initial_error_bound**2 - steady_squared) * np.exp(-gain * time)
        + steady_squared
    )
    if squared_margin < -1e-12:
        raise FloatingPointError("Eq. 37 produced a negative squared margin")
    return float(np.sqrt(max(0.0, squared_margin)))


class HighGainDisturbanceObserver:
    """Three-axis high-gain input observer with a first-order-hold update."""

    def __init__(self, gain: float, derivative_bound: float, initial_error_bound: float):
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("observer gain must be positive and finite")
        if not np.isfinite(derivative_bound) or derivative_bound < 0.0:
            raise ValueError("derivative bound must be non-negative and finite")
        if not np.isfinite(initial_error_bound) or initial_error_bound < 0.0:
            raise ValueError("initial error bound must be non-negative and finite")
        self.gain = float(gain)
        self.derivative_bound = float(derivative_bound)
        self.initial_error_bound = float(initial_error_bound)
        self.epsilon = np.zeros(3, dtype=float)
        self.time = 0.0

    def initialize(self, velocity: np.ndarray, d_hat0: Optional[np.ndarray] = None) -> None:
        """Initialize epsilon so the default initial estimate is exactly zero."""
        velocity_initial = _vector(velocity, 3, "velocity")
        estimate = np.zeros(3) if d_hat0 is None else _vector(d_hat0, 3, "d_hat0")
        self.epsilon = self.gain * velocity_initial - estimate
        self.time = 0.0

    def estimate(self, velocity: np.ndarray) -> np.ndarray:
        """Evaluate the current estimate without changing observer state."""
        return self.gain * _vector(velocity, 3, "velocity") - self.epsilon

    def error_bound(self, time: Optional[float] = None) -> float:
        """Evaluate Eq. 37 at the observer clock or an explicit time."""
        evaluation_time = self.time if time is None else float(time)
        return equation_37_margin(
            self.initial_error_bound,
            self.derivative_bound,
            self.gain,
            evaluation_time,
        )

    def update(
        self,
        velocity_start: np.ndarray,
        velocity_end: np.ndarray,
        nominal_acceleration_start: np.ndarray,
        nominal_acceleration_end: np.ndarray,
        dt: float,
    ) -> None:
        """Advance the sampled observer after receiving the next state.

        The nominal acceleration is the same residual reference used by the
        historical SIOCP diagnostic.  The first-order-hold correction avoids
        the endpoint-ZOH steady-state bias for a changing nominal input.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("observer update dt must be positive and finite")
        q_start = _vector(nominal_acceleration_start, 3, "nominal_acceleration_start")
        q_start = q_start + self.gain * _vector(velocity_start, 3, "velocity_start")
        q_end = _vector(nominal_acceleration_end, 3, "nominal_acceleration_end")
        q_end = q_end + self.gain * _vector(velocity_end, 3, "velocity_end")

        scaled_step = self.gain * float(dt)
        decay = float(np.exp(-scaled_step))
        one_minus_decay = float(-np.expm1(-scaled_step))
        if scaled_step < 1e-5:
            beta = scaled_step / 2.0 - scaled_step**2 / 6.0 + scaled_step**3 / 24.0
        else:
            beta = 1.0 - one_minus_decay / scaled_step
        self.epsilon = (
            decay * self.epsilon
            + one_minus_decay * q_start
            + beta * (q_end - q_start)
        )
        self.time += float(dt)


__all__ = ["HighGainDisturbanceObserver", "equation_37_margin"]
