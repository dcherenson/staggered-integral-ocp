"""Disturbance-observer robust CLF-ECBF control for the quadcopter case study.

This module is intentionally separate from the existing SIOCP implementation.
The equations implemented here are the high-gain input disturbance observer
and robust CLF/ECBF construction from Das and Murray, adapted to the
unmatched acceleration disturbance produced by :class:`plant.Plant`.

The public controller has two actuator interfaces.  Change only the constant
below to select the implementation used by the new runner:

    ACTUATOR_INTERFACE = "virtual_acceleration"
    ACTUATOR_INTERFACE = "direct_physical_input"

The virtual-acceleration interface gives the obstacle barrier a well-defined
relative degree two input at level hover.  The direct interface applies the
same paper construction to the existing [roll-rate, pitch-rate, thrust]
channels and is useful for studying the state-dependent authority of the
physical input map.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Optional

import numpy as np
# User-selectable switch requested for the two actuator interpretations.
ACTUATOR_INTERFACE = "virtual_acceleration"

VALID_ACTUATOR_INTERFACES = {
    "virtual_acceleration",
    "direct_physical_input",
}


class DisturbanceObserverQPError(RuntimeError):
    """Raised when the hard robust ECBF constraints are infeasible."""


@dataclass
class DOBConfig:
    """Tunable parameters for the disturbance-observer controller.

    ``derivative_bounds`` and ``initial_error_bounds`` may be supplied as
    arrays matching the generated barrier order.  If omitted, the controller
    derives sampled-time empirical values from the current plant model.  The
    latter is deliberately labeled empirical because the current Plant adds
    fresh Gaussian noise inside every ODE evaluation, which is not bounded in
    continuous time as required by the paper's theorem.
    """

    observer_gain: float = 5.0
    cbf_k_alpha: tuple[float, float] = (8.0, 4.0)
    clf_lambda: float = 0.5
    clf_penalty: float = 1.0

    kp_position: tuple[float, float, float] = (1.8, 1.8, 2.5)
    kv_position: tuple[float, float, float] = (1.8, 1.8, 2.0)
    obstacle_influence_distance: float = 1.8
    obstacle_repulsion_gain: float = 1.2
    route_clearance_margin: float = 0.25
    route_switch_distance: float = 1.0
    angle_rate_gain: float = 3.0
    angle_weight: float = 2.0

    position_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    velocity_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    goal_angle_weight: float = 1.0

    physical_rate_limit: float = 5.0
    thrust_min: float = 0.0
    thrust_max: float = 30.0
    virtual_lateral_accel_limit: float = 6.0
    virtual_vertical_accel_max: float = 15.0

    calibration_samples: int = 256
    calibration_dt: float = 0.05
    # The current Plant's per-call Gaussian noise is not continuously bounded;
    # these are sampled-time empirical calibration factors, not theorem-level
    # confidence multipliers.
    calibration_margin: float = 0.25
    initial_error_margin: float = 1.0
    calibration_percentile: float = 95.0
    calibration_seed: int = 9137

    derivative_bounds: Optional[np.ndarray] = None
    initial_error_bounds: Optional[np.ndarray] = None
    clf_derivative_bound: Optional[float] = None
    clf_initial_error_bound: Optional[float] = None


@dataclass
class SafetyBarrier:
    """One scalar safe-set function used by the robust ECBF."""

    label: str
    kind: str
    center: Optional[np.ndarray] = None
    radius: float = 0.0
    limit: float = 0.0
    sign: float = 1.0

    def h_and_hdot(self, x: np.ndarray) -> tuple[float, float]:
        if self.kind == "obstacle":
            rel = x[:3] - self.center
            return float(rel @ rel - self.radius**2), float(2.0 * rel @ x[3:6])
        return float(self.sign * (x[2] - self.limit)), float(self.sign * x[5])

    def disturbance_effect(self, x: np.ndarray, d_acc: np.ndarray) -> float:
        """Return b_e = d h^(r-1)/dt's unknown-input contribution."""
        if self.kind == "obstacle":
            return float(2.0 * (x[:3] - self.center) @ d_acc)
        return float(self.sign * d_acc[2])

    def nominal_ecbf_terms(
        self,
        x: np.ndarray,
        control: np.ndarray,
        plant: Any,
        actuator_interface: str,
    ) -> tuple[float, np.ndarray]:
        """Return the known h-double-dot term and its control coefficient."""
        if actuator_interface == "virtual_acceleration":
            acc = np.asarray(control, dtype=float)
            g_acc = np.asarray(plant.g, dtype=float)
            if self.kind == "obstacle":
                rel = x[:3] - self.center
                hddot_nom = 2.0 * (x[3:6] @ x[3:6]) + 2.0 * rel @ (g_acc + acc)
                coeff = 2.0 * rel
            else:
                hddot_nom = self.sign * (g_acc[2] + acc[2])
                coeff = self.sign * np.array([0.0, 0.0, 1.0])
            return float(hddot_nom), coeff

        if actuator_interface != "direct_physical_input":
            raise ValueError(f"Unknown actuator interface: {actuator_interface}")

        B = np.asarray(plant.g_mat(x)[3:6, :], dtype=float)
        g_acc = np.asarray(plant.g, dtype=float)
        acc_nom = g_acc + B @ np.asarray(control, dtype=float)
        if self.kind == "obstacle":
            rel = x[:3] - self.center
            hddot_nom = 2.0 * (x[3:6] @ x[3:6]) + 2.0 * rel @ acc_nom
            coeff = 2.0 * rel @ B
        else:
            hddot_nom = self.sign * acc_nom[2]
            coeff = self.sign * B[2, :]
        return float(hddot_nom), np.asarray(coeff, dtype=float)


class HighGainDisturbanceObserver:
    """Scalar observer from equations (21)-(23) and bound (37).

    The continuous observer state epsilon is advanced with the exact
    zero-order-hold solution of equation (22), using the measured z and known
    input a over one simulation sample.  This is numerically stable for the
    high gains used in the paper.
    """

    def __init__(self, gain: float, derivative_bound: float, initial_error_bound: float):
        if gain <= 0.0:
            raise ValueError("Observer gain must be positive")
        if derivative_bound < 0.0 or initial_error_bound < 0.0:
            raise ValueError("Observer bounds must be non-negative")
        self.gain = float(gain)
        self.derivative_bound = float(derivative_bound)
        self.initial_error_bound = float(initial_error_bound)
        self.epsilon = 0.0
        self.time = 0.0

    def initialize(self, z0: float, b_hat0: float = 0.0) -> None:
        self.epsilon = self.gain * float(z0) - float(b_hat0)
        self.time = 0.0

    def estimate(self, z: float) -> float:
        return float(self.gain * float(z) - self.epsilon)

    def error_bound(self, time: Optional[float] = None) -> float:
        t = self.time if time is None else max(0.0, float(time))
        steady_sq = (self.derivative_bound / self.gain) ** 2
        transient_sq = self.initial_error_bound**2 - steady_sq
        value = transient_sq * np.exp(-self.gain * t) + steady_sq
        return float(np.sqrt(max(0.0, value)))

    def update(self, z: float, a: float, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("Observer update dt must be positive")
        decay = float(np.exp(-self.gain * dt))
        epsilon_equilibrium = float(a) + self.gain * float(z)
        self.epsilon = decay * self.epsilon + (1.0 - decay) * epsilon_equilibrium
        self.time += float(dt)


def _solve_small_convex_qp(
    A: np.ndarray,
    c: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: np.ndarray,
    clf_penalty: float,
) -> tuple[Optional[np.ndarray], float]:
    """Solve the four-variable strictly convex QP by active-set enumeration.

    The paper's CLF-ECBF program has a diagonal quadratic objective and affine
    constraints.  The current controller has only three input variables plus
    the CLF relaxation, so enumerating active sets is small, deterministic,
    and avoids depending on a platform-specific QP plugin.
    """
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    nominal = np.asarray(nominal, dtype=float)
    n = nominal.size
    if n != 4 or A.shape[1] != n:
        raise ValueError("The DOB QP expects four decision variables")
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("QP bounds must match the decision dimension")

    H = np.diag([2.0, 2.0, 2.0, 2.0 * float(clf_penalty)])
    # All inequalities use the convention A_i y + c_i >= 0.
    inequality_A = np.vstack([A, np.eye(n), -np.eye(n)])
    inequality_c = np.concatenate([c, -lower, upper])
    m = inequality_A.shape[0]
    tolerance = 2e-7
    best: Optional[np.ndarray] = None
    best_objective = np.inf

    max_active = min(n, m)
    for active_count in range(max_active + 1):
        for active_indices in combinations(range(m), active_count):
            if active_count == 0:
                candidate = nominal.copy()
                multipliers = np.empty(0)
            else:
                C = inequality_A[list(active_indices)]
                d = -inequality_c[list(active_indices)]
                KKT = np.block([
                    [H, C.T],
                    [C, np.zeros((active_count, active_count))],
                ])
                rhs = np.concatenate([H @ nominal, d])
                try:
                    solution = np.linalg.solve(KKT, rhs)
                except np.linalg.LinAlgError:
                    solution, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
                candidate = solution[:n]
                multipliers = solution[n:]
                if np.linalg.norm(C @ candidate - d, ord=np.inf) > 5e-6:
                    continue
                # The equality multiplier is -lambda for g >= 0; active
                # inequality multipliers therefore must be non-positive.
                if len(multipliers) and np.max(multipliers) > tolerance:
                    continue

            values = inequality_A @ candidate + inequality_c
            if np.min(values) < -tolerance:
                continue
            objective = float(
                np.sum((candidate[:3] - nominal[:3]) ** 2)
                + clf_penalty * candidate[3] ** 2
            )
            if objective < best_objective:
                best = candidate
                best_objective = objective

    return best, best_objective


def _as_array(values: tuple[float, ...] | np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"Expected a length-{length} vector, got {array.shape}")
    return array


class DisturbanceObserverController:
    """Robust disturbance-observer CLF-ECBF-QP controller.

    The controller works with the unchanged :class:`plant.Plant` object.  In
    virtual mode its QP variable is a thrust-acceleration vector.  In direct
    mode its QP variable is the plant's physical [p, q, T] input.
    """

    def __init__(
        self,
        plant: Any,
        obstacles: list[dict[str, Any]],
        x_goal: np.ndarray,
        z_min: float = 0.8,
        z_max: float = 1.2,
        config: Optional[DOBConfig] = None,
        actuator_interface: Optional[str] = None,
        x0: Optional[np.ndarray] = None,
    ):
        self.plant = plant
        self.x_goal = np.asarray(x_goal, dtype=float).copy()
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.config = config or DOBConfig()
        self.interface = actuator_interface or ACTUATOR_INTERFACE
        if self.interface not in VALID_ACTUATOR_INTERFACES:
            raise ValueError(
                f"actuator_interface must be one of {sorted(VALID_ACTUATOR_INTERFACES)}"
            )

        self.k_alpha = _as_array(self.config.cbf_k_alpha, 2)
        if self.k_alpha[0] <= 0.0 or self.k_alpha[1] <= 0.0:
            raise ValueError("ECBF gains must be positive")

        self.barriers = self._make_barriers(obstacles)
        if x0 is None:
            x0 = np.zeros(8, dtype=float)
        self.x0 = np.asarray(x0, dtype=float).copy()

        if self.config.derivative_bounds is None or self.config.initial_error_bounds is None:
            derived = self._derive_observer_bounds()
            derivative_bounds = derived["barrier_derivative_bounds"]
            initial_error_bounds = derived["barrier_initial_error_bounds"]
            self.bound_source = "sampled_empirical_model_calibration"
        else:
            derivative_bounds = np.asarray(self.config.derivative_bounds, dtype=float)
            initial_error_bounds = np.asarray(self.config.initial_error_bounds, dtype=float)
            self.bound_source = "user_configured"

        if derivative_bounds.shape != (len(self.barriers),):
            raise ValueError("derivative_bounds must match the number of safety barriers")
        if initial_error_bounds.shape != (len(self.barriers),):
            raise ValueError("initial_error_bounds must match the number of safety barriers")
        self.derivative_bounds = np.maximum(0.0, derivative_bounds)
        self.initial_error_bounds = np.maximum(0.0, initial_error_bounds)

        if self.config.clf_derivative_bound is None or self.config.clf_initial_error_bound is None:
            derived_clf = self._derive_clf_bounds()
            self.clf_derivative_bound = derived_clf["derivative_bound"]
            self.clf_initial_error_bound = derived_clf["initial_error_bound"]
        else:
            self.clf_derivative_bound = float(self.config.clf_derivative_bound)
            self.clf_initial_error_bound = float(self.config.clf_initial_error_bound)

        self.observers = [
            HighGainDisturbanceObserver(
                gain=self.config.observer_gain,
                derivative_bound=self.derivative_bounds[i],
                initial_error_bound=self.initial_error_bounds[i],
            )
            for i in range(len(self.barriers))
        ]
        for barrier, observer in zip(self.barriers, self.observers):
            _, hdot = barrier.h_and_hdot(self.x0)
            observer.initialize(hdot, b_hat0=0.0)

        self.clf_observer = HighGainDisturbanceObserver(
            gain=self.config.observer_gain,
            derivative_bound=self.clf_derivative_bound,
            initial_error_bound=self.clf_initial_error_bound,
        )
        V0, _, _, _ = self.clf_terms(self.x0, self.baseline_control(self.x0))
        self.clf_observer.initialize(V0, b_hat0=0.0)

        self.last_solution: Optional[np.ndarray] = None
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _make_barriers(obstacles: list[dict[str, Any]]) -> list[SafetyBarrier]:
        barriers = []
        for i, obstacle in enumerate(obstacles):
            barriers.append(
                SafetyBarrier(
                    label=f"obstacle_{i}",
                    kind="obstacle",
                    center=np.asarray(obstacle["pos"], dtype=float).copy(),
                    radius=float(obstacle["r"]),
                )
            )
        return barriers

    def add_altitude_barriers(self) -> None:
        """Add the existing lower/upper altitude limits to the ECBF set."""
        self.barriers.extend(
            [
                SafetyBarrier("altitude_floor", "altitude", limit=self.z_min, sign=1.0),
                SafetyBarrier("altitude_ceiling", "altitude", limit=self.z_max, sign=-1.0),
            ]
        )
        # Rebuild observers after adding barriers.  This method is intended to
        # be called immediately after construction and before simulation.
        self._rebuild_observers()

    def _rebuild_observers(self) -> None:
        derived = self._derive_observer_bounds()
        self.derivative_bounds = derived["barrier_derivative_bounds"]
        self.initial_error_bounds = derived["barrier_initial_error_bounds"]
        self.observers = [
            HighGainDisturbanceObserver(
                self.config.observer_gain,
                self.derivative_bounds[i],
                self.initial_error_bounds[i],
            )
            for i in range(len(self.barriers))
        ]
        for barrier, observer in zip(self.barriers, self.observers):
            _, hdot = barrier.h_and_hdot(self.x0)
            observer.initialize(hdot, b_hat0=0.0)

    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.interface == "direct_physical_input":
            return (
                np.array([-self.config.physical_rate_limit, -self.config.physical_rate_limit, self.config.thrust_min]),
                np.array([self.config.physical_rate_limit, self.config.physical_rate_limit, self.config.thrust_max]),
            )
        # Keep the virtual command comfortably inside the thrust ball and
        # within the attitude-rate mapping's practical tracking envelope.
        lateral = float(self.config.virtual_lateral_accel_limit)
        vertical = float(self.config.virtual_vertical_accel_max)
        if lateral <= 0.0 or vertical <= 0.0:
            raise ValueError("Virtual acceleration limits must be positive")
        if np.sqrt(2.0 * lateral**2 + vertical**2) * self.plant.m > self.config.thrust_max:
            raise ValueError("Virtual acceleration box exceeds the physical thrust limit")
        return np.array([-lateral, -lateral, 0.0]), np.array([lateral, lateral, vertical])

    def baseline_control(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        kp = _as_array(self.config.kp_position, 3)
        kv = _as_array(self.config.kv_position, 3)
        desired_position = self.x_goal[:3].copy()
        # Use the baseline-controller freedom in the paper's QP to provide a
        # simple route over the configured obstacle row.  The safety proof is
        # still carried by the hard ECBFs; this only prevents the CLF baseline
        # from repeatedly asking to pass through the obstacle centers.
        if x[0] < self.x_goal[0] - self.config.route_switch_distance:
            x_start = self.x0[0]
            x_goal = self.x_goal[0]
            lo, hi = min(x_start, x_goal), max(x_start, x_goal)
            obstacle_tops = [
                barrier.center[1] + barrier.radius + self.config.route_clearance_margin
                for barrier in self.barriers
                if barrier.kind == "obstacle" and lo <= barrier.center[0] <= hi
            ]
            if obstacle_tops:
                desired_position[1] = max(desired_position[1], max(obstacle_tops))
        desired_total_acc = kp * (desired_position - x[:3]) + kv * (self.x_goal[3:6] - x[3:6])
        # The paper's QP minimally modifies a baseline controller.  A small
        # smooth potential-field term gives that baseline the same
        # obstacle-aware intent that the existing SIOCP MPC has, leaving the
        # robust ECBF as the hard safety mechanism.
        influence = float(self.config.obstacle_influence_distance)
        repulsion_gain = float(self.config.obstacle_repulsion_gain)
        for barrier in self.barriers:
            if barrier.kind != "obstacle" or influence <= 0.0:
                continue
            rel = x[:3] - barrier.center
            distance = max(float(np.linalg.norm(rel)), 1e-6)
            if distance < influence:
                direction = rel / distance
                scale = repulsion_gain * (1.0 / distance - 1.0 / influence) / (distance**2)
                desired_total_acc += scale * direction
        desired_thrust_acc = desired_total_acc - np.asarray(self.plant.g, dtype=float)

        if self.interface == "virtual_acceleration":
            lower, upper = self.control_bounds()
            return np.clip(desired_thrust_acc, lower, upper)

        thrust = self.plant.m * np.linalg.norm(desired_thrust_acc)
        horizontal = np.hypot(desired_thrust_acc[0], desired_thrust_acc[1])
        theta_des = np.arctan2(desired_thrust_acc[0], desired_thrust_acc[2])
        phi_des = np.arctan2(-desired_thrust_acc[1], max(horizontal, 1e-8))
        rate_gain = float(self.config.angle_rate_gain)
        angle_goal = self.x_goal[6:8] if self.x_goal.shape[0] >= 8 else np.zeros(2)
        physical = np.array(
            [
                rate_gain * (phi_des - x[6] + angle_goal[0]),
                rate_gain * (theta_des - x[7] + angle_goal[1]),
                thrust,
            ],
            dtype=float,
        )
        lower, upper = self.control_bounds()
        return np.clip(physical, lower, upper)

    def map_virtual_to_physical(self, x: np.ndarray, virtual_acceleration: np.ndarray) -> np.ndarray:
        """Map a QP thrust-acceleration command to the unchanged Plant input."""
        a_thr = np.asarray(virtual_acceleration, dtype=float)
        thrust = self.plant.m * np.linalg.norm(a_thr)
        horizontal = np.hypot(a_thr[0], a_thr[1])
        theta_des = np.arctan2(a_thr[0], a_thr[2])
        phi_des = np.arctan2(-a_thr[1], max(horizontal, 1e-8))
        physical = np.array(
            [
                self.config.angle_rate_gain * (phi_des - x[6]),
                self.config.angle_rate_gain * (theta_des - x[7]),
                thrust,
            ],
            dtype=float,
        )
        lower = np.array([-self.config.physical_rate_limit] * 2 + [self.config.thrust_min])
        upper = np.array([self.config.physical_rate_limit] * 2 + [self.config.thrust_max])
        return np.clip(physical, lower, upper)

    def clf_terms(
        self,
        x: np.ndarray,
        control: np.ndarray,
        d_acc: Optional[np.ndarray] = None,
    ) -> tuple[float, float, np.ndarray, float]:
        """Return V, LfV, LgV, and the sampled unknown term b_V."""
        x = np.asarray(x, dtype=float)
        ep = x[:3] - self.x_goal[:3]
        ev = x[3:6] - self.x_goal[3:6]
        wp = _as_array(self.config.position_weights, 3)
        wv = _as_array(self.config.velocity_weights, 3)
        V = 0.5 * float(wp @ (ep * ep) + wv @ (ev * ev))
        angle_goal = self.x_goal[6:8] if self.x_goal.shape[0] >= 8 else np.zeros(2)

        grad_p = wp * ep
        grad_v = wv * ev
        g_acc = np.asarray(self.plant.g, dtype=float)
        LfV = float(grad_p @ x[3:6] + grad_v @ g_acc)

        if self.interface == "virtual_acceleration":
            LgV = grad_v.copy()
            if self.config.goal_angle_weight:
                # Attitude tracking is handled by the mapper; it is not a QP
                # variable in virtual mode and is therefore omitted from V.
                pass
        else:
            B = np.asarray(self.plant.g_mat(x)[3:6, :], dtype=float)
            LgV = grad_v @ B
            angle_error = x[6:8] - angle_goal
            V += 0.5 * self.config.angle_weight * float(angle_error @ angle_error)
            LfV += 0.0
            LgV = np.asarray(LgV, dtype=float)
            LgV[0] += self.config.angle_weight * angle_error[0]
            LgV[1] += self.config.angle_weight * angle_error[1]

        bV = 0.0 if d_acc is None else float(grad_v @ np.asarray(d_acc, dtype=float))
        return float(V), LfV, np.asarray(LgV, dtype=float), bV

    def _derive_observer_bounds(self) -> dict[str, np.ndarray]:
        """Estimate finite sampled-time observer bounds without altering RNG state."""
        rng_state = np.random.get_state()
        np.random.seed(self.config.calibration_seed)
        try:
            n = max(8, int(self.config.calibration_samples))
            dt = float(self.config.calibration_dt)
            derivative_samples = [[] for _ in self.barriers]
            initial = np.zeros(len(self.barriers), dtype=float)
            for sample in range(n):
                ratio = sample / max(1, n - 1)
                x = self.x0.copy()
                x[:3] = (1.0 - ratio) * self.x0[:3] + ratio * self.x_goal[:3]
                x[3:6] = np.asarray(self.x_goal[3:6]) * ratio
                x[6:8] = (
                    self.x0[6:8].copy()
                    if sample == 0
                    else np.random.uniform(-0.25, 0.25, size=2)
                )
                u = self.baseline_control(x)
                d0 = self.plant.unmodeled_dynamics(0.0, x[:3], x[3:6], x[6:8]) / self.plant.m
                x1 = x.copy()
                x1[:3] += dt * x[3:6]
                if self.interface == "virtual_acceleration":
                    x1[3:6] += dt * (np.asarray(self.plant.g) + u)
                else:
                    x1[3:6] += dt * (np.asarray(self.plant.g) + self.plant.g_mat(x)[3:6, :] @ u)
                d1 = self.plant.unmodeled_dynamics(dt, x1[:3], x1[3:6], x1[6:8]) / self.plant.m
                for i, barrier in enumerate(self.barriers):
                    b0 = barrier.disturbance_effect(x, d0)
                    b1 = barrier.disturbance_effect(x1, d1)
                    # Equation (37) uses the observer's initial error only at
                    # the actual initial state.  Do not replace it with the
                    # largest disturbance seen later in calibration; that
                    # would incorrectly make the transient margin global.
                    if sample == 0:
                        initial[i] = abs(b0)
                    derivative_samples[i].append(abs(b1 - b0) / dt)
            percentile = float(np.clip(self.config.calibration_percentile, 50.0, 100.0))
            deriv = np.array([
                np.percentile(samples, percentile) if samples else 0.0
                for samples in derivative_samples
            ])
            deriv = np.maximum(1e-3, self.config.calibration_margin * deriv)
            initial = np.maximum(1e-3, self.config.initial_error_margin * initial)
            return {
                "barrier_derivative_bounds": deriv,
                "barrier_initial_error_bounds": initial,
            }
        finally:
            np.random.set_state(rng_state)

    def _derive_clf_bounds(self) -> dict[str, float]:
        rng_state = np.random.get_state()
        np.random.seed(self.config.calibration_seed + 1)
        try:
            n = max(8, int(self.config.calibration_samples))
            dt = float(self.config.calibration_dt)
            initial = 0.0
            derivative_samples = []
            for sample in range(n):
                ratio = sample / max(1, n - 1)
                x = self.x0.copy()
                x[:3] = (1.0 - ratio) * self.x0[:3] + ratio * self.x_goal[:3]
                x[3:6] = np.asarray(self.x_goal[3:6]) * ratio
                x[6:8] = (
                    self.x0[6:8].copy()
                    if sample == 0
                    else np.random.uniform(-0.25, 0.25, size=2)
                )
                u = self.baseline_control(x)
                d0 = self.plant.unmodeled_dynamics(0.0, x[:3], x[3:6], x[6:8]) / self.plant.m
                V0, _, _, b0 = self.clf_terms(x, u, d0)
                x1 = x.copy()
                x1[:3] += dt * x[3:6]
                if self.interface == "virtual_acceleration":
                    x1[3:6] += dt * (np.asarray(self.plant.g) + u)
                else:
                    x1[3:6] += dt * (np.asarray(self.plant.g) + self.plant.g_mat(x)[3:6, :] @ u)
                d1 = self.plant.unmodeled_dynamics(dt, x1[:3], x1[3:6], x1[6:8]) / self.plant.m
                _, _, _, b1 = self.clf_terms(x1, self.baseline_control(x1), d1)
                if sample == 0:
                    initial = abs(b0)
                derivative_samples.append(abs(b1 - b0) / dt)
            percentile = float(np.clip(self.config.calibration_percentile, 50.0, 100.0))
            deriv = np.percentile(derivative_samples, percentile) if derivative_samples else 0.0
            return {
                "initial_error_bound": max(1e-3, self.config.initial_error_margin * initial),
                "derivative_bound": max(1e-3, self.config.calibration_margin * deriv),
            }
        finally:
            np.random.set_state(rng_state)

    def compute_control(self, x: np.ndarray, time: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve the robust CLF-ECBF-QP and return a physical Plant input."""
        x = np.asarray(x, dtype=float)
        qp_nominal = self.baseline_control(x)
        lower, upper = self.control_bounds()
        b_rows = []
        b_offsets = []
        barrier_diag = []

        for barrier, observer in zip(self.barriers, self.observers):
            h, hdot = barrier.h_and_hdot(x)
            b_hat = observer.estimate(hdot)
            M = observer.error_bound()
            a_nom_zero, coeff = barrier.nominal_ecbf_terms(
                x, np.zeros_like(qp_nominal), self.plant, self.interface
            )
            # nominal_ecbf_terms is affine in the selected input.  Evaluate at
            # zero, then add the coefficient times the QP input below.
            c = a_nom_zero + b_hat - M + self.k_alpha @ np.array([h, hdot])
            row = np.zeros(4, dtype=float)
            row[:3] = coeff
            b_rows.append(row)
            b_offsets.append(float(c))
            barrier_diag.append(
                {
                    "label": barrier.label,
                    "h": h,
                    "hdot": hdot,
                    "b_hat": b_hat,
                    "M": M,
                    "a_nom_zero": a_nom_zero,
                    "coeff": coeff.copy(),
                    "constraint_offset": c,
                }
            )

        V, LfV, LgV, _ = self.clf_terms(x, qp_nominal)
        b_hat_V = self.clf_observer.estimate(V)
        M_V = self.clf_observer.error_bound()
        clf_rhs = -self.config.clf_lambda * V - LfV - b_hat_V - M_V
        clf_row = np.zeros(4, dtype=float)
        clf_row[:3] = -LgV
        clf_row[3] = 1.0
        b_rows.append(clf_row)
        b_offsets.append(float(clf_rhs))

        A = np.asarray(b_rows, dtype=float)
        c = np.asarray(b_offsets, dtype=float)

        # Each row represents A_i y + c_i >= 0.  For the CLF row this is
        # -LgV*u + delta + clf_rhs >= 0.
        qp_lower = np.concatenate([lower, [-1e4]])
        qp_upper = np.concatenate([upper, [1e4]])
        y_nominal = np.concatenate([np.clip(qp_nominal, lower, upper), [0.0]])
        solution, objective_value = _solve_small_convex_qp(
            A=A,
            c=c,
            lower=qp_lower,
            upper=qp_upper,
            nominal=y_nominal,
            clf_penalty=self.config.clf_penalty,
        )
        feasible = solution is not None
        if not feasible:
            message = f"DOB CLF-ECBF QP failed at t={time:.3f}s: no feasible active set"
            self.last_diagnostics = {"success": False, "message": message, "constraints": A @ y_nominal + c}
            raise DisturbanceObserverQPError(message)

        qp_solution = np.asarray(solution, dtype=float)
        qp_input = qp_solution[:3]
        delta = float(qp_solution[3])
        physical_input = (
            self.map_virtual_to_physical(x, qp_input)
            if self.interface == "virtual_acceleration"
            else qp_input.copy()
        )

        residuals = np.array([
            barrier["constraint_offset"] + barrier["coeff"] @ qp_input
            for barrier in barrier_diag
        ])
        clf_residual = float((A @ qp_solution + c)[-1])
        diagnostics = {
            "success": True,
            "time": float(time),
            "interface": self.interface,
            "qp_input": qp_input.copy(),
            "physical_input": physical_input.copy(),
            "baseline_input": qp_nominal.copy(),
            "delta": delta,
            "barriers": barrier_diag,
            "barrier_residuals": residuals,
            "clf_V": V,
            "clf_LfV": LfV,
            "clf_LgV": LgV.copy(),
            "clf_b_hat": b_hat_V,
            "clf_M": M_V,
            "clf_residual": clf_residual,
            "solver_message": f"active-set QP objective={objective_value:.9g}",
            "bound_source": self.bound_source,
        }
        self.last_solution = qp_solution.copy()
        self.last_diagnostics = diagnostics
        return physical_input, diagnostics

    def update_observers(
        self,
        x: np.ndarray,
        applied_qp_input: np.ndarray,
        dt: float,
        measurement_state: Optional[np.ndarray] = None,
    ) -> None:
        """Advance observers after the known input is applied.

        ``x`` is the state at which the known nominal dynamics are evaluated.
        ``measurement_state`` can be the post-step measured state, allowing a
        sampled implementation to use the newest h^(r-1) measurement while
        retaining the nominal a_e from the interval that just elapsed.
        """
        measured = x if measurement_state is None else np.asarray(measurement_state, dtype=float)
        for barrier, observer in zip(self.barriers, self.observers):
            _, hdot = barrier.h_and_hdot(measured)
            a_nom, _ = barrier.nominal_ecbf_terms(
                x, applied_qp_input, self.plant, self.interface
            )
            observer.update(hdot, a_nom, dt)
        V, _, _, _ = self.clf_terms(measured, applied_qp_input)
        _, LfV, LgV, _ = self.clf_terms(x, applied_qp_input)
        self.clf_observer.update(V, LfV + float(LgV @ applied_qp_input), dt)

    def observer_snapshot(self, x: np.ndarray, d_acc: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Return estimates, bounds, and optional sampled true effects for logging."""
        estimates = []
        bounds = []
        true_effects = []
        for barrier, observer in zip(self.barriers, self.observers):
            _, hdot = barrier.h_and_hdot(x)
            estimates.append(observer.estimate(hdot))
            bounds.append(observer.error_bound())
            if d_acc is not None:
                true_effects.append(barrier.disturbance_effect(x, d_acc))
        V, _, _, true_clf = self.clf_terms(x, self.baseline_control(x), d_acc)
        return {
            "barrier_b_hat": np.asarray(estimates, dtype=float),
            "barrier_M": np.asarray(bounds, dtype=float),
            "barrier_b_true": np.asarray(true_effects, dtype=float) if d_acc is not None else None,
            "clf_V": V,
            "clf_b_hat": self.clf_observer.estimate(V),
            "clf_M": self.clf_observer.error_bound(),
            "clf_b_true": true_clf if d_acc is not None else None,
        }


def make_controller(
    plant: Any,
    obstacles: list[dict[str, Any]],
    x_goal: np.ndarray,
    x0: np.ndarray,
    z_min: float = 0.8,
    z_max: float = 1.2,
    config: Optional[DOBConfig] = None,
    actuator_interface: Optional[str] = None,
) -> DisturbanceObserverController:
    """Convenience constructor with all current altitude barriers enabled."""
    config_for_init = config
    full_barrier_count = len(obstacles) + 2
    if config is not None and (
        config.derivative_bounds is not None
        or config.initial_error_bounds is not None
    ):
        # The base constructor initially sees obstacle barriers only.  Defer
        # full user-provided arrays until the two altitude barriers are added.
        config_for_init = replace(
            config,
            derivative_bounds=None,
            initial_error_bounds=None,
        )
    controller = DisturbanceObserverController(
        plant=plant,
        obstacles=obstacles,
        x_goal=x_goal,
        z_min=z_min,
        z_max=z_max,
        config=config_for_init,
        actuator_interface=actuator_interface,
        x0=x0,
    )
    controller.add_altitude_barriers()
    if config is not None and (
        config.derivative_bounds is not None
        or config.initial_error_bounds is not None
    ):
        if config.derivative_bounds is None or config.initial_error_bounds is None:
            raise ValueError("Both observer bound arrays must be supplied together")
        derivative_bounds = np.asarray(config.derivative_bounds, dtype=float)
        initial_error_bounds = np.asarray(config.initial_error_bounds, dtype=float)
        if derivative_bounds.shape != (full_barrier_count,):
            raise ValueError("derivative_bounds must match obstacles plus two altitude barriers")
        if initial_error_bounds.shape != (full_barrier_count,):
            raise ValueError("initial_error_bounds must match obstacles plus two altitude barriers")
        controller.config = config
        controller.derivative_bounds = np.maximum(0.0, derivative_bounds)
        controller.initial_error_bounds = np.maximum(0.0, initial_error_bounds)
        controller.bound_source = "user_configured"
        controller.observers = [
            HighGainDisturbanceObserver(
                config.observer_gain,
                controller.derivative_bounds[i],
                controller.initial_error_bounds[i],
            )
            for i in range(full_barrier_count)
        ]
        for barrier, observer in zip(controller.barriers, controller.observers):
            _, hdot = barrier.h_and_hdot(controller.x0)
            observer.initialize(hdot, b_hat0=0.0)
    return controller
