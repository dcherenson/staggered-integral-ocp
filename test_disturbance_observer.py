"""Tests for the paper-faithful disturbance-observer controller."""

import numpy as np

from disturbance_observer import (
    DOBConfig,
    DisturbanceObserverController,
    HighGainDisturbanceObserver,
    SafetyBarrier,
    make_controller,
)
from plant import Plant
from run_disturbance_observer import simulate_disturbance_observer
from scenarios.adaptation_off import scenario


def _explicit_config() -> DOBConfig:
    # Six barriers are produced: four obstacles plus two altitude limits.
    return DOBConfig(
        derivative_bounds=np.full(6, 0.1),
        initial_error_bounds=np.full(6, 0.1),
        clf_derivative_bound=0.1,
        clf_initial_error_bound=0.1,
        obstacle_repulsion_gain=0.0,
    )


def test_observer_equation_and_error_bound():
    observer = HighGainDisturbanceObserver(gain=2.0, derivative_bound=1.0, initial_error_bound=2.0)
    observer.initialize(z0=0.0, b_hat0=0.0)
    assert np.isclose(observer.estimate(0.0), 0.0)
    assert np.isclose(observer.error_bound(0.0), 2.0)
    observer.update(z=1.0, a=0.0, dt=0.1)
    assert observer.time == 0.1
    assert observer.error_bound() >= 0.5
    assert observer.error_bound(100.0) >= 0.5


def test_virtual_obstacle_ecbf_terms():
    plant = Plant(spatial_mode=False)
    barrier = SafetyBarrier(
        label="obstacle", kind="obstacle", center=np.array([1.0, -0.5, 1.0]), radius=0.4
    )
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    h, hdot = barrier.h_and_hdot(x)
    hddot, coefficient = barrier.nominal_ecbf_terms(
        x, np.zeros(3), plant, "virtual_acceleration"
    )
    assert np.isclose(h, 1.09)
    assert np.isclose(hdot, 0.0)
    assert np.isclose(hddot, 0.0)
    assert np.allclose(coefficient, np.array([-2.0, 1.0, 0.0]))


def test_both_actuator_interfaces_solve_small_qp():
    for interface in ("virtual_acceleration", "direct_physical_input"):
        plant = Plant(spatial_mode=False)
        controller = make_controller(
            plant=plant,
            obstacles=scenario.obstacles,
            x_goal=scenario.x_goal,
            x0=scenario.x0,
            config=_explicit_config(),
            actuator_interface=interface,
        )
        physical_u, diagnostics = controller.compute_control(scenario.x0, time=0.0)
        assert diagnostics["success"]
        assert np.all(np.isfinite(physical_u))
        assert np.all(diagnostics["barrier_residuals"] >= -1e-6)


def test_short_simulation_logs_observer_data():
    result = simulate_disturbance_observer(
        scenario,
        actuator_interface="virtual_acceleration",
        t_end=0.1,
        dob_config=_explicit_config(),
    )
    assert result["status"] in {"completed", "goal_reached", "collision", "qp_failure"}
    assert result["state"].shape[1] == 8
    assert result["barrier_b_hat"].shape[1] == 6
    assert result["barrier_M"].shape == result["barrier_b_hat"].shape


if __name__ == "__main__":
    test_observer_equation_and_error_bound()
    test_virtual_obstacle_ecbf_terms()
    test_both_actuator_interfaces_solve_small_qp()
    test_short_simulation_logs_observer_data()
    print("All disturbance-observer tests PASSED!")

