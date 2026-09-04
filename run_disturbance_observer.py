"""Run and visualize the disturbance-observer quadcopter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from disturbance_observer import (
    ACTUATOR_INTERFACE,
    DOBConfig,
    DisturbanceObserverQPError,
    make_controller,
)
from main import load_scenario
from plant import Plant


def _stack(values: list[np.ndarray], width: int = 0) -> np.ndarray:
    if values:
        return np.asarray(values)
    return np.empty((0, width), dtype=float)


def _clearance(x_history: np.ndarray, obstacles: list[dict[str, Any]]) -> np.ndarray:
    if not obstacles:
        return np.full(len(x_history), np.inf)
    return np.min(
        np.stack([
            np.linalg.norm(x_history[:, :3] - obstacle["pos"], axis=1) - obstacle["r"]
            for obstacle in obstacles
        ],
        axis=1,
    ),
    axis=1,
    )


def simulate_disturbance_observer(
    cfg: Any,
    actuator_interface: Optional[str] = None,
    t_end: Optional[float] = None,
    dob_config: Optional[DOBConfig] = None,
) -> dict[str, Any]:
    """Simulate the unchanged Plant under the DOB robust CLF-ECBF controller."""
    np.random.seed(cfg.seed)
    plant = Plant(spatial_mode=cfg.spatial_wind)
    interface = actuator_interface or ACTUATOR_INTERFACE
    controller = make_controller(
        plant=plant,
        obstacles=cfg.obstacles,
        x_goal=cfg.x_goal,
        x0=cfg.x0,
        z_min=0.8,
        z_max=1.2,
        config=dob_config,
        actuator_interface=interface,
    )

    dt = float(cfg.dt_sim)
    horizon = float(cfg.t_end if t_end is None else t_end)
    t = 0.0
    x = np.asarray(cfg.x0, dtype=float).copy()

    t_steps: list[float] = []
    x_steps: list[np.ndarray] = [x.copy()]
    physical_controls: list[np.ndarray] = []
    qp_controls: list[np.ndarray] = []
    baseline_controls: list[np.ndarray] = []
    barrier_h: list[np.ndarray] = []
    barrier_hdot: list[np.ndarray] = []
    barrier_b_hat: list[np.ndarray] = []
    barrier_M: list[np.ndarray] = []
    barrier_b_true: list[np.ndarray] = []
    barrier_residuals: list[np.ndarray] = []
    clf_values: list[float] = []
    clf_slack: list[float] = []
    clf_b_hat: list[float] = []
    clf_M: list[float] = []
    clf_b_true: list[float] = []
    wind_history: list[np.ndarray] = []
    solver_messages: list[str] = []
    status = "completed"
    failure_message = ""

    while t < horizon - 1e-12:
        x_old = x.copy()
        try:
            physical_u, diagnostics = controller.compute_control(x_old, time=t)
        except DisturbanceObserverQPError as exc:
            status = "qp_failure"
            failure_message = str(exc)
            break

        qp_u = diagnostics["qp_input"]
        x = plant.step(x_old, physical_u, t, dt)
        controller.update_observers(x_old, qp_u, dt, measurement_state=x)

        # This sampled value is for diagnostics only.  The current Plant
        # generates fresh Gaussian noise inside every ODE evaluation, so it is
        # not claimed to be the exact force used by every RK45 substep.
        d_acc_sample = plant.unmodeled_dynamics(
            t, x_old[:3], x_old[3:6], x_old[6:8]
        ) / plant.m
        snapshot = controller.observer_snapshot(x_old, d_acc=d_acc_sample)

        t_steps.append(t)
        x_steps.append(x.copy())
        physical_controls.append(np.asarray(physical_u, dtype=float))
        qp_controls.append(np.asarray(qp_u, dtype=float))
        baseline_controls.append(np.asarray(diagnostics["baseline_input"], dtype=float))
        barrier_h.append(np.asarray([item["h"] for item in diagnostics["barriers"]]))
        barrier_hdot.append(np.asarray([item["hdot"] for item in diagnostics["barriers"]]))
        barrier_b_hat.append(snapshot["barrier_b_hat"])
        barrier_M.append(snapshot["barrier_M"])
        barrier_b_true.append(snapshot["barrier_b_true"])
        barrier_residuals.append(diagnostics["barrier_residuals"])
        clf_values.append(float(diagnostics["clf_V"]))
        clf_slack.append(float(diagnostics["delta"]))
        clf_b_hat.append(float(snapshot["clf_b_hat"]))
        clf_M.append(float(snapshot["clf_M"]))
        clf_b_true.append(float(snapshot["clf_b_true"]))
        wind_history.append(plant.wind_velocity(t, x_old[:3]))
        solver_messages.append(diagnostics["solver_message"])

        t += dt

        collision = any(
            np.linalg.norm(x[:3] - obstacle["pos"]) < obstacle["r"]
            for obstacle in cfg.obstacles
        )
        if collision:
            status = "collision"
            break
        if np.linalg.norm(x[:3] - cfg.x_goal[:3]) < cfg.goal_radius:
            status = "goal_reached"
            break

    x_history = np.asarray(x_steps)
    t_history = np.concatenate([np.asarray(t_steps), [t]])
    clearances = _clearance(x_history, cfg.obstacles)
    goal_distances = np.linalg.norm(x_history[:, :3] - cfg.x_goal[:3], axis=1)
    min_clearance = float(np.min(clearances)) if len(clearances) else np.inf
    result = {
        "algorithm": "disturbance_observer",
        "interface": interface,
        "scenario": cfg.name,
        "status": status,
        "failure_message": failure_message,
        "time": t_history,
        "state": x_history,
        "physical_control": _stack(physical_controls, 3),
        "qp_control": _stack(qp_controls, 3),
        "baseline_control": _stack(baseline_controls, 3),
        "barrier_h": _stack(barrier_h, len(controller.barriers)),
        "barrier_hdot": _stack(barrier_hdot, len(controller.barriers)),
        "barrier_b_hat": _stack(barrier_b_hat, len(controller.barriers)),
        "barrier_M": _stack(barrier_M, len(controller.barriers)),
        "barrier_b_true": _stack(barrier_b_true, len(controller.barriers)),
        "barrier_residuals": _stack(barrier_residuals, len(controller.barriers)),
        "clf_V": np.asarray(clf_values),
        "clf_slack": np.asarray(clf_slack),
        "clf_b_hat": np.asarray(clf_b_hat),
        "clf_M": np.asarray(clf_M),
        "clf_b_true": np.asarray(clf_b_true),
        "wind": _stack(wind_history, 3),
        "clearance": clearances,
        "goal_distance": goal_distances,
        "barrier_labels": np.asarray([barrier.label for barrier in controller.barriers]),
        "observer_derivative_bounds": controller.derivative_bounds.copy(),
        "observer_initial_error_bounds": controller.initial_error_bounds.copy(),
        "clf_derivative_bound": controller.clf_derivative_bound,
        "clf_initial_error_bound": controller.clf_initial_error_bound,
        "bound_source": controller.bound_source,
        "solver_messages": np.asarray(solver_messages, dtype=str),
        "controller": controller,
        "plant": plant,
        "config": cfg,
    }
    return result


def _save_npz(result: dict[str, Any], path: Path) -> None:
    skip = {"controller", "plant", "config"}
    arrays = {
        key: value
        for key, value in result.items()
        if key not in skip and isinstance(value, (np.ndarray, float, int, str))
    }
    np.savez_compressed(path, **arrays)


def _plot_top_down(result: dict[str, Any], output_dir: Path) -> None:
    cfg = result["config"]
    state = result["state"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(state[:, 0], state[:, 1], color="tab:blue", lw=2, label="DOB trajectory")
    ax.scatter(state[0, 0], state[0, 1], color="black", s=50, label="Start", zorder=5)
    ax.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=180, label="Goal", zorder=6)
    ax.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="green", alpha=0.15))
    for obstacle in cfg.obstacles:
        ax.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.25))
    for i in range(0, len(result["wind"]), max(1, len(result["wind"]) // 12)):
        if i >= len(state) - 1:
            break
        wind = result["wind"][i][:2] / 3.0
        ax.quiver(state[i, 0], state[i, 1], wind[0], wind[1], color="purple", alpha=0.65, scale=15)
    handles = [
        Line2D([0], [0], color="tab:blue", lw=2, label="DOB trajectory"),
        Patch(facecolor="red", edgecolor="none", alpha=0.25, label="Obstacle"),
        Patch(facecolor="green", edgecolor="none", alpha=0.15, label="Goal region"),
        Line2D([0], [0], color="purple", marker=r"$\rightarrow$", linestyle="None", markersize=12, label="Wind"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(f"Disturbance-observer trajectory ({result['interface']})")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "top_down.png", dpi=150)
    plt.close(fig)


def _plot_observer_effects(result: dict[str, Any], output_dir: Path) -> None:
    times = result["time"][:-1]
    labels = result["barrier_labels"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, label in enumerate(labels):
        if len(times) == 0:
            continue
        axes[0].plot(times, result["barrier_b_true"][:, i], lw=1.2, label=f"true {label}")
        axes[0].plot(times, result["barrier_b_hat"][:, i], "--", lw=1.2, label=f"estimate {label}")
        error = np.abs(result["barrier_b_true"][:, i] - result["barrier_b_hat"][:, i])
        axes[1].plot(times, error, lw=1.2, label=f"|error| {label}")
        axes[1].plot(times, result["barrier_M"][:, i], "k--", lw=0.9, alpha=0.7)
    axes[0].set_ylabel(r"$b_e$")
    axes[0].set_title("Disturbance effect on each barrier derivative")
    axes[1].set_ylabel(r"$|b_e-\hat b_e|$ and $M_b$")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "observer_effects_and_bounds.png", dpi=150)
    plt.close(fig)


def _plot_behavior(result: dict[str, Any], output_dir: Path) -> None:
    cfg = result["config"]
    times = result["time"]
    step_times = times[:-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    axes[0, 0].plot(times, result["goal_distance"], color="tab:blue")
    axes[0, 0].axhline(cfg.goal_radius, color="tab:green", ls="--", label="goal radius")
    axes[0, 0].set_ylabel("Goal distance (m)")
    axes[0, 0].legend()
    axes[0, 1].plot(times, result["clearance"], color="tab:orange")
    axes[0, 1].axhline(0.0, color="red", ls="--", label="collision boundary")
    axes[0, 1].set_ylabel("Minimum clearance (m)")
    axes[0, 1].legend()
    axes[1, 0].plot(step_times, result["clf_V"], label="V")
    axes[1, 0].plot(step_times, result["clf_slack"], label=r"$\delta$")
    axes[1, 0].set_ylabel("CLF value / slack")
    axes[1, 0].set_xlabel("Time (s)")
    controls = result["physical_control"]
    if len(controls):
        for i, label in enumerate((r"$\dot\phi$", r"$\dot\theta$", "T")):
            axes[1, 1].plot(step_times, controls[:, i], label=label)
    axes[1, 1].set_ylabel("Physical control")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()
    for row in axes:
        for ax in row:
            ax.grid(True, linestyle=":", alpha=0.7)
    fig.suptitle(f"DOB safety and goal metrics ({result['status']})")
    fig.tight_layout()
    fig.savefig(output_dir / "goal_clearance_clf_control.png", dpi=150)
    plt.close(fig)


def _save_animation(result: dict[str, Any], output_dir: Path) -> None:
    cfg = result["config"]
    state = result["state"]
    times = result["time"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for obstacle in cfg.obstacles:
        ax.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.25))
    ax.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="green", alpha=0.15))
    ax.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=180, zorder=5)
    path, = ax.plot([], [], color="tab:blue", lw=2, label="DOB trajectory")
    point, = ax.plot([], [], "o", color="tab:blue", ms=7)
    wind_arrow = None

    x_margin = 0.6
    ax.set_xlim(float(np.min(state[:, 0]) - x_margin), float(np.max(state[:, 0]) + x_margin))
    ax.set_ylim(float(np.min(state[:, 1]) - 1.0), float(np.max(state[:, 1]) + 1.0))
    ax.set_aspect("equal")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")

    def init():
        path.set_data([], [])
        point.set_data([], [])
        return (path, point)

    def update(frame: int):
        nonlocal wind_arrow
        path.set_data(state[: frame + 1, 0], state[: frame + 1, 1])
        point.set_data([state[frame, 0]], [state[frame, 1]])
        if wind_arrow is not None:
            wind_arrow.remove()
        if frame < len(result["wind"]):
            wind = result["wind"][frame][:2] / 3.0
        else:
            wind = np.zeros(2)
        wind_arrow = ax.quiver(
            state[frame, 0], state[frame, 1], wind[0], wind[1],
            color="purple", scale=15, width=0.005, alpha=0.7,
        )
        ax.set_title(
            f"DOB top-down | t={times[frame]:.2f}s | clearance={result['clearance'][frame]:.2f}m"
        )
        return (path, point)

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(state), interval=max(20, 1000 * float(cfg.dt_sim)), blit=False
    )
    ani.save(output_dir / "top_down_animation.gif", writer="pillow", fps=max(1, int(round(1.0 / cfg.dt_sim))))
    plt.close(fig)


def save_artifacts(result: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_npz(result, output_path / "run_data.npz")
    _plot_top_down(result, output_path)
    _plot_observer_effects(result, output_path)
    _plot_behavior(result, output_path)
    if len(result["state"]) > 1:
        _save_animation(result, output_path)
    metrics = {
        "status": result["status"],
        "interface": result["interface"],
        "simulated_time": float(result["time"][-1]),
        "minimum_clearance": float(np.min(result["clearance"])),
        "final_goal_distance": float(result["goal_distance"][-1]),
        "goal_reached": result["status"] == "goal_reached",
        "bound_source": result["bound_source"],
        "observer_derivative_bounds": result["observer_derivative_bounds"].tolist(),
        "observer_initial_error_bounds": result["observer_initial_error_bounds"].tolist(),
        "clf_derivative_bound": float(result["clf_derivative_bound"]),
        "clf_initial_error_bound": float(result["clf_initial_error_bound"]),
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Disturbance-observer quadcopter simulation")
    parser.add_argument("--scenario", default="adaptation_off")
    parser.add_argument("--interface", choices=sorted({"virtual_acceleration", "direct_physical_input"}), default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--output-dir", default="output/disturbance_observer")
    parser.add_argument("--no-artifacts", action="store_true", help="Run simulation without writing plots/GIF")
    args = parser.parse_args()
    cfg = load_scenario(args.scenario)
    result = simulate_disturbance_observer(cfg, actuator_interface=args.interface, t_end=args.t_end)
    print(f"DOB scenario: {cfg.name}")
    print(f"Interface: {result['interface']} | status: {result['status']}")
    print(f"Simulated time: {result['time'][-1]:.2f}s")
    print(f"Minimum obstacle clearance: {np.min(result['clearance']):.3f}m")
    print(f"Final goal distance: {result['goal_distance'][-1]:.3f}m")
    if result["failure_message"]:
        print(result["failure_message"])
    if not args.no_artifacts:
        output_path = save_artifacts(result, args.output_dir)
        print(f"Artifacts saved to {output_path}")


if __name__ == "__main__":
    main()
