"""Plotting and artifact helpers for SIOCP and disturbance-observer runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon



def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _aligned_predictions(result: dict[str, Any]):
    """Align controller predictions with state samples for plotting/animation."""
    state = np.asarray(result["state"], dtype=float)
    horizon = int(result["config"].mpc_horizon)
    z = np.asarray(result.get("z_pred_history", []), dtype=float)
    phi = np.asarray(result.get("phi_pred_history", []), dtype=float)
    if z.ndim != 3 or z.shape[1:] != (horizon, 8):
        z = np.empty((0, horizon, 8))
    if phi.ndim != 2 or phi.shape[1] != horizon:
        phi = np.empty((0, horizon))
    # Prediction i is computed at state sample i, before the following plant
    # transition.  Only a terminal sample (or a failed initial solve) needs a
    # plotting fallback; prepending one would shift every sampled tube by dt.
    if len(z) == 0:
        tube_history = np.asarray(result.get("tube_history", [0.05]), dtype=float)
        z = np.tile(state[0], (horizon, 1))[None, ...]
        phi = np.full((1, horizon), float(tube_history[0]) if tube_history.size else 0.05)
    if len(z) < len(state):
        z = np.concatenate((z, np.repeat(z[-1:, ...], len(state) - len(z), axis=0)), axis=0)
        phi = np.concatenate((phi, np.repeat(phi[-1:, ...], len(state) - len(phi), axis=0)), axis=0)
    return z[:len(state)], phi[:len(state)]


def _draw_continuous_tube(axis, path_xy, radii, color="c", alpha=0.2):
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return None
    radii = np.full(len(path_xy), float(radii)) if np.isscalar(radii) else np.asarray(radii, dtype=float)
    tangents = np.zeros_like(path_xy)
    tangents[0] = path_xy[1] - path_xy[0]
    tangents[-1] = path_xy[-1] - path_xy[-2]
    if len(path_xy) > 2:
        tangents[1:-1] = path_xy[2:] - path_xy[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    left = path_xy + normals * radii[:, None]
    right = path_xy - normals * radii[:, None]
    end_angle = np.arctan2(tangents[-1, 1], tangents[-1, 0])
    angles = np.linspace(end_angle + np.pi / 2, end_angle - np.pi / 2, 15)
    center, radius = path_xy[-1], radii[-1]
    cap = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    points = np.vstack((left[:-1], cap, right[-2::-1]))
    return axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, alpha=alpha, zorder=2))


def _plot_top_down(result: dict[str, Any], destination: Path, *, filename="top_down_tube.png", title="DOB-DT-MPC Top-Down"):
    cfg = result["config"]
    state = np.asarray(result["state"])
    time_values = np.asarray(result["time"])
    z_pred, phi_pred = _aligned_predictions(result)
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.plot([cfg.x0[0], cfg.x_goal[0]], [cfg.x0[1], cfg.x_goal[1]], "k--", alpha=0.5)
    axis.plot(state[:, 0], state[:, 1], "b-", linewidth=2)
    for obstacle in cfg.obstacles:
        axis.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="r", alpha=0.2))
    axis.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=150)
    axis.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="g", alpha=0.2))
    stride = max(1, int(round(0.75 / cfg.dt_sim)))
    for index in range(0, len(time_values), stride):
        _draw_continuous_tube(axis, z_pred[index, :, :2], phi_pred[index], color="c", alpha=0.2)
        wind = result["plant"].wind_velocity(time_values[index], state[index, :3]) / 3.0
        if np.linalg.norm(wind[:2]) > 1e-3:
            axis.quiver(state[index, 0], state[index, 1], wind[0], wind[1], color="purple", width=0.005, scale=15, alpha=0.7, zorder=5)
    handles = [
        Line2D([0], [0], color="b", lw=2, label="Trajectory"),
        Patch(facecolor="c", edgecolor="none", alpha=0.2, label="Predicted Tube"),
        Patch(facecolor="r", edgecolor="none", alpha=0.2, label="Obstacle"),
        Patch(facecolor="g", edgecolor="none", alpha=0.2, label="Goal Region"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markersize=15, label="Goal"),
        Line2D([0], [0], color="purple", marker=r"$\rightarrow$", markersize=15, linestyle="None", label="Wind Vector"),
    ]
    axis.set_xlabel("X Position (m)"); axis.set_ylabel("Y Position (m)"); axis.set_title(title)
    axis.legend(handles=handles, loc="upper right"); axis.grid(True); axis.set_aspect("equal")
    fig.tight_layout(); fig.savefig(destination / filename, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_animation(result: dict[str, Any], destination: Path, *, filename="top_down_animation.gif", title="DOB-DT-MPC Top-Down"):
    cfg = result["config"]
    state = np.asarray(result["state"])
    time_values = np.asarray(result["time"])
    z_pred, phi_pred = _aligned_predictions(result)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot([cfg.x0[0], cfg.x_goal[0]], [cfg.x0[1], cfg.x_goal[1]], "k--", alpha=0.5, label="Reference")
    line, = axis.plot([], [], "b-", linewidth=2, label="Quadcopter Path")
    point = axis.scatter([], [], color="blue", s=50, zorder=5)
    prediction, = axis.plot([], [], "c-", alpha=0.5, linewidth=1, zorder=2, label="Prediction")
    for obstacle in cfg.obstacles:
        axis.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.2))
    axis.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=150, zorder=6)
    axis.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="g", alpha=0.2))
    x_lo, x_hi = np.min(state[:, 0]) - 0.5, np.max(state[:, 0]) + 0.5
    y_lo, y_hi = np.min(state[:, 1]) - 0.5, np.max(state[:, 1]) + 0.5
    axis.set_xlim(x_lo, x_hi); axis.set_ylim(y_lo, y_hi); axis.set_aspect("equal")
    axis.set_xlabel("X Position (m)"); axis.set_ylabel("Y Position (m)"); axis.grid(True)
    current_tube = None

    def update(index):
        nonlocal current_tube
        line.set_data(state[:index + 1, 0], state[:index + 1, 1])
        point.set_offsets(state[index:index + 1, :2])
        prediction.set_data(z_pred[index, :, 0], z_pred[index, :, 1])
        if current_tube is not None:
            current_tube.remove()
        current_tube = _draw_continuous_tube(axis, z_pred[index, :, :2], phi_pred[index], color="c", alpha=0.2)
        axis.set_title(f"{title} | t={time_values[index]:.1f}s | $\\Phi$={result['tube_history'][min(index, len(result['tube_history']) - 1)]:.2f}m")
        return line, point, prediction

    ani = animation.FuncAnimation(fig, update, frames=range(len(state)), interval=cfg.dt_sim * 1000, blit=False)
    ani.save(destination / filename, writer="pillow", fps=1.0 / cfg.dt_sim)
    plt.close(fig)


def _plot_bound(result: dict[str, Any], destination: Path):
    cfg = result["config"]
    indices = np.arange(len(result["time"]))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.fill_between(indices, 0, result["disturbance_bound"], color="#aae0fa", alpha=0.8, label=f"{result['algorithm']} Bound")
    axis.plot(indices, result["disturbance_bound"], "k-", linewidth=1.5)
    axis.plot(indices, result["disturbance_norm"], color="#D95319", linewidth=1.5, label="Transition acceleration residual")
    axis.set_xlim(0, len(indices) - 1); axis.set_ylim(bottom=0)
    axis.set_xlabel(f"Time (x {cfg.dt_sim:.2f} s)"); axis.set_ylabel(r"$\Vert d \Vert$")
    axis.legend(loc="upper right"); axis.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout(); fig.savefig(destination / f"{result['algorithm']}_vs_true.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_nn(result: dict[str, Any], destination: Path):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(result["time"], result["theta_history"], alpha=0.1, linewidth=1)
    axis.set_xlabel("Time (s)"); axis.set_ylabel(r"NN Weight Values $\theta$")
    axis.set_title("Neural Network Parameters Evolution"); axis.grid(True)
    fig.tight_layout(); fig.savefig(destination / "nn_params_vs_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def save_artifacts(result: dict[str, Any], output_dir: str | Path = "output/dob_dtmpc") -> Path:
    """Save data, plots, and animation for one controller run."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(destination / "run_data.npz", **arrays)
    with (destination / "dtmpc_update_times.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("update_index", "simulation_time_s", "update_time_s", "included_in_average"))
        writer.writerows((i, ts, duration, i > 1) for i, (ts, duration) in enumerate(zip(result["dtmpc_update_simulation_time_s"], result["dtmpc_update_time_s"]), 1))
    _plot_bound(result, destination)
    _plot_nn(result, destination)
    label = "SIOCP" if result["algorithm"] == "siocp" else "DOB-DT-MPC"
    _plot_top_down(result, destination, title=f"{label} Top-Down")
    _plot_animation(result, destination, title=f"{label} Top-Down")
    # Retain the compact diagnostics as a useful complement to the historical style.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result["time"], result["goal_distance"]); axes[0].axhline(result["config"].goal_radius, ls="--", color="green")
    axes[1].plot(result["time"], result["clearance"]); axes[1].axhline(0.0, ls="--", color="red")
    axes[0].set_title("Goal distance"); axes[1].set_title("Obstacle clearance")
    for axis in axes: axis.set_xlabel("Time (s)"); axis.grid(True)
    fig.tight_layout(); fig.savefig(destination / "goal_and_clearance.png", dpi=150); plt.close(fig)
    metrics = {
        "algorithm": result["algorithm"], "scenario": result["scenario"], "status": result["status"],
        "simulated_time_s": float(result["time"][-1]), "minimum_clearance_m": float(np.min(result["clearance"])),
        "final_goal_distance_m": float(result["goal_distance"][-1]),
        "empirical_coverage": _json_value(result["empirical_coverage"]),
        "bound_misses": int(np.count_nonzero(result["bound_miss"])),
        "replay_id": result["replay_id"], "seed": result["seed"], "formal_safety_guarantee": result["formal_safety_guarantee"],
        "observer_gain": result["observer_gain"],
        "initial_mpc_bound": float(result["control_bound"][0]) if len(result["control_bound"]) else None,
        "observer_initial_error_bound": result["observer_initial_error_bound"],
        "initial_bound_policy": result["initial_bound_policy"],
        "average_dtmpc_update_time_s": _json_value(result["average_dtmpc_update_time_s"]),
        "failure_message": result["failure_message"],
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_value))
    return destination
