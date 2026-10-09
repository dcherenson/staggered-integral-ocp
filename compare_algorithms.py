"""Paired-replay comparison of SIOCP and DOB-DT-MPC."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import FormatStrFormatter

from comparison_simulation import load_scenario, simulate_dob_dtmpc, simulate_siocp
from plant import Plant
from comparison_noise import ReplayGaussianNoise
from comparison_plots import _aligned_predictions, _draw_continuous_tube, save_artifacts
from ssml import SSMLNet, assign_params


# Match the plotting defaults used by the historical SIOCP ``main.py``
# figures (``top_down_tube.png`` and ``si-ocp_vs_true.png``).
_SIOCP_PLOT_RC = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "figure.titlesize": 20,
}


def _write_siocp_update_times(run: dict[str, Any], path: Path) -> None:
    durations = np.asarray(run["siocp_update_time_s"], dtype=float)
    simulation_times = np.asarray(run["siocp_update_simulation_time_s"], dtype=float)
    if durations.shape != simulation_times.shape:
        raise ValueError("SIOCP timing arrays have different lengths")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("update_index", "simulation_time_s", "update_time_s", "included_in_average"))
        writer.writerows((i, ts, duration, i > 1) for i, (ts, duration) in enumerate(zip(simulation_times, durations), 1))


def _plot_trajectories(runs: list[dict[str, Any]], output_dir: Path) -> None:
    cfg = runs[0]["config"]
    fig, axis = plt.subplots(figsize=(9, 6))
    for run in runs:
        axis.plot(run["state"][:, 0], run["state"][:, 1], lw=2, label=run["display_name"])
    for obstacle in cfg.obstacles:
        axis.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.22))
    axis.add_patch(Circle(tuple(cfg.x_goal[:2]), cfg.goal_radius, color="green", alpha=0.15))
    axis.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=170)
    axis.set_xlabel("X position (m)"); axis.set_ylabel("Y position (m)")
    axis.set_title("Paired-replay SIOCP versus DOB-DT-MPC"); axis.grid(True); axis.set_aspect("equal")
    axis.legend(); fig.tight_layout(); fig.savefig(output_dir / "trajectory_overlay.png", dpi=150); plt.close(fig)


def _plot_goal_clearance(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in runs:
        axes[0].plot(run["time"], run["goal_distance"], label=run["display_name"])
        axes[1].plot(run["time"], run["clearance"], label=run["display_name"])
    cfg = runs[0]["config"]
    axes[0].axhline(cfg.goal_radius, color="green", ls="--"); axes[1].axhline(0.0, color="red", ls="--")
    axes[0].set_title("Goal distance"); axes[1].set_title("Obstacle clearance")
    for axis in axes: axis.set_xlabel("Time (s)"); axis.grid(True); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "goal_and_clearance.png", dpi=150); plt.close(fig)


def _plot_bound_diagnostics(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for run in runs:
        axes[0].plot(run["time"], run["disturbance_norm"], "--", label=f"{run['display_name']} residual")
        axes[0].plot(run["time"], run["disturbance_bound"], label=f"{run['display_name']} bound")
    dob = next(run for run in runs if run["algorithm"] == "dob_dtmpc")
    ct = np.asarray(dob["control_time"])
    axes[1].plot(ct, dob["observer_margin"], label="DOB Eq. 37 margin")
    it = np.asarray(dob["time"])[1:1 + len(dob["observed_estimation_error"])]
    axes[1].plot(it, dob["observed_estimation_error"], "--", label="DOB estimation error")
    axes[0].set_title("Disturbance bound passed to DT-MPC"); axes[1].set_title("DOB observer diagnostics")
    for axis in axes: axis.set_xlabel("Time (s)"); axis.set_ylabel("m/s²"); axis.grid(True); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "bound_diagnostics.png", dpi=150); plt.close(fig)


def _scenario_pair(runs: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    pair = [run for run in runs if run["display_name"].endswith(f"({scenario})")]
    if len(pair) != 2:
        raise ValueError(f"expected one SIOCP/DOB pair for {scenario!r}")
    return pair


def _plot_pair_top_down(pair: list[dict[str, Any]], destination: Path, scenario: str) -> None:
    colors = {"siocp": "b", "dob_dtmpc": "#c44e52"}
    tube_colors = {"siocp": "c", "dob_dtmpc": "#f0a3a7"}
    cfg = pair[0]["config"]
    with plt.rc_context(_SIOCP_PLOT_RC):
        # The historical SIOCP figure uses an 8x8 canvas, equal data scaling,
        # and automatic limits; retain those choices so the SVG drops into the
        # existing Figma frame without a rescale.
        fig, axis = plt.subplots(figsize=(8, 8))
        axis.plot([cfg.x0[0], cfg.x_goal[0]], [cfg.x0[1], cfg.x_goal[1]], "k--", alpha=0.5)
        handles = [
            Line2D([0], [0], color="b", lw=2, label="SI-OCP Trajectory"),
            Patch(facecolor="c", edgecolor="none", alpha=0.2, label="SI-OCP Tube"),
            Line2D([0], [0], color=colors["dob_dtmpc"], lw=2, label="DOB-DT-MPC Trajectory"),
            Patch(facecolor=tube_colors["dob_dtmpc"], edgecolor="none", alpha=0.3, label="DOB-DT-MPC Tube"),
        ]
        for run in pair:
            color = colors[run["algorithm"]]
            state = np.asarray(run["state"])
            axis.plot(state[:, 0], state[:, 1], color=color, lw=2)
            z_pred, phi_pred = _aligned_predictions(run)
            stride = max(1, int(round(0.75 / cfg.dt_sim)))
            for index in range(0, len(state), stride):
                _draw_continuous_tube(
                    axis, z_pred[index, :, :2], phi_pred[index],
                    color=tube_colors[run["algorithm"]], alpha=0.2,
                )
                if run["algorithm"] == "siocp":
                    wind = run["plant"].wind_velocity(run["time"][index], state[index, :3]) / 3.0
                    if np.linalg.norm(wind[:2]) > 1e-3:
                        axis.quiver(
                            state[index, 0], state[index, 1], wind[0], wind[1],
                            color="purple", width=0.005, scale=15, alpha=0.7, zorder=5,
                        )
        for obstacle in cfg.obstacles:
            axis.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="r", alpha=0.2))
        axis.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=150)
        axis.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="g", alpha=0.2))
        handles += [
            Patch(facecolor="r", edgecolor="none", alpha=0.2, label="Obstacle"),
            Patch(facecolor="g", edgecolor="none", alpha=0.2, label="Goal Region"),
            Line2D([0], [0], color="purple", marker=r"$\rightarrow$", markersize=15, linestyle="None", label="Wind Vector"),
        ]
        axis.set_xlabel("X Position (m)"); axis.set_ylabel("Y Position (m)")
        axis.legend(
            handles=handles, loc="upper right", ncol=2, fontsize=10,
            borderpad=0.4, labelspacing=0.5, handlelength=2.0, handletextpad=0.8,
        )
        axis.grid(True); axis.set_aspect("equal")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(destination / f"top_down_overlay.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_pair_bound(pair: list[dict[str, Any]], destination: Path, scenario: str) -> None:
    siocp = next(run for run in pair if run["algorithm"] == "siocp")
    dob = next(run for run in pair if run["algorithm"] == "dob_dtmpc")
    cfg = dob["config"]
    # Reproduce the original SIOCP plot's instantaneous true model mismatch
    # separately on each trajectory. Replay lookup is immutable, so these
    # post-simulation diagnostics cannot perturb either controller.
    model = SSMLNet()
    truth = {}
    with torch.no_grad():
        for run in (siocp, dob):
            if not isinstance(run["plant"].noise_source, ReplayGaussianNoise):
                raise ValueError("historical truth diagnostic requires the paired replay")
            values = np.zeros(len(run["time"]), dtype=float)
            for index in range(len(values) - 1):
                state = run["state"][index]
                assign_params(model, torch.tensor(run["theta_history"][index], dtype=torch.float32))
                nn_acc = model(torch.tensor(state[3:8], dtype=torch.float32)).numpy()
                actual_acc = run["plant"].unmodeled_dynamics(
                    run["time"][index], state[:3], state[3:6], state[6:8]
                ) / run["plant"].m
                values[index + 1] = np.linalg.norm(actual_acc - nn_acc)
            truth[run["algorithm"]] = values
    with plt.rc_context(_SIOCP_PLOT_RC):
        # Match the rendered canvas of the top-down overlay so the two SVGs
        # can be stacked directly in Figma.  The tight bounding box around the
        # historical 8x8 top-down figure is approximately 574.5 x 228.1 pt;
        # this wide canvas gives the bound panel that same outer size while
        # retaining its time-series aspect ratio.
        fig, axis = plt.subplots(figsize=(8.227, 3.375))
        sample_index = np.arange(len(siocp["time"]))
        axis.fill_between(sample_index, 0, siocp["disturbance_bound"], color="#aae0fa", alpha=0.8, label="SI-OCP")
        axis.plot(sample_index, siocp["disturbance_bound"], "k-", linewidth=1.5)
        dob_indices = np.asarray(dob["control_time"]) / cfg.dt_sim
        axis.step(
            dob_indices, dob["control_bound"], where="post", color="blue", linewidth=2,
            label=r"DOB-DT-MPC Bound $\bar d_{\mathrm{DOB}}$",
        )
        axis.plot(sample_index, truth["siocp"], color="#D95319", linewidth=1.5, label=r"SI-OCP Truth $\Vert d(t) \Vert$")
        axis.plot(
            np.arange(len(dob["time"])), truth["dob_dtmpc"], color="#c44e52", linestyle="--",
            linewidth=1.7, marker="o", markevery=5, markersize=2.5,
            label=r"DOB-DT-MPC Truth $\Vert d(t) \Vert$",
        )
        axis.set_xlim(0, int(cfg.t_end / cfg.dt_sim)); axis.set_ylim(bottom=0)
        axis.set_yticks(np.arange(0.0, 2.01, 0.5))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_xlabel(f"Time (x {cfg.dt_sim:.2f} s)"); axis.set_ylabel(r"$\Vert d \Vert$")
        axis.legend(
            loc="upper right", fontsize=10,
            borderpad=0.4, labelspacing=0.5, handlelength=2.0, handletextpad=0.8,
        )
        axis.grid(True, linestyle=":", alpha=0.7)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(destination / f"bound_overlay.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_pair_nn(pair: list[dict[str, Any]], destination: Path, scenario: str) -> None:
    colors = {"siocp": "#1f77b4", "dob_dtmpc": "#ff7f0e"}
    fig, axis = plt.subplots(figsize=(10, 6))
    for run in pair:
        axis.plot(run["time"], run["theta_history"], color=colors[run["algorithm"]], alpha=0.1, linewidth=1)
    axis.set_xlabel("Time (s)"); axis.set_ylabel(r"NN Weight Values $\theta$")
    axis.set_title(f"Neural Network Parameters Overlay | {scenario}"); axis.grid(True)
    axis.legend([Line2D([0], [0], color=colors[r["algorithm"]], lw=2) for r in pair], [r["display_name"] for r in pair])
    fig.tight_layout(); fig.savefig(destination / "nn_params_overlay.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_pair_goal_clearance(pair: list[dict[str, Any]], destination: Path, scenario: str) -> None:
    colors = {"siocp": "#1f77b4", "dob_dtmpc": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in pair:
        color = colors[run["algorithm"]]
        axes[0].plot(run["time"], run["goal_distance"], color=color, label=run["display_name"])
        axes[1].plot(run["time"], run["clearance"], color=color, label=run["display_name"])
    axes[0].axhline(pair[0]["config"].goal_radius, color="green", ls="--")
    axes[1].axhline(0.0, color="red", ls="--")
    axes[0].set_title("Goal distance"); axes[1].set_title("Obstacle clearance")
    for axis in axes: axis.set_xlabel("Time (s)"); axis.grid(True); axis.legend(fontsize=8)
    fig.suptitle(f"Goal and Clearance Overlay | {scenario}")
    fig.tight_layout(); fig.savefig(destination / "goal_and_clearance_overlay.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_pair_animation(pair: list[dict[str, Any]], destination: Path, scenario: str) -> None:
    # Keep the original SIOCP animation's layout and visual conventions.
    with plt.rc_context(_SIOCP_PLOT_RC):
        _render_pair_animation(pair, destination)


def _render_pair_animation(pair: list[dict[str, Any]], destination: Path) -> None:
    colors = {"siocp": "b", "dob_dtmpc": "#ff7f0e"}
    tube_colors = {"siocp": "c", "dob_dtmpc": "#ff7f0e"}
    cfg = pair[0]["config"]
    states = [np.asarray(run["state"]) for run in pair]
    predictions = [_aligned_predictions(run) for run in pair]
    siocp_index = next(i for i, run in enumerate(pair) if run["algorithm"] == "siocp")
    siocp = pair[siocp_index]
    horizon = max(len(state) for state in states)
    frame_dt = float(cfg.dt_sim)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot([cfg.x0[0], cfg.x_goal[0]], [cfg.x0[1], cfg.x_goal[1]], "k--", alpha=0.5)
    lines = [axis.plot([], [], color=colors[run["algorithm"]], lw=2)[0] for run in pair]
    prediction_lines = [axis.plot([], [], color=tube_colors[run["algorithm"]], alpha=0.3, lw=1, zorder=2)[0] for run in pair]
    points = [axis.scatter([], [], color=colors[run["algorithm"]], s=50, zorder=5) for run in pair]
    for obstacle in cfg.obstacles:
        axis.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.2))
    axis.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=150, zorder=6)
    axis.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="g", alpha=0.2))
    siocp_state = states[siocp_index]
    axis.set_xlim(np.min(siocp_state[:, 0]) - 0.5, np.max(siocp_state[:, 0]) + 0.5)
    axis.set_ylim(-2.5, 2.5)
    axis.set_aspect("equal"); axis.set_xlabel("X Position (m)"); axis.set_ylabel("Y Position (m)")
    handles = [
        Line2D([0], [0], color="b", lw=2, label="Trajectory"),
        Patch(facecolor="c", edgecolor="none", alpha=0.2, label="Predicted Tube"),
        Patch(facecolor="r", edgecolor="none", alpha=0.2, label="Obstacle"),
        Patch(facecolor="g", edgecolor="none", alpha=0.2, label="Goal Region"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markersize=15, label="Goal"),
        Line2D([0], [0], color="purple", marker=r"$\rightarrow$", markersize=15, linestyle="None", label="Wind Vector"),
        Line2D([0], [0], color=colors["dob_dtmpc"], lw=2, label="DOB Trajectory"),
        Patch(facecolor=tube_colors["dob_dtmpc"], edgecolor="none", alpha=0.2, label="DOB Predicted Tube"),
    ]
    axis.legend(handles=handles, loc="upper right"); axis.grid(True)
    tubes = [None] * len(pair)
    wind_arrow = None

    def update(frame):
        nonlocal wind_arrow
        for index, state in enumerate(states):
            k = min(frame, len(state) - 1)
            lines[index].set_data(state[:k + 1, 0], state[:k + 1, 1])
            points[index].set_offsets(state[k:k + 1, :2])
            z_pred, phi_pred = predictions[index]
            prediction_lines[index].set_data(z_pred[k, :, 0], z_pred[k, :, 1])
            if tubes[index] is not None:
                tubes[index].remove()
            tubes[index] = _draw_continuous_tube(axis, z_pred[k, :, :2], phi_pred[k], color=tube_colors[pair[index]["algorithm"]], alpha=0.2)
        k = min(frame, len(siocp_state) - 1)
        axis.set_title(rf'DTMPC Top-Down | t={frame * frame_dt:.1f}s | $\Phi$={siocp["tube_history"][k]:.2f}m')
        if wind_arrow is not None:
            wind_arrow.remove()
        wind = siocp["plant"].wind_velocity(siocp["time"][k], siocp_state[k, :3]) / 3.0
        wind_arrow = axis.quiver(siocp_state[k, 0], siocp_state[k, 1], wind[0], wind[1], color="purple", width=0.005, scale=15, alpha=0.7, zorder=5)
        return (*lines, *prediction_lines, *points)

    ani = animation.FuncAnimation(fig, update, frames=range(horizon), interval=frame_dt * 1000, blit=False)
    ani.save(destination / "top_down_overlay_animation.gif", writer="pillow", fps=1.0 / frame_dt)
    plt.close(fig)


def _write_pair_visuals(runs: list[dict[str, Any]], scenarios: tuple[str, ...], output_dir: Path) -> None:
    for scenario in scenarios:
        destination = output_dir / scenario
        destination.mkdir(parents=True, exist_ok=True)
        pair = _scenario_pair(runs, scenario)
        _plot_pair_top_down(pair, destination, scenario)
        _plot_pair_bound(pair, destination, scenario)
        _plot_pair_nn(pair, destination, scenario)
        _plot_pair_goal_clearance(pair, destination, scenario)
        _plot_pair_animation(pair, destination, scenario)


def _metrics(run: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(run["state"], dtype=float)
    coverage = run["empirical_coverage"]
    return {
        "algorithm": run["display_name"], "status": run["status"],
        "simulated_time_s": float(run["time"][-1]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1))),
        "minimum_clearance_m": float(np.min(run["clearance"])),
        "final_goal_distance_m": float(run["goal_distance"][-1]),
        "goal_reached": run["status"] == "goal_reached",
        "empirical_coverage": coverage,
        "bound_misses": int(np.count_nonzero(run["bound_miss"])),
        "average_siocp_update_time_s": float(run["average_siocp_update_time_s"]) if run["algorithm"] == "siocp" else None,
        "siocp_update_count": int(run["siocp_update_count"]) if run["algorithm"] == "siocp" else None,
        "siocp_average_update_count": int(run["siocp_average_update_count"]) if run["algorithm"] == "siocp" else None,
        "average_dtmpc_update_time_s": float(run["average_dtmpc_update_time_s"]) if run["average_dtmpc_update_time_s"] is not None else None,
        "noise_id": run["noise_id"], "formal_safety_guarantee": run["formal_safety_guarantee"],
        "observer_gain": run["observer_gain"],
        "initial_mpc_bound": float(run["control_bound"][0]) if len(run["control_bound"]) else None,
        "observer_initial_error_bound": run["observer_initial_error_bound"],
        "initial_bound_policy": run["initial_bound_policy"],
        "failure_message": run["failure_message"],
    }


def run_comparison(
    scenarios: tuple[str, ...] = ("adaptation_off", "adaptation_on"),
    output_dir: str | Path = "output/comparison_dob_dtmpc",
    t_end: float | None = None,
    algorithm: str = "both",
) -> list[dict[str, Any]]:
    if algorithm not in {"both", "siocp", "dob"}:
        raise ValueError("algorithm must be 'both', 'siocp', or 'dob'")
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    selected = ("siocp", "dob") if algorithm == "both" else (algorithm,)
    runs = []
    for name in scenarios:
        cfg = load_scenario(name)
        if t_end is not None: cfg = replace(cfg, t_end=float(t_end))
        replay = ReplayGaussianNoise(cfg.seed, cfg.dt_sim, cfg.t_end + cfg.dt_sim)
        for method in selected:
            plant = Plant(spatial_mode=cfg.spatial_wind, noise_source=replay)
            simulate = simulate_siocp if method == "siocp" else simulate_dob_dtmpc
            run = simulate(cfg, plant=plant)
            label = "SIOCP" if method == "siocp" else "DOB-DT-MPC"
            run["display_name"] = f"{label} ({name})"
            runs.append(run)
            if method == "siocp":
                _write_siocp_update_times(run, output_path / f"siocp_{name}_update_times.csv")
            if algorithm != "both" or method == "dob":
                save_artifacts(run, output_path / run["algorithm"] / name)
    replay_ids = {run["noise_id"] for run in runs}
    if len(replay_ids) != 1:
        raise RuntimeError(f"comparison scenarios do not share a replay identifier: {replay_ids}")
    if algorithm == "both":
        _plot_trajectories(runs, output_path); _plot_goal_clearance(runs, output_path); _plot_bound_diagnostics(runs, output_path)
        _write_pair_visuals(runs, scenarios, output_path)
    rows = [_metrics(run) for run in runs]
    with (output_path / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return runs


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare SIOCP and DOB-DT-MPC")
    parser.add_argument("--scenario", choices=("adaptation_off", "adaptation_on", "both"), default="both")
    parser.add_argument("--algorithm", choices=("both", "siocp", "dob"), default="both")
    parser.add_argument("--output-dir", default="output/comparison_dob_dtmpc")
    parser.add_argument("--t-end", type=float, default=None)
    args = parser.parse_args(argv)
    names = ("adaptation_off", "adaptation_on") if args.scenario == "both" else (args.scenario,)
    runs = run_comparison(names, args.output_dir, args.t_end, args.algorithm)
    for run in runs:
        row = _metrics(run)
        print(f"{row['algorithm']}: {row['status']} | clearance={row['minimum_clearance_m']:.3f}m | goal distance={row['final_goal_distance_m']:.3f}m")
        if row["average_siocp_update_time_s"] is not None:
            print(
                f"  Average SIOCP DT-MPC update: {row['average_siocp_update_time_s']:.6f}s "
                f"over {row['siocp_average_update_count']} updates (first update excluded)"
            )
    return runs


if __name__ == "__main__":
    main()
