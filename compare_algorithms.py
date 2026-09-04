"""Compare the existing SIOCP runner with the new DOB controller."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import main as existing_siocp
from main import load_scenario
from run_disturbance_observer import save_artifacts, simulate_disturbance_observer


def capture_existing_siocp(scenario_name: str) -> dict[str, Any]:
    """Run the unchanged main.main and capture its local result arrays.

    This wrapper intentionally executes the existing SIOCP code itself rather
    than copying its simulation loop.  The comparison therefore follows any
    current SIOCP behavior without editing that implementation.
    """
    captured: dict[str, Any] = {}
    target_code = existing_siocp.main.__code__

    def local_trace(frame, event, arg):
        if event == "return":
            for name in (
                "cfg", "x_history", "t_history", "dist_bound_history",
                "disturbance_norm_history", "x_goal", "goal_radius", "sys_plant",
            ):
                if name in frame.f_locals:
                    captured[name] = frame.f_locals[name]
        return local_trace

    def global_trace(frame, event, arg):
        if frame.f_code is target_code:
            return local_trace
        return None

    old_argv = sys.argv.copy()
    sys.argv = ["main.py", "--scenario", scenario_name]
    sys.settrace(global_trace)
    try:
        existing_siocp.main()
    finally:
        sys.settrace(None)
        sys.argv = old_argv
    if "x_history" not in captured:
        raise RuntimeError("Could not capture the existing SIOCP result arrays")
    return captured


def _normalise_siocp(captured: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    cfg = captured["cfg"]
    state = np.asarray(captured["x_history"])
    time = np.asarray(captured["t_history"])
    clearance = np.min(
        np.stack([
            np.linalg.norm(state[:, :3] - obstacle["pos"], axis=1) - obstacle["r"]
            for obstacle in cfg.obstacles
        ], axis=1),
        axis=1,
    )
    goal_distance = np.linalg.norm(state[:, :3] - cfg.x_goal[:3], axis=1)
    if float(np.min(clearance)) < 0.0:
        status = "collision"
    elif float(np.min(goal_distance)) < cfg.goal_radius:
        status = "goal_reached"
    elif float(time[-1]) < float(cfg.t_end) - float(cfg.dt_sim) * 0.5:
        status = "early_stop"
    else:
        status = "completed"
    return {
        "algorithm": f"SIOCP ({scenario_name})",
        "scenario": cfg.name,
        "status": status,
        "time": time,
        "state": state,
        "clearance": clearance,
        "goal_distance": goal_distance,
        "disturbance_bound": np.asarray(captured.get("dist_bound_history", [])),
        "disturbance_norm": np.asarray(captured.get("disturbance_norm_history", [])),
        "config": cfg,
    }


def _plot_trajectories(runs: list[dict[str, Any]], output_dir: Path) -> None:
    cfg = runs[0]["config"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for color, run in zip(colors, runs):
        state = run["state"]
        ax.plot(state[:, 0], state[:, 1], color=color, lw=2, label=run["algorithm"])
    for obstacle in cfg.obstacles:
        ax.add_patch(Circle(tuple(obstacle["pos"][:2]), obstacle["r"], color="red", alpha=0.22))
    ax.add_patch(Circle((cfg.x_goal[0], cfg.x_goal[1]), cfg.goal_radius, color="green", alpha=0.15))
    ax.scatter(cfg.x_goal[0], cfg.x_goal[1], color="gold", marker="*", s=190, zorder=5)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("SIOCP versus disturbance-observer trajectories")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_overlay.png", dpi=150)
    plt.close(fig)


def _plot_goal_clearance(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in runs:
        axes[0].plot(run["time"], run["goal_distance"], lw=1.8, label=run["algorithm"])
        axes[1].plot(run["time"], run["clearance"], lw=1.8, label=run["algorithm"])
    cfg = runs[0]["config"]
    axes[0].axhline(cfg.goal_radius, color="green", ls="--", label="goal radius")
    axes[1].axhline(0.0, color="red", ls="--", label="collision boundary")
    axes[0].set_ylabel("Goal distance (m)")
    axes[1].set_ylabel("Minimum obstacle clearance (m)")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Goal convergence and obstacle clearance")
    fig.tight_layout()
    fig.savefig(output_dir / "goal_and_clearance.png", dpi=150)
    plt.close(fig)


def _plot_disturbance_margins(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run in runs:
        if run["algorithm"].startswith("SIOCP"):
            if len(run["disturbance_bound"]):
                ax.plot(run["time"][: len(run["disturbance_bound"])], run["disturbance_bound"], lw=1.6, label=f"{run['algorithm']} bound")
            if len(run["disturbance_norm"]):
                ax.plot(run["time"][: len(run["disturbance_norm"])], run["disturbance_norm"], ls="--", lw=1.1, label=f"{run['algorithm']} sampled true norm")
        else:
            if len(run["barrier_M"]):
                ax.plot(run["time"][:-1], np.max(run["barrier_M"], axis=1), lw=1.8, label="DOB max observer margin")
            if len(run["barrier_b_true"]):
                ax.plot(run["time"][:-1], np.max(np.abs(run["barrier_b_true"]), axis=1), ls="--", lw=1.1, label="DOB max sampled effect")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Disturbance bound / effect")
    ax.set_title("Disturbance-bound comparison")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "disturbance_bounds.png", dpi=150)
    plt.close(fig)


def _metrics(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": run["algorithm"],
        "status": run["status"],
        "simulated_time_s": float(run["time"][-1]),
        "minimum_clearance_m": float(np.min(run["clearance"])),
        "final_goal_distance_m": float(run["goal_distance"][-1]),
        "goal_reached": bool(run["status"] == "goal_reached"),
    }


def run_comparison(
    dob_scenario_name: str = "adaptation_off",
    siocp_scenarios: tuple[str, ...] = ("adaptation_off", "adaptation_on"),
    actuator_interface: str | None = None,
    output_dir: str | Path = "output/comparison",
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for scenario_name in siocp_scenarios:
        captured = capture_existing_siocp(scenario_name)
        runs.append(_normalise_siocp(captured, scenario_name))

    dob_cfg = load_scenario(dob_scenario_name)
    dob_run = simulate_disturbance_observer(
        dob_cfg,
        actuator_interface=actuator_interface,
    )
    # Save the full DOB artifacts under the comparison directory as well as
    # the standalone runner's normal output when requested separately.
    save_artifacts(dob_run, output_path / "disturbance_observer")
    runs.append(dob_run)

    _plot_trajectories(runs, output_path)
    _plot_goal_clearance(runs, output_path)
    _plot_disturbance_margins(runs, output_path)
    metric_rows = [_metrics(run) for run in runs]
    with (output_path / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing SIOCP and DOB controllers")
    parser.add_argument("--dob-scenario", default="adaptation_off")
    parser.add_argument(
        "--siocp-scenario",
        choices=("adaptation_off", "adaptation_on", "both"),
        default="both",
    )
    parser.add_argument("--interface", choices=("virtual_acceleration", "direct_physical_input"), default=None)
    parser.add_argument("--output-dir", default="output/comparison")
    args = parser.parse_args()
    if args.siocp_scenario == "both":
        siocp_scenarios = ("adaptation_off", "adaptation_on")
    else:
        siocp_scenarios = (args.siocp_scenario,)
    runs = run_comparison(
        dob_scenario_name=args.dob_scenario,
        siocp_scenarios=siocp_scenarios,
        actuator_interface=args.interface,
        output_dir=args.output_dir,
    )
    print(f"Comparison artifacts saved to {args.output_dir}")
    for run in runs:
        row = _metrics(run)
        print(
            f"{row['algorithm']}: {row['status']} | "
            f"clearance={row['minimum_clearance_m']:.3f}m | "
            f"goal distance={row['final_goal_distance_m']:.3f}m"
        )


if __name__ == "__main__":
    main()
