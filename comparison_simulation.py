"""DT-MPC simulation entry points for SIOCP and DOB-DT-MPC comparisons."""

from __future__ import annotations

import collections
import importlib
import time
from typing import Any

import numpy as np
import torch

from controller import DynamicTubeMPC
from disturbance_observer import HighGainDisturbanceObserver
from ocp import StaggeredDriftScoreOCP
from plant import Plant
from comparison_noise import ReplayGaussianNoise
from ssml import (
    assign_params,
    compute_jacobian,
    flatten_params,
    get_or_train_model,
    spectral_normalization_clip,
)


def load_scenario(name: str):
    try:
        module = importlib.import_module(f"scenarios.{name}")
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Error: scenario '{name}' not found") from exc
    if not hasattr(module, "scenario"):
        raise SystemExit(f"Error: scenarios/{name}.py must define `scenario`")
    return module.scenario


def compute_dist_bound(q_k, L_d, T_p):
    """Convert an SIOCP integral quantile into the scalar DT-MPC bound."""
    q_k = np.asarray(q_k, dtype=float)
    thresh = 0.5 * float(L_d) * float(T_p) ** 2
    safe_T = max(float(T_p), 1e-8)
    return np.where(q_k < thresh, q_k / safe_T + 0.5 * L_d * T_p,
                    np.sqrt(2.0 * L_d * q_k))


def _run_dtmpc(
    cfg: Any,
    *,
    plant: Plant | None = None,
    bound_source: str = "siocp",
) -> dict[str, Any]:
    """Run the unchanged DT-MPC plant/controller loop with an injected bound."""
    if bound_source not in {"siocp", "dob"}:
        raise ValueError("bound_source must be 'siocp' or 'dob'")

    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    dt_sim, dt_mpc = float(cfg.dt_sim), float(cfg.dt_mpc)
    n_substeps = int(round(dt_mpc / dt_sim))
    if n_substeps < 1 or not np.isclose(n_substeps * dt_sim, dt_mpc):
        raise ValueError("dt_mpc must be an integer multiple of dt_sim")

    sys_plant = plant if plant is not None else Plant(
        spatial_mode=cfg.spatial_wind,
        noise_source=ReplayGaussianNoise(cfg.seed, cfg.dt_sim, cfg.t_end + cfg.dt_sim),
    )
    controller = DynamicTubeMPC(sys_plant, cfg.obstacles, H=cfg.mpc_horizon, dt=dt_mpc)
    T_window = int(cfg.mpc_horizon * dt_mpc / dt_sim)
    T_p = T_window * dt_sim
    siocp_initial_bound = float(compute_dist_bound(cfg.q_init_ocp, cfg.ddot_bound, T_p))
    ocp = None
    observer = None
    if bound_source == "siocp":
        ocp = StaggeredDriftScoreOCP(
            alpha=cfg.alpha_ocp, eta_const=cfg.eta_ocp,
            N_threads=T_window, q_init=cfg.q_init_ocp,
        )
    else:
        observer = HighGainDisturbanceObserver(
            gain=5.0, derivative_bound=float(cfg.ddot_bound),
            # Match the bound used by SIOCP's first MPC solve. This also
            # initializes Eq. 37 consistently, rather than overriding only
            # the first DOB control input.
            initial_error_bound=siocp_initial_bound,
        )
        observer.initialize(np.asarray(cfg.x0[3:6], dtype=float))

    model = get_or_train_model()
    theta_0 = flatten_params(model).clone().detach()
    theta = theta_0.clone().detach()
    x = np.asarray(cfg.x0, dtype=float).copy()
    x_goal = np.asarray(cfg.x_goal, dtype=float)
    # Historical SIOCP and the DOB observer both use the fixed hover input as
    # their nominal residual reference.
    u_old = np.array([0.0, 0.0, 9.81 * sys_plant.m], dtype=float)
    u = u_old.copy()

    x_history, t_history = [x.copy()], [0.0]
    physical_history = []
    control_times, control_bounds = [], []
    initial_history_bound = siocp_initial_bound if observer is not None else float(cfg.dist_bound_init)
    bound_history, residual_norm_history = [initial_history_bound], [0.0]
    residual_vectors = []
    observer_estimates, observer_margins = [], []
    observer_step_estimates, observer_step_margins = [], []
    estimation_errors, bound_misses = [], []
    z_pred_history, phi_pred_history = [], []
    tube_history, theta_history = [float(controller.Phi)], [theta.numpy().copy()]
    past_states = collections.deque(maxlen=T_window)
    past_nominal = collections.deque(maxlen=T_window)
    update_times, update_sim_times = [], []
    correct_bounds = total_steps = 0
    t = 0.0
    status = "safe_not_reached"
    failure_message = ""
    last_prediction = None
    current_bound = initial_history_bound

    while t <= float(cfg.t_end) + 1e-12:
        is_update = int(round(t / dt_sim)) % n_substeps == 0
        d_hat_used = margin_used = None
        if observer is not None:
            # The estimate is sampled causally at the beginning of every
            # plant interval; the held scalar bound is refreshed only by MPC.
            d_hat_step = observer.estimate(x[3:6])
            margin_step = float(observer.error_bound(t))
            observer_step_estimates.append(d_hat_step.copy())
            observer_step_margins.append(margin_step)
        if is_update:
            started = time.perf_counter()
            if bound_source == "siocp":
                bound_for_control = float(compute_dist_bound(ocp.get_quantile(), cfg.ddot_bound, T_p))
            else:
                d_hat_used = d_hat_step
                margin_used = margin_step
                bound_for_control = float(np.linalg.norm(d_hat_used) + margin_used)
            if not np.isfinite(bound_for_control) or bound_for_control < 0.0:
                status = "controller_failure"
                failure_message = "nonfinite or negative disturbance bound"
                break
            control_times.append(float(t))
            control_bounds.append(bound_for_control)
            current_bound = bound_for_control
            if observer is not None:
                observer_estimates.append(d_hat_used.copy())
                observer_margins.append(margin_used)
            try:
                u, z_pred, phi_pred, success = controller.compute_u(
                    x, x_goal, bound_for_control, model_nn=model
                )
            except Exception as exc:
                success = False
                z_pred = phi_pred = None
                failure_message = f"DT-MPC exception: {exc}"
            update_times.append(time.perf_counter() - started)
            update_sim_times.append(float(t))
            if not success or np.asarray(u).shape != (3,) or not np.all(np.isfinite(u)):
                status = "controller_failure"
                if not failure_message:
                    failure_message = "DT-MPC solver failed or returned a nonfinite input"
                break
            last_prediction = (np.asarray(z_pred), np.asarray(phi_pred))
        if last_prediction is not None:
            z_pred_history.append(last_prediction[0])
            phi_pred_history.append(last_prediction[1])

        x_old = x.copy()
        x = sys_plant.step(x_old, u, t, dt_sim)
        x_in = np.r_[x_old[3:6], x_old[6:8]]
        x_in_end = np.r_[x[3:6], x[6:8]]
        with torch.no_grad():
            nn_start = model(torch.tensor(x_in, dtype=torch.float32)).numpy()
            nn_end = model(torch.tensor(x_in_end, dtype=torch.float32)).numpy()
        nominal_start = sys_plant.f(x_old) + sys_plant.g_mat(x_old) @ u_old
        nominal_start += np.r_[np.zeros(3), nn_start, np.zeros(2)]
        nominal_end = sys_plant.f(x) + sys_plant.g_mat(x) @ u_old
        nominal_end += np.r_[np.zeros(3), nn_end, np.zeros(2)]
        transition_residual = (x[3:6] - x_old[3:6]) / dt_sim - nominal_start[3:6]
        residual_vectors.append(transition_residual.copy())
        residual_norm_history.append(float(np.linalg.norm(transition_residual)))

        if observer is not None:
            observer.update(x_old[3:6], x[3:6], nominal_start[3:6], nominal_end[3:6], dt_sim)
            error = float(np.linalg.norm(transition_residual - d_hat_step))
            miss = bool(error > margin_step + 1e-12)
            estimation_errors.append(error)
            bound_misses.append(miss)
            total_steps += 1
            correct_bounds += int(not miss)
            bound_history.append(current_bound)
        else:
            past_states.append(x_old.copy())
            past_nominal.append(nominal_start.copy())
            if len(past_states) >= T_window:
                x_buf, f_buf = np.asarray(past_states), np.asarray(past_nominal)
                prediction_errors = x_buf[-1] - np.flip(x_buf, axis=0)
                prediction_errors -= np.cumsum(np.flip(f_buf, axis=0), axis=0) * dt_sim
                q = ocp.update(float(np.max(np.linalg.norm(prediction_errors, axis=1))))
                next_bound = float(compute_dist_bound(q, cfg.ddot_bound, T_p))
            else:
                next_bound = float(cfg.dist_bound_init)
            bound_history.append(next_bound)
            total_steps += 1
            correct_bounds += int(np.linalg.norm(transition_residual) <= next_bound)

        J = compute_jacobian(model, x_in).detach().numpy()
        theta_dot = cfg.gamma_lr * np.dot(J.T, transition_residual) - cfg.lambd * (theta.numpy() - theta_0.numpy())
        theta = theta + torch.tensor(theta_dot, dtype=torch.float32) * dt_sim
        assign_params(model, theta)
        spectral_normalization_clip(model)

        t += dt_sim
        x_history.append(x.copy())
        t_history.append(t)
        physical_history.append(np.asarray(u, dtype=float).copy())
        tube_history.append(float(controller.Phi))
        theta_history.append(theta.numpy().copy())
        if any(np.linalg.norm(x[:3] - obs["pos"]) < obs["r"] for obs in cfg.obstacles):
            status = "collision"
            break
        if np.linalg.norm(x[:3] - x_goal[:3]) < cfg.goal_radius:
            status = "goal_reached"
            break

    updates = np.asarray(update_times, dtype=float)
    update_sim_times = np.asarray(update_sim_times, dtype=float)
    steady = updates[1:]
    average = float(np.mean(steady)) if steady.size else float("nan")
    if bound_source == "siocp":
        print(f"Average SIOCP DT-MPC update time: {average:.6f}s across {steady.size} updates (first update excluded)")
    x_history, t_history = np.asarray(x_history), np.asarray(t_history)
    clearance = (
        np.min(np.stack([np.linalg.norm(x_history[:, :3] - o["pos"], axis=1) - o["r"] for o in cfg.obstacles], axis=1), axis=1)
        if cfg.obstacles else np.full(len(x_history), np.inf)
    )
    goal_distance = np.linalg.norm(x_history[:, :3] - x_goal[:3], axis=1)
    replay = sys_plant.noise_source
    replay_id, noise_seed, noise_period = replay.identifier, replay.seed, replay.sample_period
    misses = np.asarray(bound_misses, dtype=bool)
    bound_miss = misses if observer is not None else np.asarray(residual_norm_history[1:]) > np.asarray(bound_history[1:])
    result = {
        "algorithm": "siocp" if bound_source == "siocp" else "dob_dtmpc",
        "scenario": cfg.name, "status": status,
        "failure_message": failure_message,
        "time": t_history, "state": x_history,
        "physical_control": np.asarray(physical_history, dtype=float).reshape(-1, 3),
        "clearance": clearance, "goal_distance": goal_distance,
        "disturbance_bound": np.asarray(bound_history, dtype=float),
        "control_time": np.asarray(control_times, dtype=float), "control_bound": np.asarray(control_bounds, dtype=float),
        "disturbance_norm": np.asarray(residual_norm_history, dtype=float),
        "realized_average_disturbance": np.asarray(residual_vectors, dtype=float).reshape(-1, 3),
        "empirical_coverage": float(correct_bounds / total_steps) if total_steps else float("nan"),
        "prediction_miss_semantics": (
            "observer estimation error versus Eq. 37 margin"
            if observer is not None else
            "transition acceleration residual versus the SIOCP bound"
        ),
        "observer_estimate": np.asarray(observer_estimates, dtype=float).reshape(-1, 3),
        "observer_margin": np.asarray(observer_margins, dtype=float),
        "observer_estimate_interval": np.asarray(observer_step_estimates, dtype=float).reshape(-1, 3),
        "observer_margin_interval": np.asarray(observer_step_margins, dtype=float),
        "observed_estimation_error": np.asarray(estimation_errors, dtype=float),
        "bound_miss": bound_miss, "bound_miss_time": t_history[1:1 + len(bound_miss)][bound_miss],
        "bound_source": bound_source, "formal_safety_guarantee": False if observer is not None else None,
        "observer_gain": 5.0 if observer is not None else None,
        "observer_initial_error_bound": siocp_initial_bound if observer is not None else None,
        "initial_bound_policy": "ocp_quantile" if observer is None else "siocp_quantile_eq37",
        "seed": noise_seed, "noise_seed": noise_seed, "replay_id": replay_id, "noise_id": replay_id,
        "noise_sample_period": noise_period, "dt_sim": dt_sim, "dt_control": dt_mpc,
        "spatial_wind": bool(cfg.spatial_wind), "config": cfg, "plant": sys_plant,
        "z_pred_history": np.asarray(z_pred_history, dtype=float), "phi_pred_history": np.asarray(phi_pred_history, dtype=float),
        "tube_history": np.asarray(tube_history, dtype=float), "theta_history": np.asarray(theta_history, dtype=float),
        "dtmpc_update_time_s": updates, "dtmpc_update_simulation_time_s": update_sim_times,
        "average_dtmpc_update_time_s": average, "dtmpc_update_count": int(updates.size),
        "dtmpc_average_update_count": int(steady.size),
    }
    if bound_source == "siocp":
        result.update({
            "siocp_update_time_s": updates, "siocp_update_simulation_time_s": update_sim_times,
            "average_siocp_update_time_s": average, "siocp_update_count": int(updates.size),
            "siocp_average_update_count": int(steady.size),
        })
    return result


def simulate_siocp(cfg, *, plant=None):
    """Run SIOCP with its historical quantile-derived MPC bound."""
    return _run_dtmpc(
        cfg, plant=plant, bound_source="siocp",
    )


def simulate_dob_dtmpc(cfg, *, plant=None):
    """Run DT-MPC using the causal high-gain observer bound."""
    return _run_dtmpc(
        cfg, plant=plant,
        bound_source="dob",
    )

