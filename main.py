"""
main.py — Conformal Adaptive Control Simulation

Runs a Dynamic Tube MPC (DTMPC) simulation with an SSML neural network for
disturbance estimation and Online Conformal Prediction (OCP) for safety bounds.

Usage:
    python main.py --scenario adaptation_off   # frozen pre-trained model (default)
    python main.py --scenario adaptation_on    # online weight adaptation enabled

The scenario files in scenarios/ define all tunable parameters. To create a new
scenario, copy an existing one and modify the ScenarioConfig fields.
"""
import os
import argparse
import collections
import importlib
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.switch_backend('Agg')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 20
})

from plant import Plant
from ocp import StaggeredDriftScoreOCP
from controller import DynamicTubeMPC
from ssml import (
    get_or_train_model, flatten_params, assign_params,
    compute_jacobian, spectral_normalization_clip
)


def load_scenario(name: str):
    """Imports and returns the `scenario` object from scenarios/<name>.py."""
    try:
        module = importlib.import_module(f"scenarios.{name}")
    except ModuleNotFoundError:
        raise SystemExit(
            f"Error: scenario '{name}' not found. "
            f"Expected file: scenarios/{name}.py"
        )
    if not hasattr(module, 'scenario'):
        raise SystemExit(
            f"Error: scenarios/{name}.py must define a `scenario` variable "
            f"of type ScenarioConfig."
        )
    return module.scenario


def compute_dist_bound(q_k, L_d, T_p):
    """
    Converts an OCP integral quantile q_k to a disturbance distance bound.

    Uses a piecewise formula derived from the Lipschitz structure of the disturbance:
        - If the implied triangle fits in T_p: d_bar = sqrt(2 * L_d * q_k)
        - Otherwise:                           d_bar = q_k / T_p + 0.5 * L_d * T_p
    """
    thresh = 0.5 * L_d * (T_p ** 2)
    T_safe = max(T_p, 1e-8)
    return np.where(
        q_k < thresh,
        q_k / T_safe + 0.5 * L_d * T_p,
        np.sqrt(2.0 * L_d * q_k)
    )


def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Conformal Adaptive Control Simulation")
    parser.add_argument(
        "--scenario", default="adaptation_off",
        help="Scenario name to load from scenarios/<name>.py (default: adaptation_off)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    cfg = load_scenario(args.scenario)
    print(f"\n{'='*60}")
    print(f"  Scenario: {cfg.name}")
    print(f"  gamma_lr = {cfg.gamma_lr}  |  t_end = {cfg.t_end}s  |  H = {cfg.mpc_horizon}")
    print(f"{'='*60}\n")

    # ── Seeds ─────────────────────────────────────────────────────────────────
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Timing ────────────────────────────────────────────────────────────────
    t = 0.0
    dt_sim = cfg.dt_sim
    dt_mpc = cfg.dt_mpc
    n_substeps = int(round(dt_mpc / dt_sim))
    t_end = cfg.t_end

    # ── System components ─────────────────────────────────────────────────────
    sys_plant = Plant(spatial_mode=cfg.spatial_wind)
    sys_controller = DynamicTubeMPC(
        plant=sys_plant,
        obstacles=cfg.obstacles,
        H=cfg.mpc_horizon,
        dt=dt_mpc,
    )

    # ── OCP ───────────────────────────────────────────────────────────────────
    T_window = int(cfg.mpc_horizon * dt_mpc / dt_sim)  # steps in one MPC horizon
    ocp_integral = StaggeredDriftScoreOCP(
        alpha=cfg.alpha_ocp,
        eta_const=cfg.eta_ocp,
        N_threads=T_window,
        q_init=cfg.q_init_ocp,
    )
    T_p = T_window * dt_sim  # horizon duration in seconds

    # ── SSML model ───────────────────────────────────────────────────────────
    model = get_or_train_model()
    theta_0_flat = flatten_params(model).clone().detach()
    theta_flat   = flatten_params(model).clone().detach()

    # ── State ─────────────────────────────────────────────────────────────────
    x     = np.copy(cfg.x0)
    u     = np.array([0.0, 0.0, 9.81 * sys_plant.m])
    u_old = np.copy(u)

    x_goal      = cfg.x_goal
    goal_radius = cfg.goal_radius

    # ── Logging buffers ───────────────────────────────────────────────────────
    x_plotting         = [np.copy(x)]
    t_plotting         = [t]
    z_pred_plotting    = [np.tile(x, (sys_controller.H, 1))]
    phi_pred_plotting  = [np.tile(sys_controller.Phi, sys_controller.H)]

    past_states        = collections.deque(maxlen=T_window)
    past_f_nom         = collections.deque(maxlen=T_window)

    dist_bound_plotting = [cfg.dist_bound_init]
    xd_plotting       = []
    tube_plotting     = [sys_controller.Phi]
    theta_plotting    = [theta_flat.detach().numpy().copy()]
    disturbance_norm_plotting = [0.0]

    correct_bounds_count = 0
    total_steps_count    = 0



    print("Running SSML DTMPC 3D Flight Simulation...")
    mpc_step_counter   = 0
    t_sim_block_start  = time.time()
    xd = x_goal

    # ── Simulation loop ───────────────────────────────────────────────────────
    while t <= t_end:

        # 1. MPC update (every n_substeps plant steps)
        if int(t / dt_sim + 0.5) % n_substeps == 0:
            mpc_step_counter += 1
            t_solve_start = time.time()
            dist_bound_ocp = float(compute_dist_bound(ocp_integral.get_quantile(), cfg.ddot_bound, T_p))
            u, z_pred, phi_pred, success = sys_controller.compute_u(
                x, xd, dist_bound_ocp, model_nn=model
            )
            solve_time = time.time() - t_solve_start

            if mpc_step_counter % 10 == 0:
                t_now = time.time()
                sim_block_time = t_now - t_sim_block_start
                print(f"t: {t:.2f}s | MPC Solve: {solve_time:.4f}s | Last 50 steps: {sim_block_time:.4f}s")
                t_sim_block_start = t_now

            if not success:
                print(f"Solver failed at t={t:.2f}s! Ending simulation early.")
                break
            z_pred_plotting.append(z_pred)
            phi_pred_plotting.append(phi_pred)
            xd_plotting.append(np.copy(xd))
        else:
            z_pred_plotting.append(z_pred_plotting[-1])
            phi_pred_plotting.append(phi_pred_plotting[-1])
            xd_plotting.append(xd_plotting[-1])

        x_old = np.copy(x)

        # 2. Plant step
        x = sys_plant.step(x_old, u, t, dt_sim)

        # 3. Neural network prediction
        x_in = np.concatenate((x_old[3:6], x_old[6:8]))
        with torch.no_grad():
            f_nn_acc = model(torch.tensor(x_in, dtype=torch.float32)).numpy()

        x_dot_pred = (
            sys_plant.f(x_old)
            + sys_plant.g_mat(x_old) @ u_old
            + np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])
        )

        # True dynamics (ground truth, for monitoring only)
        x_dot_true     = sys_plant.dynamics(t, x_old, u_old)
        error_acc_true = (x_dot_true - x_dot_pred)[3:6]

        # Finite-difference acceleration mismatch for DNN
        v_dot_est  = (x[3:6] - x_old[3:6]) / dt_sim
        error_acc  = v_dot_est - x_dot_pred[3:6]

        # 4. Rolling-window integral OCP score
        past_states.append(np.copy(x_old))
        past_f_nom.append(np.copy(x_dot_pred))

        L     = len(past_states)
        S_I   = 0.0

        if L >= T_window:
            x_buffer     = np.array(past_states)
            f_nom_buffer = np.array(past_f_nom)

            x_k           = x_buffer[-1]
            f_nom_rev     = np.flip(f_nom_buffer, axis=0)
            x_rev         = np.flip(x_buffer, axis=0)
            integrals_rev = np.cumsum(f_nom_rev, axis=0) * dt_sim
            pred_errors   = x_k - x_rev - integrals_rev
            error_norms   = np.linalg.norm(pred_errors, axis=1)
            S_I           = np.max(error_norms)

            q_I        = ocp_integral.update(S_I)
            dist_bound = float(compute_dist_bound(q_I, cfg.ddot_bound, T_p))
        else:
            dist_bound = cfg.dist_bound_init

        # 6. Jacobian calculation
        J = compute_jacobian(model, x_in).detach().numpy()

        # 7. Coverage tracking
        total_steps_count += 1
        if np.linalg.norm(error_acc_true) <= dist_bound:
            correct_bounds_count += 1

        # 8. Online adaptation (only when gamma_lr > 0)
        theta_dot = cfg.gamma_lr * np.dot(J.T, error_acc) - cfg.lambd * (
            theta_flat.detach().numpy() - theta_0_flat.numpy()
        )
        theta_flat         = theta_flat + torch.tensor(theta_dot, dtype=torch.float32) * dt_sim

        assign_params(model, theta_flat)
        spectral_normalization_clip(model)



        # 9. Log
        t += dt_sim
        x_plotting.append(np.copy(x))
        t_plotting.append(t)
        dist_bound_plotting.append(dist_bound)
        tube_plotting.append(sys_controller.Phi)
        theta_plotting.append(theta_flat.detach().numpy().copy())
        disturbance_norm_plotting.append(np.linalg.norm(error_acc_true))

        # 10. Termination checks
        stop_flag = False
        for obs in cfg.obstacles:
            if np.linalg.norm(x[:3] - obs['pos']) < obs['r']:
                print(f"COLLISION at t={t:.2f}s! Penetrated obstacle at {obs['pos']}")
                stop_flag = True
                break
        goal_dist = np.linalg.norm(x[:3] - x_goal[:3])
        if goal_dist < goal_radius:
            print(f"Goal region reached at t={t:.2f}s! Distance: {goal_dist:.3f}m")
            stop_flag = True
        if stop_flag:
            break

    # Pad terminal target
    xd_plotting.append(np.copy(x_goal))

    # ── Convert to arrays ─────────────────────────────────────────────────────
    x_history         = np.array(x_plotting)
    t_history         = np.array(t_plotting)
    xd_history        = np.array(xd_plotting)
    z_pred_hist       = np.array(z_pred_plotting)
    phi_pred_hist     = np.array(phi_pred_plotting)
    dist_bound_history = np.array(dist_bound_plotting)
    disturbance_norm_history = np.array(disturbance_norm_plotting)
    tube_history      = np.array(tube_plotting)
    theta_history     = np.array(theta_plotting)

    # ── Validation ────────────────────────────────────────────────────────────
    min_obs_dist = min(
        np.min(np.linalg.norm(x_history[:, :3] - obs['pos'], axis=1) - obs['r'])
        for obs in cfg.obstacles
    )
    print(f"Minimum distance to any obstacle surface: {min_obs_dist:.3f}m")

    if total_steps_count > 0:
        coverage = correct_bounds_count / total_steps_count
        print(f"\n--- Empirical Coverage Report ---")
        print(f"Disturbance bounded {correct_bounds_count}/{total_steps_count} steps.")
        print(f"Empirical Coverage: {coverage:.2%} (Target: {1-cfg.alpha_ocp:.1%})")
        print(f"---------------------------------\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # SI-OCP bounds vs. true error
    indices = np.arange(len(t_history))
    fig_bounds, ax_b1 = plt.subplots(1, 1, figsize=(10, 5))
    ax_b1.fill_between(indices, 0, dist_bound_history, color='#aae0fa', alpha=0.8, label='SI-OCP Prediction Bound')
    ax_b1.plot(indices, dist_bound_history, 'k-', linewidth=1.5)
    ax_b1.plot(indices, disturbance_norm_history, color='#D95319', label=r'True $d(t)$', linewidth=1.5)
    ax_b1.set_xlim(0, int(t_end / dt_sim)); ax_b1.set_ylim(bottom=0)
    ax_b1.set_xlabel(f'Time (x {dt_sim:.2f} s)'); ax_b1.set_ylabel(r'$\Vert d \Vert$')
    ax_b1.legend(loc='upper right'); ax_b1.grid(True, linestyle=':', alpha=0.7)
    fig_bounds.tight_layout()
    fig_bounds.savefig('output/si-ocp_vs_true.png', dpi=150, bbox_inches='tight')
    print("Plot saved to output/si-ocp_vs_true.png")

    # Neural network parameters
    fig_theta, ax_theta = plt.subplots(1, 1, figsize=(10, 6))
    ax_theta.plot(t_history, theta_history, alpha=0.1, linewidth=1)
    ax_theta.set_xlabel('Time (s)'); ax_theta.set_ylabel(r'NN Weight Values $\theta$')
    ax_theta.set_title('Neural Network Parameters Evolution')
    ax_theta.set_xlim([0, t_end]); ax_theta.grid(True); fig_theta.tight_layout()
    fig_theta.savefig('output/nn_params_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to output/nn_params_vs_time.png")

    # Top-down tube plot
    fig_top, ax_top = plt.subplots(1, 1, figsize=(8, 8))
    ax_top.plot(xd_history[:, 0], xd_history[:, 1], 'k--', alpha=0.5)
    ax_top.plot(x_history[:, 0], x_history[:, 1], 'b-', linewidth=2)
    for obs in cfg.obstacles:
        ax_top.add_patch(plt.Circle((obs['pos'][0], obs['pos'][1]), obs['r'], color='r', alpha=0.2))
    ax_top.scatter(x_goal[0], x_goal[1], color='gold', marker='*', s=150)
    ax_top.add_patch(plt.Circle((x_goal[0], x_goal[1]), goal_radius, color='g', alpha=0.2))

    from matplotlib.patches import Polygon, Circle
    def draw_continuous_tube(ax, path_xy, R, color='c', alpha=0.2):
        radii = np.full(len(path_xy), R) if np.isscalar(R) else np.asarray(R)
        if len(path_xy) < 2:
            return None
        N = len(path_xy)
        tangents = np.zeros_like(path_xy)
        for i in range(1, N - 1):
            t_vec = path_xy[i+1] - path_xy[i-1]
            tangents[i] = t_vec / (np.linalg.norm(t_vec) + 1e-8)
        t0 = path_xy[1] - path_xy[0]
        tangents[0] = t0 / (np.linalg.norm(t0) + 1e-8)
        tn = path_xy[N-1] - path_xy[N-2]
        tangents[-1] = tn / (np.linalg.norm(tn) + 1e-8)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        left_pts  = path_xy + normals * radii[:, np.newaxis]
        right_pts = path_xy - normals * radii[:, np.newaxis]
        theta_t = np.arctan2(tangents[-1, 1], tangents[-1, 0])
        angles = np.linspace(theta_t + np.pi/2, theta_t - np.pi/2, 15)
        center, r_end = path_xy[-1], radii[-1]
        semicircle_pts = np.vstack([
            center[0] + r_end * np.cos(angles),
            center[1] + r_end * np.sin(angles)
        ]).T
        poly_pts = np.vstack((left_pts[:-1], semicircle_pts, right_pts[-2::-1]))
        return ax.add_patch(Polygon(poly_pts, closed=True, facecolor=color, edgecolor=color, alpha=alpha, zorder=2))

    step_0_5 = int(0.75 / dt_sim)
    for i in range(0, len(t_history), step_0_5):
        draw_continuous_tube(ax_top, z_pred_hist[i][:, :2], phi_pred_hist[i], color='c', alpha=0.2)
        wind_vec = sys_plant.wind_velocity(t_history[i], x_history[i, :3]) / 3
        if np.linalg.norm(wind_vec[:2]) > 1e-3:
            ax_top.quiver(x_history[i, 0], x_history[i, 1], wind_vec[0], wind_vec[1],
                          color='purple', width=0.005, scale=15, alpha=0.7, zorder=5)

    ax_top.set_xlabel('X Position (m)'); ax_top.set_ylabel('Y Position (m)')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    custom_handles = [
        Line2D([0], [0], color='b', lw=2, label='Trajectory'),
        Patch(facecolor='c', edgecolor='none', alpha=0.2, label='Predicted Tube'),
        Patch(facecolor='r', edgecolor='none', alpha=0.2, label='Obstacle'),
        Patch(facecolor='g', edgecolor='none', alpha=0.2, label='Goal Region'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Goal'),
        Line2D([0], [0], color='purple', marker=r'$\rightarrow$', markersize=15,
               linestyle='None', label='Wind Vector'),
    ]
    ax_top.legend(handles=custom_handles, loc='upper right')
    ax_top.grid(True); ax_top.set_aspect('equal'); fig_top.tight_layout()
    fig_top.savefig('output/top_down_tube.png', dpi=150, bbox_inches='tight')
    print("Plot saved to output/top_down_tube.png")

    # ── Top-down animation ────────────────────────────────────────────────────
    print("Generating top-down animation...")
    fig_td = plt.figure(figsize=(8, 6))
    ax_td  = fig_td.add_subplot(111)

    line_ref_td,  = ax_td.plot(xd_history[:, 0], xd_history[:, 1], 'k--', alpha=0.5)
    line_true_td, = ax_td.plot([], [], 'b-', label='Quadcopter Path', linewidth=2)
    scatter_true_td = ax_td.scatter([], [], color='blue', s=50, zorder=5)
    line_pred_td, = ax_td.plot([], [], 'c-', alpha=0.3, linewidth=1, zorder=2)
    
    ax_td.scatter(x_goal[0], x_goal[1], color='gold', marker='*', s=150, zorder=6)
    ax_td.add_patch(Circle((x_goal[0], x_goal[1]), goal_radius, color='g', alpha=0.2))
    
    current_tube_patch = None
    quiver_wind_td = None

    def init_td():
        nonlocal current_tube_patch
        line_true_td.set_data([], [])
        scatter_true_td.set_offsets(np.empty((0, 2)))
        line_pred_td.set_data([], [])
        if current_tube_patch is not None:
            current_tube_patch.remove()
            current_tube_patch = None
        return scatter_true_td, line_true_td, line_ref_td, line_pred_td

    def update_graph_td(k):
        nonlocal quiver_wind_td, current_tube_patch
        scatter_true_td.set_offsets(np.column_stack((x_history[k:k+1, 0], x_history[k:k+1, 1])))
        line_true_td.set_data(x_history[:k+1, 0], x_history[:k+1, 1])
        line_pred_td.set_data(z_pred_hist[k, :, 0], z_pred_hist[k, :, 1])
        
        # Draw current predicted tube polygon
        if current_tube_patch is not None:
            current_tube_patch.remove()
        current_tube_patch = draw_continuous_tube(ax_td, z_pred_hist[k][:, :2], phi_pred_hist[k], color='c', alpha=0.2)
        
        ax_td.set_title(rf'DTMPC Top-Down | t={t_history[k]:.1f}s | $\Phi$={tube_history[k]:.2f}m')
        if quiver_wind_td is not None:
            quiver_wind_td.remove()
        wind_vec = sys_plant.wind_velocity(t_history[k], x_history[k, :3]) / 3
        quiver_wind_td = ax_td.quiver(
            x_history[k, 0], x_history[k, 1], wind_vec[0], wind_vec[1],
            color='purple', width=0.005, scale=15, alpha=0.7, zorder=5
        )
        return scatter_true_td, line_true_td, line_ref_td, line_pred_td
    for obs in cfg.obstacles:
        ax_td.add_patch(Circle((obs['pos'][0], obs['pos'][1]), obs['r'], color='red', alpha=0.2))
    x_lo = np.min(x_history[:, 0]) - 0.5; x_hi = np.max(x_history[:, 0]) + 0.5
    ax_td.set_xlim([x_lo, x_hi]); ax_td.set_ylim([-2.5, 2.5])
    ax_td.set_aspect('equal'); ax_td.set_xlabel('X Position (m)'); ax_td.set_ylabel('Y Position (m)')
    ax_td.legend(handles=custom_handles, loc='upper right')
    ax_td.grid(True)
    # Calculate real-time animation parameters
    ani_skip = 1
    ani_interval = ani_skip * cfg.dt_sim * 1000  # ms between frames for real-time
    ani_fps = 1.0 / (ani_skip * cfg.dt_sim)      # frames per second for real-time

    ani_td = animation.FuncAnimation(
        fig_td, update_graph_td, init_func=init_td,
        frames=range(0, len(x_history), ani_skip), interval=ani_interval, blit=False
    )
    ani_td.save('output/top_down_animation.gif', writer='pillow', fps=ani_fps)
    print(f"Top-down animation saved to output/top_down_animation.gif")




if __name__ == '__main__':
    main()

