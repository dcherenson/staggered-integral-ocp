"""
Scenario: Adaptation OFF

The SSML neural network is used as a fixed feed-forward predictor.
Pre-trained weights are frozen and no online adaptation occurs.
The OCP still updates its quantile estimate to bound residual errors.

Usage:
    python main.py --scenario adaptation_off
"""
import numpy as np
from scenarios.config import ScenarioConfig

scenario = ScenarioConfig(
    name="Adaptation OFF",

    # ── RNG ───────────────────────────────────────────────────────────────────
    seed=1337,

    # ── Timing ────────────────────────────────────────────────────────────────
    dt_sim=0.05,          # Plant integration step (s)
    dt_mpc=0.05,          # MPC update period (s)
    t_end=10.0,           # Maximum simulation time (s)

    # ── Initial & goal state ──────────────────────────────────────────────────
    x0=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    x_goal=np.array([6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    goal_radius=0.5,

    # ── Environment ───────────────────────────────────────────────────────────
    spatial_wind=True,
    obstacles=[
        {'pos': np.array([3.0,  0.6, 1.0]), 'r': 0.7},
        {'pos': np.array([1.0, -0.5, 1.0]), 'r': 0.4},
        {'pos': np.array([3.0, -0.7, 1.0]), 'r': 0.3},
        {'pos': np.array([5.0,  0.4, 1.0]), 'r': 0.4},
    ],

    # ── MPC ───────────────────────────────────────────────────────────────────
    mpc_horizon=10,

    # ── OCP ───────────────────────────────────────────────────────────────────
    alpha_ocp=0.1,        # 90% target coverage
    eta_ocp=0.5,          # OCP step size
    q_init_ocp=0.5,       # Initial quantile
    ddot_bound=2.0,       # Lipschitz bound on disturbance derivative
    dist_bound_init=2.0,  # Initial disturbance bound

    # ── SSML Adaptation ───────────────────────────────────────────────────────
    gamma_lr=0.0,         # Adaptation disabled — pre-trained weights frozen
    lambd=0.1,            # (unused when gamma_lr=0, kept for config completeness)
)
