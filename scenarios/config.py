"""
ScenarioConfig: defines all parameters needed to fully specify a simulation run.

Load a scenario in main.py with:
    from scenarios.adaptation_on import scenario
or via CLI:
    python main.py --scenario adaptation_on
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ScenarioConfig:
    """
    All parameters required to fully specify a simulation scenario.
    """

    # ── Scenario identity ─────────────────────────────────────────────────────
    name: str
    """Human-readable name shown in terminal output."""

    # ── RNG seeds ─────────────────────────────────────────────────────────────
    seed: int = 1337
    """Numpy and PyTorch RNG seed for reproducibility."""

    # ── Timing ────────────────────────────────────────────────────────────────
    dt_sim: float = 0.05
    """Plant integration step size (seconds). Controls simulation fidelity."""

    dt_mpc: float = 0.05
    """MPC update period (seconds). Must be a multiple of dt_sim."""

    t_end: float = 10.0
    """Maximum simulation duration (seconds)."""

    # ── Initial & goal state ──────────────────────────────────────────────────
    x0: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    """Initial state [px, py, pz, vx, vy, vz, phi, theta] (m, m/s, rad)."""

    x_goal: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    """Goal state [px, py, pz, vx, vy, vz, phi, theta]."""

    goal_radius: float = 0.5
    """Acceptance radius around goal position (meters)."""

    # ── Environment ───────────────────────────────────────────────────────────
    spatial_wind: bool = True
    """Enable spatially-varying wind disturbances in the plant."""

    obstacles: list = field(default_factory=lambda: [
        {'pos': np.array([3.0,  0.6, 1.0]), 'r': 0.7},
        {'pos': np.array([1.0, -0.5, 1.0]), 'r': 0.4},
        {'pos': np.array([3.0, -0.7, 1.0]), 'r': 0.3},
        {'pos': np.array([5.0,  0.4, 1.0]), 'r': 0.4},
    ])
    """
    List of spherical obstacles, each a dict with:
        'pos': np.ndarray [x, y, z]  — obstacle center (m)
        'r':   float                 — obstacle radius (m)
    """

    # ── MPC ───────────────────────────────────────────────────────────────────
    mpc_horizon: int = 10
    """Number of prediction steps H in the DTMPC horizon."""

    # ── Online Conformal Prediction (OCP) ─────────────────────────────────────
    alpha_ocp: float = 0.1
    """Mis-coverage rate alpha in (0, 1). Target coverage = 1 - alpha."""

    eta_ocp: float = 0.5
    """OCP quantile step size (eta_const)."""

    q_init_ocp: float = 0.5
    """Initial OCP quantile estimate q_0."""

    ddot_bound: float = 2.0
    """
    Lipschitz bound on disturbance derivative (L_d).
    Used to convert OCP quantile into a distance bound via:
        d_bar = sqrt(2 * L_d * q_k)   or   q_k/T + 0.5 * L_d * T  (piecewise)
    """

    dist_bound_init: float = 3.0
    """Initial disturbance bound used before the OCP window is full."""

    # ── SSML Adaptation ───────────────────────────────────────────────────────
    gamma_lr: float = 0.0
    """
    Online adaptation learning rate for neural network weights.
    Set to 0.0 to freeze the pre-trained model (no adaptation).
    Set > 0.0 to enable online weight adaptation from observed errors.
    """

    lambd: float = 0.1
    """
    L2 regularization coefficient pulling weights back toward pre-trained values.
    Only active when gamma_lr > 0.
    """
