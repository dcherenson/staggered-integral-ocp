import numpy as np
from scipy.integrate import solve_ivp

class Plant:
    """
    Quadcopter Simulation (3D with Attitude).
    State x = [px, py, pz, vx, vy, vz, phi, theta]^T
    Control u = [p, q, T]^T (Roll rate, Pitch rate, Net Thrust)
    """
    def __init__(self, spatial_mode=False):
        self.m = 1.0  # kg
        self.g = np.array([0, 0, -9.81])
        self.spatial_mode = spatial_mode  # Toggle for online spatial gradient shift
        
        # Aerodynamic drag coefficients (velocity-dependent)
        self.k_drag = np.array([0.3, 0.3, 0.15])  # Drag per axis (N·s/m)
        # Angle-dependent coupling coefficients (phi/theta modulate disturbance)
        self.k_angle = 0.4  # Lateral force coupling from tilt

    def wind_velocity(self, t, p):
        v_wind_x = 2.0 * np.sin(0.5 * t) + 1.0 * np.sin(2.0 * t)
        v_wind_y = 2.4 * np.cos(0.4 * t) + 1.2 * np.cos(1.8 * t)
        v_wind_z = 1.0 * np.sin(0.3 * t)
        
        if self.spatial_mode:
            v_wind_x += 0.5 * (p[0] - 3.0)
            v_wind_y += 0.5 * np.clip(p[1] - 0.5, -1.0, 1.0)
            v_wind_z += 0.2 * p[2]
        wind_vec = np.array([v_wind_x, v_wind_y, v_wind_z])
        
        goal_pos = np.array([6.0, 0.0, 1.0])
        dist = np.linalg.norm(p - goal_pos)
        scale = np.clip(dist / 2.0, 0.0, 1.0)
        return wind_vec * scale

    def unmodeled_dynamics(self, t, p, v, angles):
        """Realistic unmodeled drag using body-frame relative velocity."""
        phi, theta = angles
        # Additive noise
        noise = np.random.randn(3) * np.array([0.2, 0.2, 0.1])
        v_wind = self.wind_velocity(t, p)
        v_rel = v - v_wind
        
        # Rotation matrix R_X(phi) @ R_Y(theta) from body to world
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        R = np.array([
            [cy, 0, sy],
            [sx*sy, cx, -sx*cy],
            [-cx*sy, sx, cx*cy]
        ])
        
        v_b = R.T @ v_rel
        # Body frame quadratic drag coefficients
        D_body = np.diag([0.3, 0.3, 0.6]) # Higher drag vertically due to rotors
        
        # Drag force in world frame: F = - R * D * (v_b * |v_b|)
        drag_force = -self.m * R @ D_body @ (v_b * np.abs(v_b))
        

        
        return drag_force + noise

    def f(self, x: np.ndarray) -> np.ndarray:
        # Expected nominal drift dynamics [dp, dv, dAngles] where dv = g
        return np.array([x[3], x[4], x[5], 0.0, 0.0, -9.81, 0.0, 0.0])

    def g_mat(self, x: np.ndarray) -> np.ndarray:
        # Input u = [p, q, T]^T
        # dv = R(phi, theta) * [0, 0, T]^T / m
        # dAngles = [p, q]
        phi = x[6]
        theta = x[7]
        
        mat = np.zeros((8, 3))
        # Acceleration from Thrust
        mat[3, 2] = np.sin(theta) / self.m
        mat[4, 2] = -np.cos(theta) * np.sin(phi) / self.m
        mat[5, 2] = np.cos(theta) * np.cos(phi) / self.m
        
        # Angular rates from control
        mat[6, 0] = 1.0  # p -> dot{phi}
        mat[7, 1] = 1.0  # q -> dot{theta}
        return mat

    def Delta(self, x: np.ndarray, t: float) -> np.ndarray:
        """Full-state additive mismatch (true - nominal), including vel/angle terms."""
        d = self.unmodeled_dynamics(t, x[0:3], x[3:6], x[6:8])
        return np.concatenate((np.zeros(3), d / self.m))

    def dynamics(self, t, state, u):
        """True dynamics: includes velocity- and angle-dependent disturbance."""
        # state = [px, py, pz, vx, vy, vz, phi, theta]
        p = state[0:3]
        v = state[3:6]
        angles = state[6:8]
        phi, theta = angles

        # True disturbance depends on velocity and body angles
        d = self.unmodeled_dynamics(t, p, v, angles)

        dp = v
        # m a = m g + R * T + d
        dv = self.g + d / self.m
        # Control u = [p, q, T]
        p_rate = u[0]
        q_rate = u[1]
        thrust = u[2]

        dv[0] += thrust / self.m * np.sin(theta)
        dv[1] += thrust / self.m * (-np.cos(theta) * np.sin(phi))
        dv[2] += thrust / self.m * (np.cos(theta) * np.cos(phi))

        dAngles = np.array([p_rate, q_rate])

        return np.concatenate((dp, dv, dAngles))

    def step(self, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Computes the state derivative and integrates one step forward."""
        sol = solve_ivp(self.dynamics, [t, t + dt], x, args=(u,), method="RK45")
        return sol.y[:, -1]
