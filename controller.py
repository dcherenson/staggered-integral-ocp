import numpy as np
from plant import Plant


class DynamicTubeMPC:
    """
    Implements Dynamic Tube MPC (DTMPC) treating the tube geometry size (Phi) 
    and control bandwidth (alpha) as decision variables. Supports 3D Space and SSML DNN.
    """
    def __init__(self, plant, obstacles: list, H: int = 5,
                 dt: float = 0.1, eta: float = 0.1):
        self.plant = plant
        self.obstacles = obstacles # List of dicts {'pos': np.array([x,y,z]), 'r': radius}
        self.H = H
        self.dt = dt
        self.eta = 0.0
        self.z_min = 0.8   # Altitude floor (m)
        self.z_max = 1.2   # Altitude ceiling (m)
        
        self.u_max = 30.0  # Increased for z-axis gravity compensation
        self.alpha_min = 0.5
        self.alpha_max = 10.0
        self.Phi = 0.05  # Initial tube size
        self.use_terminal_vel = True # Toggle terminal velocity constraints
        self._prev_opt = None  # Warm-start: previous solution
        self._z_prev = None    # Ancillary controller: predicted nominal state at next step

        # Ancillary feedback gains: u = v* + K(z_prev - x)
        # Mapped to the 8D quad's physical control channels [p_rate, q_rate, thrust]:
        #   z[2] (height) / z[5] (vz)  → thrust correction
        #   z[0] (x-pos) / z[3] (vx)  → pitch rate q (sin(theta) → ax)
        #   z[1] (y-pos) / z[4] (vy)  → roll rate p  (-cos(θ)sin(φ) → ay)
        self.K_pos_z  = 3.0   # z-position error → thrust
        self.K_vel_z  = 2.0   # z-velocity error → thrust
        self.K_pos_xy = 2.0   # x/y-position error → pitch/roll rate
        self.K_vel_xy = 3.5   # x/y-velocity error → pitch/roll rate

    def compute_u(self, x: np.ndarray, xd: np.ndarray, 
                  disturbance_bound: float, d_hat: np.ndarray = None, model_nn=None):
        import torch
        # Check divergence
        if np.any(np.isnan(x)):
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False
                
        # For point-to-point, we start with a small tube centered on the drone
        self.Phi = 0.05
        
        # Evaluate Network Once for Zero-Order Hold Disturbance Prediction over the horizon
        if model_nn is not None and x.shape[0] == 8:
            # Assume hover state initial for disturbance evaluation
            x_in = np.concatenate((x[3:6], x[6:8]))
            with torch.no_grad():
                f_nn_acc = model_nn(torch.tensor(x_in, dtype=torch.float32)).numpy()
        else:
            f_nn_acc = np.zeros(3)

        # ── Build symbolic NLP with CasADi + IPOPT ────────────────────────────
        import casadi as ca

        n_V     = 3 * self.H   # virtual control variables
        n_alpha = self.H       # bandwidth variables
        n_opt   = n_V + n_alpha

        OPT = ca.SX.sym('OPT', n_opt)
        # CasADi reshape is column-major; to match numpy row-major (step-first)
        # we reshape as (3, H) then transpose -> V_sym[j,:] = [p_j, q_j, T_j]
        V_sym     = ca.reshape(OPT[:n_V], 3, self.H).T
        alpha_sym = OPT[n_V:]

        # Numeric constants into CasADi
        x0_ca   = ca.DM(x)
        xd_ca   = ca.DM(xd)
        d_bar   = ca.DM(disturbance_bound)
        nn_acc  = ca.DM(f_nn_acc)

        # Symbolic rollout
        z   = x0_ca
        phi = ca.DM(self.Phi)
        cost  = ca.DM(0.0)
        g_sym = []   # inequality constraints (>= 0)

        obs_pos_list = [ca.DM(o['pos']) for o in self.obstacles]
        obs_rad_list = [o['r'] for o in self.obstacles]

        for j in range(self.H):
            Vj = V_sym[j, :].T     # 3×1

            # ── nominal 8D quad dynamics (linearised g_mat at current z) ──
            phi_z  = z[6]
            theta_z = z[7]
            # Compute f(z) symbolically
            f_z = ca.vertcat(z[3], z[4], z[5],
                             ca.DM(0.0), ca.DM(0.0), ca.DM(-9.81),
                             ca.DM(0.0), ca.DM(0.0))
            # g_mat(z) @ Vj  (only thrust rows depend on angles)
            m = self.plant.m
            sin_t = ca.sin(theta_z);  cos_t = ca.cos(theta_z)
            sin_p = ca.sin(phi_z);    cos_p = ca.cos(phi_z)
            gV = ca.vertcat(ca.DM(0.0), ca.DM(0.0), ca.DM(0.0),
                            sin_t / m * Vj[2],
                            -cos_t * sin_p / m * Vj[2],
                             cos_t * cos_p / m * Vj[2],
                            Vj[0],   # p -> phi_dot
                            Vj[1])   # q -> theta_dot

            # NN feed-forward (3 acc channels, pad to 8)
            nn_full = ca.vertcat(ca.DM([0, 0, 0]), nn_acc, ca.DM([0, 0]))
            d_h = ca.DM(np.zeros(8)) if d_hat is None else ca.DM(d_hat)
            z_dot = f_z + gV + nn_full + d_h
            z = z + z_dot * self.dt

            # Tube dynamics
            phi = phi + self.dt * (-alpha_sym[j] * phi + d_bar + self.eta)

            # Stage cost
            pos_err = z[:3] - xd_ca[:3]
            vel     = z[3:6]
            cost += 10.0 * ca.dot(pos_err, pos_err)
            cost +=  2.0 * ca.dot(vel, vel)  # low penalty — terminal cost handles braking
            cost +=  1.0 * ca.dot(Vj, Vj)
            cost +=  0.5 * phi**2

            # Obstacle clearance constraints: dist - r - phi >= 0
            for obs_p, obs_r in zip(obs_pos_list, obs_rad_list):
                diff = z[:3] - obs_p
                dist = ca.sqrt(ca.dot(diff, diff) + 1e-6)  # smooth sqrt
                g_sym.append(dist - obs_r - phi)

            # Altitude constraints
            g_sym.append(z[2] - self.z_min)
            g_sym.append(self.z_max - z[2])

        # Terminal cost
        pos_err_f = z[:3] - xd_ca[:3]
        vel_f     = z[3:6] - xd_ca[3:6]
        cost += 75.0 * ca.dot(pos_err_f, pos_err_f)
        cost +=  50.0 * ca.dot(vel_f, vel_f)

        # Terminal Velocity Hard Constraints (Equality)
        num_ineq = len(g_sym)
        if self.use_terminal_vel:
            g_sym.append(z[3] - xd_ca[3])
            g_sym.append(z[4] - xd_ca[4])
            g_sym.append(z[5] - xd_ca[5])

        g_ca = ca.vertcat(*g_sym)

        # Bounds
        lbx = []
        ubx = []
        for _ in range(self.H):
            lbx += [-5.0, -5.0,  0.0]
            ubx += [ 5.0,  5.0, 30.0]
        lbx += [self.alpha_min] * self.H
        ubx += [self.alpha_max] * self.H

        lbg = [0.0] * g_ca.shape[0]
        # Inequality constraints are (>= 0), so ubg = inf
        # Terminal velocity constraints are (== 0), so ubg = 0
        ubg = [ca.inf] * num_ineq 
        if self.use_terminal_vel:
            ubg += [0.0] * 3

        # Warm-start
        if self._prev_opt is not None:
            OPT_init = np.zeros(n_opt)
            prev_V = self._prev_opt[:n_V].reshape(self.H, 3)
            OPT_init[:3*(self.H-1)] = prev_V[1:].flatten()
            OPT_init[3*(self.H-1):n_V] = prev_V[-1]
            prev_a = self._prev_opt[n_V:]
            OPT_init[n_V:n_opt-1] = prev_a[1:]
            OPT_init[n_opt-1] = prev_a[-1]
        else:
            OPT_init = np.zeros(n_opt)
            for j in range(self.H):
                OPT_init[j*3 + 2] = 9.81 * self.plant.m
            OPT_init[n_V:] = 1.0

        nlp  = {'x': OPT, 'f': cost, 'g': g_ca}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 300,
            'ipopt.tol': 1e-3,
            'ipopt.constr_viol_tol': 1e-3,
            'ipopt.acceptable_tol': 5e-2,
            'ipopt.acceptable_constr_viol_tol': 5e-2,
            'ipopt.acceptable_iter': 5,       # accept after 5 consecutive acceptable iters
            'ipopt.mu_strategy': 'adaptive',  # better near non-convex constraint boundaries
            'ipopt.nlp_scaling_method': 'gradient-based',
            'print_time': 0,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        def _cold_init():
            """Default hover warm-start (no previous solution)."""
            x0 = np.zeros(n_opt)
            for j in range(self.H):
                x0[j*3 + 2] = 9.81 * self.plant.m
            x0[n_V:] = 1.0
            return x0

        sol = solver(x0=OPT_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()

        # On Restoration_Failed the warm-start led to a bad iterate; retry cold.
        if stats.get('return_status') == 'Restoration_Failed':
            sol = solver(x0=_cold_init(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()

        success = stats['success'] or stats['return_status'] in (
            'Solve_Succeeded', 'Solved_To_Acceptable_Level',
            'Maximum_Iterations_Exceeded')

        # Verify no hard constraint violation even if "acceptable"
        sol_x = np.array(sol['x']).flatten()
        # ── end CasADi NLP ─────────────────────────────────────────────────────

        if success:
            # Double-check obstacle constraints with numpy rollout
            def check_constraints(opt_flat):
                V  = opt_flat[:n_V].reshape(self.H, 3)
                al = opt_flat[n_V:]
                z_ = np.copy(x); phi_ = self.Phi
                cons = []
                for j in range(self.H):
                    for obs, r in zip(self.obstacles, obs_rad_list):
                        cons.append(np.linalg.norm(z_[:3] - obs['pos']) - r - phi_)
                    cons += [z_[2] - self.z_min, self.z_max - z_[2]]
                    phi_val = phi_z = z_[6]; theta_val = z_[7]
                    gV_ = np.zeros(8)
                    gV_[3] = np.sin(theta_val) / m * V[j, 2]
                    gV_[4] = -np.cos(theta_val) * np.sin(phi_val) / m * V[j, 2]
                    gV_[5] = np.cos(theta_val) * np.cos(phi_val) / m * V[j, 2]
                    gV_[6] = V[j, 0]; gV_[7] = V[j, 1]
                    f_ = self.plant.f(z_)
                    nn_ = np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])
                    d_h_np = np.zeros(8) if d_hat is None else d_hat
                    z_ = z_ + (f_ + gV_ + nn_ + d_h_np) * self.dt
                    phi_ = phi_ + self.dt * (-al[j] * phi_ + disturbance_bound + self.eta)
                return np.array(cons)

            if not stats['success']:
                test_cons = check_constraints(sol_x)
                if np.any(test_cons < -0.05):
                    return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False

        if not success:
            print(f"Solver failed (IPOPT status: {stats.get('return_status', 'unknown')})")
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False

        self._prev_opt = sol_x

        V_opt      = sol_x[:n_V].reshape(self.H, 3)
        alphas_opt = sol_x[n_V:]

        # Apply first control (virtual)
        u_out = V_opt[0].copy()

        # ── Ancillary correction: u = v* + K(z_prev - x) ──────────────────
        if self._z_prev is not None:
            e = self._z_prev - x
            du_T = self.K_pos_z * e[2] + self.K_vel_z * e[5]
            du_p = self.K_pos_xy * e[1] + self.K_vel_xy * e[4]
            du_q = self.K_pos_xy * e[0] + self.K_vel_xy * e[3]
            u_out += np.array([du_p, du_q, du_T])
        # ──────────────────────────────────────────────────────────────────

        # Update Phi internally
        self.Phi = self.Phi + self.dt * (-alphas_opt[0] * self.Phi + disturbance_bound + self.eta)

        if np.any(np.isnan(u_out)):
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False
        u_norm = np.linalg.norm(u_out)
        if u_norm > self.u_max:
            u_out = u_out / u_norm * self.u_max

        # Compute full prediction horizon & store nominal next state for ancillary
        z_pred   = []
        phi_pred = []
        z   = np.copy(x)
        phi = self.Phi
        for j in range(self.H):
            d_h_np = np.zeros(8) if d_hat is None else d_hat
            z_dot = self.plant.f(z) + self.plant.g_mat(z) @ V_opt[j] + np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)]) + d_h_np
            z   = z   + z_dot * self.dt
            phi = phi + self.dt * (-alphas_opt[j] * phi + disturbance_bound + self.eta)
            z_pred.append(np.copy(z))
            phi_pred.append(phi)

        # Store z_pred[0] as the predicted nominal state at the NEXT timestep
        self._z_prev = z_pred[0].copy()

        return u_out, np.array(z_pred), np.array(phi_pred), True
