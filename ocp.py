import numpy as np


class StaggeredDriftScoreOCP:
    """
    Staggered Online Conformal Prediction (OCP) for drift scores.
    
    Uses N independent threads to handle delayed feedback (where thread j 
    handles every N-th timestep), preventing overlap between a prediction 
    and its eventual ground-truth realization.
    
    Each thread tracks its own quantile estimate q_k using online gradient 
    descent on the pinball loss (Eq 29 from the paper).
    """
    def __init__(self, alpha: float, eta_const: float, N_threads: int, q_init: float = 0.0):
        """
        Args:
            alpha (float): User-specified mis-coverage rate in (0, 1). Target coverage = 1 - alpha.
            eta_const (float): Constant step size for quantile updates.
            N_threads (int): Number of independent OCP threads (matching the feedback delay/horizon).
            q_init (float): Initial value for all quantile estimates.
        """
        if not (0 < alpha < 1):
            raise ValueError("Mis-coverage rate alpha must be between 0 and 1.")
            
        self.alpha = alpha
        self.eta_const = eta_const
        self.N = N_threads
        self.qs = np.full(self.N, max(0.0, q_init))
        self.k_step = 0
        
    def update(self, S_k: float) -> float:
        """
        Updates the active thread's quantile estimate based on the current score.
        Thread j = k % N is updated at each step.
        
        Args:
            S_k: The non-conformity score (observed drift) at the current step.
            
        Returns:
            The newly updated quantile for the active thread.
        """
        j = self.k_step % self.N
        
        # Indicator function: 1 if score EXCEEDS quantile (violation), else 0
        indicator = 1.0 if S_k > self.qs[j] else 0.0
        
        # Pinball loss gradient update
        self.qs[j] = max(0.0, self.qs[j] + self.eta_const * (indicator - self.alpha))
        q_out = self.qs[j]
        
        self.k_step += 1
        return q_out
        
    def get_quantile(self) -> float:
        """Returns the quantile estimate of the currently active thread."""
        j = self.k_step % self.N
        return self.qs[j]

    def get_dist_bound_from_quantile(self, q_k: float, T: float, L_d: float) -> float:
        """
        Converts an integral OCP quantile q_k to a disturbance distance bound d_bar.

        Args:
            q_k: Current quantile estimate.
            T: Time window length (seconds).
            L_d: Lipschitz bound on the disturbance derivative.

        Returns:
            d_bar: Upper bound on the disturbance displacement over T.
        """
        d_triangle = np.sqrt(2.0 * L_d * q_k)
        triangle_base = 2.0 * d_triangle / L_d
        if triangle_base <= T:
            # Disturbance is locally 'small' relative to the window
            d_bar = d_triangle
        else:
            # Window is too short to contain the full disturbance growth triangle
            d_bar = (q_k / T) + 0.5 * L_d * T
        return d_bar
