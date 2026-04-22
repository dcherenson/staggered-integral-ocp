"""
test_ocp.py — Unit tests for the Online Conformal Prediction (OCP) module.

Run with:
    python test_ocp.py
"""
import numpy as np
from ocp import StaggeredDriftScoreOCP


def test_staggered_ocp_update():
    """Tests the basic pinball-loss quantile update logic in Step 0."""
    print("Testing StaggeredDriftScoreOCP.update() basic logic...")

    # For N=1, it behaves like standard OCP
    ocp = StaggeredDriftScoreOCP(alpha=0.1, eta_const=0.5, N_threads=1, q_init=1.0)
    assert np.isclose(ocp.get_quantile(), 1.0), "Initial quantile should be 1.0"

    # S_1 = 0.5 <= q_0 = 1.0  →  S_1 not > q_0  →  indicator = 0
    # q_1 = max(0, 1.0 + 0.5 * (0 - 0.1)) = 0.95
    q1 = ocp.update(0.5)
    assert np.isclose(q1, 0.95), f"Expected q1=0.95, got {q1:.4f}"
    print(f"  q1 = {q1:.4f}  ✓")

    # S_2 = 2.0 > q_1  →  indicator = 1
    # q_2 = max(0, 0.95 + 0.5 * (1 - 0.1)) = 0.95 + 0.45 = 1.40
    q2 = ocp.update(2.0)
    assert np.isclose(q2, 1.40), f"Expected q2=1.40, got {q2:.4f}"
    print(f"  q2 = {q2:.4f}  ✓")
    print("  Basic update logic PASSED\n")


def test_staggered_ocp_non_negative():
    """Tests that q_k is always clamped to >= 0."""
    print("Testing StaggeredDriftScoreOCP non-negativity clamp...")

    ocp = StaggeredDriftScoreOCP(alpha=0.9, eta_const=1.0, N_threads=1, q_init=0.05)
    for _ in range(10):
        ocp.update(100.0)  # S always > q, but high alpha leads to negative shift
    assert ocp.get_quantile() >= 0.0, "q_k must never go negative"
    print("  Non-negativity PASSED\n")


def test_staggered_ocp_threads():
    """Tests that StaggeredDriftScoreOCP rotates threads correctly."""
    print("Testing StaggeredDriftScoreOCP thread rotation...")

    N = 3
    ocp = StaggeredDriftScoreOCP(alpha=0.1, eta_const=0.5, N_threads=N, q_init=1.0)

    # All threads should start at q_init
    assert np.all(np.isclose(ocp.qs, 1.0)), "All threads should initialise to q_init"

    # Step 0 → thread 0 updates; step 1 → thread 1; step 2 → thread 2; step 3 → thread 0 again
    ocp.update(0.5)   # thread 0: S <= q → q goes down (0.95)
    ocp.update(5.0)   # thread 1: S > q  → q goes up   (1.45)
    ocp.update(5.0)   # thread 2: S > q  → q goes up   (1.45)
    
    assert np.isclose(ocp.qs[0], 0.95)
    assert np.isclose(ocp.qs[1], 1.45)
    assert np.isclose(ocp.qs[2], 1.45)
    
    ocp.update(1.0)   # thread 0 again: S > 0.95 → q goes up (0.95 + 0.45 = 1.40)
    assert np.isclose(ocp.qs[0], 1.40)
    
    print(f"  Thread quantiles after 4 steps: {ocp.qs}  ✓")
    print("  Thread rotation PASSED\n")


def test_staggered_ocp_coverage():
    """
    Long-run coverage test for StaggeredDriftScoreOCP.
    Scores drawn from Uniform[0,1], alpha=0.2 → target coverage 80%.
    """
    print("Testing StaggeredDriftScoreOCP long-run coverage...")

    np.random.seed(0)
    N_threads = 10
    ocp = StaggeredDriftScoreOCP(alpha=0.2, eta_const=0.2, N_threads=N_threads, q_init=0.5)
    N = 10000
    violations = 0
    for _ in range(N):
        s = np.random.uniform(0, 1)
        if s > ocp.get_quantile():
            violations += 1
        ocp.update(s)

    empirical_violation_rate = violations / N
    # Tolerance for stochastic convergence
    assert abs(empirical_violation_rate - 0.2) < 0.05, (
        f"Coverage too far from target: {empirical_violation_rate:.3f} vs 0.2"
    )
    print(f"  Empirical violation rate: {empirical_violation_rate:.3f} (target: 0.20)  ✓")
    print("  Coverage PASSED\n")


def test_dist_bound_formula():
    """Tests get_dist_bound_from_quantile for both branches of the piecewise formula."""
    print("Testing StaggeredDriftScoreOCP.get_dist_bound_from_quantile()...")

    ocp = StaggeredDriftScoreOCP(alpha=0.1, eta_const=0.5, N_threads=5, q_init=0.0)

    L_d = 2.0
    T   = 1.0

    # Small q: triangle fits in T → d_bar = sqrt(2 * L_d * q)
    q_small = 0.01
    d_small = ocp.get_dist_bound_from_quantile(q_small, T, L_d)
    expected_small = np.sqrt(2.0 * L_d * q_small)
    assert np.isclose(d_small, expected_small, rtol=1e-5)
    print(f"  Small-q branch: d_bar = {d_small:.6f}  ✓")

    # Large q: triangle overflows T → d_bar = q/T + 0.5*L_d*T
    q_large = 10.0
    d_large = ocp.get_dist_bound_from_quantile(q_large, T, L_d)
    expected_large = q_large / T + 0.5 * L_d * T
    assert np.isclose(d_large, expected_large, rtol=1e-5)
    print(f"  Large-q branch: d_bar = {d_large:.6f}  ✓")
    print("  get_dist_bound_from_quantile PASSED\n")


if __name__ == "__main__":
    test_staggered_ocp_update()
    test_staggered_ocp_non_negative()
    test_staggered_ocp_threads()
    test_staggered_ocp_coverage()
    test_dist_bound_formula()
    print("=" * 50)
    print("All OCP tests PASSED!")
    print("=" * 50)
