"""Metric tests on synthetic trajectories with known ground truth.

The co-articulation index is the headline number, and its sign convention involves
folding by reach direction.  A flipped sign would invert the conclusion, so it is
pinned here against data built to a known slope.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import metrics, probes


def make_traj(t1_vals, t2_vals, slope, intercept=0.3, T=60, go=10, acq=40):
    """Synthetic rollout where velocity at T1 acquisition is, by construction,
    ``intercept + slope * d_next`` in direction-folded coordinates."""
    n = len(t1_vals)
    t1, t2 = np.asarray(t1_vals, float), np.asarray(t2_vals, float)
    sign = np.sign(t1)
    d_next = (t2 - t1) * sign
    v_out = intercept + slope * d_next

    v = np.zeros((T, n))
    x = np.zeros((T, n))
    for i in range(n):
        v[go:acq + 1, i] = np.linspace(0, v_out[i] * sign[i], acq + 1 - go)
        v[acq, i] = v_out[i] * sign[i]  # velocity at the acquisition sample
        x[:, i] = np.cumsum(v[:, i]) * 0.02

    acq_step = np.full((n, 3), -1)
    acq_step[:, 0] = acq
    return {
        "v": v, "x": x, "acq_step": acq_step,
        "go_step": np.full(n, go), "dt": 0.02,
        "alive": np.ones((T, n), bool),
        "end_step": np.full(n, T - 1),
    }


def design_for(t1_vals, t2_vals, preview=2):
    t1 = np.asarray(t1_vals, float)
    return {"t1": t1, "t2": np.asarray(t2_vals, float),
            "preview": np.full(len(t1), preview),
            "direction": np.sign(t1)}


def test_ci_recovers_a_known_positive_slope():
    t2 = np.linspace(-0.9, 0.9, 25)
    t1 = np.full(25, 0.4)
    traj = make_traj(t1, t2, slope=0.5)
    res = metrics.co_articulation(traj, design_for(t1, t2))
    assert res["ci"] == pytest.approx(0.5, abs=1e-6)


def test_ci_recovers_a_known_negative_slope():
    t2 = np.linspace(-0.9, 0.9, 25)
    t1 = np.full(25, 0.4)
    traj = make_traj(t1, t2, slope=-0.35)
    res = metrics.co_articulation(traj, design_for(t1, t2))
    assert res["ci"] == pytest.approx(-0.35, abs=1e-6)


def test_ci_is_zero_when_velocity_ignores_the_next_target():
    t2 = np.linspace(-0.9, 0.9, 25)
    t1 = np.full(25, 0.4)
    traj = make_traj(t1, t2, slope=0.0)
    res = metrics.co_articulation(traj, design_for(t1, t2))
    assert res["ci"] == pytest.approx(0.0, abs=1e-9)


def test_direction_folding_pools_leftward_and_rightward_reaches():
    """The two reach directions must reinforce, not cancel.  Without folding, mirrored
    trials would produce a slope near zero regardless of the true effect."""
    g = np.linspace(-0.9, 0.9, 25)
    t1 = np.r_[np.full(25, 0.4), np.full(25, -0.4)]
    t2 = np.r_[g, -g]
    traj = make_traj(t1, t2, slope=0.5)
    res = metrics.co_articulation(traj, design_for(t1, t2))
    assert res["ci"] == pytest.approx(0.5, abs=1e-6)

    # both directions individually agree with the pooled estimate
    for sel in (slice(0, 25), slice(25, 50)):
        mask = np.zeros(50, dtype=bool)
        mask[sel] = True
        sub = metrics.co_articulation(
            metrics._subset(traj, mask), design_for(t1[sel], t2[sel])
        )
        assert sub["ci"] == pytest.approx(0.5, abs=1e-6)


def test_positive_ci_means_momentum_carried_toward_the_next_target():
    """Ground the sign in physical terms: T2 further along the reach direction must
    correspond to a larger outward velocity at T1."""
    t2 = np.linspace(-0.9, 0.9, 10)  # index 0 = back toward start, index -1 = onward
    t1 = np.full(10, 0.4)
    traj = make_traj(t1, t2, slope=0.5)
    v_back, v_onward = traj["v"][40, 0], traj["v"][40, -1]
    assert v_onward > v_back, "T2 further along the reach must raise outward velocity"
    res = metrics.co_articulation(traj, design_for(t1, t2))
    assert res["ci"] > 0


def test_subset_keeps_time_major_and_per_trial_arrays_aligned():
    t2 = np.linspace(-0.9, 0.9, 6)
    t1 = np.full(6, 0.4)
    traj = make_traj(t1, t2, slope=0.5)
    traj["targets0"] = np.zeros((6, 3))  # per-trial (N, K) with N != T
    mask = np.array([True, False, True, False, True, False])
    sub = metrics._subset(traj, mask)
    assert sub["v"].shape == (60, 3)
    assert sub["acq_step"].shape == (3, 3)
    assert sub["targets0"].shape == (3, 3)
    assert sub["go_step"].shape == (3,)


def test_cv_ridge_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 40))
    w = rng.standard_normal(40)
    assert metrics._cv_ridge_r2(X, X @ w) > 0.95
    assert metrics._cv_ridge_r2(X, rng.standard_normal(120)) < 0.3


def test_reaction_time_decreases_with_a_planted_effect():
    """RT extraction should track a real difference across delay conditions."""
    n, T = 40, 80
    v = np.zeros((T, n))
    delays = np.r_[np.full(20, 10), np.full(20, 40)]
    onsets = np.r_[np.full(20, 12), np.full(20, 4)]  # long delay -> earlier onset
    acq = np.zeros(n, int)
    for i in range(n):
        s = delays[i] + onsets[i]
        v[s:s + 20, i] = np.linspace(0.1, 1.0, 20)
        acq[i] = s + 19
    acq_step = np.full((n, 3), -1)
    acq_step[:, 0] = acq
    traj = {"v": v, "x": np.cumsum(v, 0) * 0.02, "acq_step": acq_step,
            "go_step": delays, "dt": 0.02, "alive": np.ones((T, n), bool),
            "end_step": np.full(n, T - 1)}
    rt = metrics.reaction_times(traj, {"delay_steps": delays})
    by = {r["delay_s"]: r["rt_mean"] for r in rt["by_delay"]}
    assert by[0.2] > by[0.8], "longer hold must yield shorter measured RT here"


# ------------------------------------------------------- the competence gate
def make_outcome_traj(n_targets, success, wall_steps=None):
    """Minimal rollout record carrying only what ``outcome_breakdown`` reads."""
    n_targets = np.asarray(n_targets, dtype=np.int64)
    success = np.asarray(success, dtype=bool)
    n, K = len(n_targets), 3
    # Acquired every target on success, only the first otherwise.
    acq = np.full((n, K), -1, dtype=np.int64)
    for i, (k, ok) in enumerate(zip(n_targets, success)):
        acq[i, : (k if ok else 1)] = 1
    return {
        "n_targets": n_targets,
        "success": success,
        "acq_step": acq,
        "false_start": np.zeros(n, dtype=bool),
        "timeout": ~success,
        "wall_steps": (np.zeros(n, dtype=np.int64) if wall_steps is None
                       else np.asarray(wall_steps, dtype=np.int64)),
    }


def test_competence_gate_flags_a_single_reach_policy():
    """The exact signature of the failed ablation arms: every single-target trial
    solved, every multi-target trial failed.  Overall success looks like 'partial
    competence'; the split shows there is no second movement at all."""
    n_targets = np.array([1] * 20 + [2] * 30 + [3] * 50)
    success = n_targets == 1
    out = probes.outcome_breakdown(make_outcome_traj(n_targets, success))
    assert out["success_rate"] == pytest.approx(0.20)
    assert out["success_by_n_targets"] == {1: 1.0, 2: 0.0, 3: 0.0}
    assert out["success_multi_target"] == 0.0
    assert out["competent"] is False


def test_competence_gate_passes_a_working_agent():
    n_targets = np.array([1] * 20 + [2] * 30 + [3] * 50)
    success = np.ones(100, dtype=bool)
    success[:3] = False
    out = probes.outcome_breakdown(make_outcome_traj(n_targets, success))
    assert out["competent"] is True
    assert out["success_multi_target"] == pytest.approx(1.0)


def test_competence_floor_is_configurable():
    n_targets = np.full(10, 2)
    success = np.array([True] * 5 + [False] * 5)
    assert probes.outcome_breakdown(make_outcome_traj(n_targets, success),
                                    floor=0.4)["competent"] is True
    assert probes.outcome_breakdown(make_outcome_traj(n_targets, success),
                                    floor=0.9)["competent"] is False


def test_wall_contact_is_reported():
    n_targets = np.full(4, 1)
    out = probes.outcome_breakdown(
        make_outcome_traj(n_targets, np.ones(4, bool), wall_steps=[0, 0, 5, 90]))
    assert out["wall_contact_rate"] == pytest.approx(0.5)
    assert out["mean_wall_steps"] == pytest.approx(23.75)
