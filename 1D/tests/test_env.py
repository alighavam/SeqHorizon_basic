"""Environment correctness tests.

These guard the parts of the testbed that a wrong result would be blamed on later:
plant integration, the acquisition/ordering rules, the hold-and-go-cue logic, preview
masking (which *is* the causal manipulation), and jump application.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach.config import Config
from seqreach.env import JUMP_CURRENT, JUMP_FUTURE, BatchedSeqReach1D, GymSeqReach1D


def quiet_cfg(**over):
    """Deterministic config: no motor noise, no delay, no shaping, no jumps."""
    base = {
        "plant.sigma_sdn": 0.0,
        "plant.tau_act": 0.0,
        "task.use_delay": False,
        "task.p_jump": 0.0,
        "reward.shaping": "none",
    }
    base.update(over)
    return Config().override(base)


# ------------------------------------------------------------------ dynamics
def test_double_integrator_matches_closed_form():
    """u constant, no damping, no lag -> x(t) = 0.5*a*t^2 under semi-implicit Euler."""
    cfg = quiet_cfg(**{"plant.damping": 0.0})
    env = BatchedSeqReach1D(cfg, num_envs=4, seed=0)
    env.reset()
    dt, acc, n = cfg.plant.dt, 3.0, 25
    for _ in range(n):
        env.step(np.full(4, acc))
    # semi-implicit Euler: v_k = k*a*dt, x_n = sum_{k=1..n} v_k*dt = a*dt^2*n(n+1)/2
    expect = acc * dt**2 * n * (n + 1) / 2
    assert np.allclose(env.x, expect, atol=1e-9)
    assert np.allclose(env.v, acc * dt * n, atol=1e-9)


def test_damping_decays_velocity():
    cfg = quiet_cfg(**{"plant.damping": 2.0})
    env = BatchedSeqReach1D(cfg, num_envs=2, seed=0)
    # Unreachable target (|x| is clipped at x_limit) so the episode cannot end and
    # autoreset cannot wipe the state we are measuring.
    env.reset({"n_targets": 1, "targets": np.array([[5.0, 0.0, 0.0]]), "preview": 1})
    env.v[:] = 1.0
    for _ in range(50):
        env.step(np.zeros(2))
    assert np.all(env.v < 0.4) and np.all(env.v > 0.0)


def test_activation_lag_slows_force_onset():
    slow = BatchedSeqReach1D(quiet_cfg(**{"plant.tau_act": 0.1}), 1, seed=0)
    fast = BatchedSeqReach1D(quiet_cfg(**{"plant.tau_act": 0.0}), 1, seed=0)
    slow.reset(), fast.reset()
    for _ in range(5):
        slow.step(np.array([8.0]))
        fast.step(np.array([8.0]))
    assert slow.x[0] < fast.x[0]  # lagged force produces less displacement


def test_force_is_clipped_to_u_max():
    cfg = quiet_cfg()
    a = BatchedSeqReach1D(cfg, 1, seed=0)
    b = BatchedSeqReach1D(cfg, 1, seed=0)
    a.reset(), b.reset()
    a.step(np.array([cfg.plant.u_max]))
    b.step(np.array([cfg.plant.u_max * 100]))
    assert np.isclose(a.x[0], b.x[0])


def test_signal_dependent_noise_scales_with_action():
    """SDN variance must grow with |u| -- it is off at u=0 by construction."""
    cfg = Config().override({"task.use_delay": False, "plant.sigma_sdn": 0.2,
                             "reward.shaping": "none", "task.p_jump": 0.0})
    spread = []
    for u in (0.0, 8.0):
        env = BatchedSeqReach1D(cfg, num_envs=512, seed=3)
        env.reset()
        env.step(np.full(512, u))
        spread.append(env.v.std())
    assert spread[0] == 0.0 < spread[1]


# --------------------------------------------------------------- acquisition
def _drive_to(env, pos, steps=400):
    """Crude PD controller, enough to exercise the acquisition logic."""
    for _ in range(steps):
        u = 30.0 * (pos - env.x) - 8.0 * env.v
        out = env.step(np.clip(u, -8, 8))
        if out[2].any() or out[3].any():
            return out
    return out


def test_targets_must_be_acquired_in_order():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 2, "targets": np.array([[0.5, -0.5, 0.0]]), "preview": 2})
    _drive_to(env, -0.5, steps=60)  # sit on T2 while T1 is still current
    assert env.cur[0] == 0, "reaching T2 before T1 must not count"
    assert env.ep_hits[0] == 0


def test_final_target_requires_dwell():
    """Acquisition terminates the episode, so assert on the termination signal:
    reading env.cur afterwards would see the autoreset trial, not the finished one."""
    cfg = quiet_cfg(**{"task.dwell_final": 25})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[0.4, 0.0, 0.0]]), "preview": 1})
    fired = []
    for _ in range(40):
        _, _, term, _, _ = env.step(np.array([0.0]))
        fired.append(bool(term[0]))
        env.x[:] = 0.4  # park inside tolerance; v stays 0 so physics holds it there
    assert not any(fired[:25]), "must not acquire before the dwell requirement is met"
    assert sum(fired) == 1 and fired[25], "acquire on exactly the 25th step inside"


def test_intermediate_target_is_pass_through():
    """dwell_inter=1: crossing the tolerance band once is enough (no stopping)."""
    cfg = quiet_cfg(**{"task.dwell_inter": 1})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 2, "targets": np.array([[0.3, 0.9, 0.0]]), "preview": 2})
    for _ in range(200):
        _, _, term, trunc, _ = env.step(np.array([8.0]))  # blast straight through
        if env.cur[0] >= 1:
            break
    assert env.cur[0] == 1
    assert env.v[0] > 0.5, "should still be moving when an intermediate target is taken"


def test_completing_sequence_terminates_with_reward():
    cfg = quiet_cfg(**{"task.dwell_final": 3})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[0.3, 0.0, 0.0]]), "preview": 1})
    _, rew, term, _, info = _drive_to(env, 0.3)
    assert term[0] and rew[0] > cfg.reward.r_final - 1
    assert info["episode"]["success"][0]


# ------------------------------------------------------------- hold / go cue
def test_moving_before_go_cue_aborts_trial():
    cfg = Config().override({"plant.sigma_sdn": 0.0, "reward.shaping": "none",
                             "task.p_jump": 0.0})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[0.8, 0.0, 0.0]]),
               "delay_steps": 100, "preview": 1})
    term = False
    for _ in range(60):
        _, rew, t_, _, _ = env.step(np.array([8.0]))
        if t_[0]:
            term = True
            assert rew[0] < 0
            break
    assert term, "leaving the hold window before the go cue must abort"


def test_no_acquisition_before_go_cue():
    cfg = quiet_cfg(**{"task.use_delay": True})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[0.02, 0.0, 0.0]]),
               "delay_steps": 30, "preview": 1})
    fired = []
    for _ in range(60):  # target sits inside tolerance of the start position
        _, _, term, _, _ = env.step(np.zeros(1))
        fired.append(bool(term[0]))
    assert not any(fired[:30]), "targets must not be acquirable during the delay"
    assert fired[30 + cfg.task.dwell_final - 1], "acquirable once the go cue fires"


def test_go_flag_observation_flips_at_go_step():
    cfg = quiet_cfg(**{"task.use_delay": True})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    obs = env.reset({"n_targets": 1, "targets": np.array([[0.5, 0.0, 0.0]]),
                     "delay_steps": 10, "preview": 1})
    assert obs[0, 3] == 0.0
    for _ in range(10):
        obs, *_ = env.step(np.zeros(1))
    assert obs[0, 3] == 1.0


# ------------------------------------------------------------------- preview
def test_preview_window_masks_future_targets():
    """The causal manipulation: H=1 must hide T2 until T1 is acquired."""
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 2, seed=0)
    tg = np.array([[0.3, 0.7, 0.0], [0.3, 0.7, 0.0]])
    obs = env.reset({"n_targets": 2, "targets": tg, "preview": np.array([1, 2])})
    slot1 = 4 + 4 * 1
    assert obs[0, slot1] == 0.0, "H=1 must hide the second target"
    assert obs[1, slot1] == 1.0, "H=2 must show it"
    assert obs[0, slot1 + 3] == 0.0, "hidden target's relative position must read zero"

    for _ in range(200):
        env.step(np.full(2, 8.0))
        if env.cur[0] >= 1:
            break
    obs = env._obs()
    assert obs[0, slot1] == 1.0, "T2 must be revealed once T1 is acquired"


def test_observation_is_egocentric_and_finite():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 8, seed=1)
    obs = env.reset()
    assert obs.shape == (8, env.obs_dim) and np.isfinite(obs).all()
    for _ in range(300):
        obs, r, *_ = env.step(np.random.uniform(-8, 8, 8))
        assert np.isfinite(obs).all() and np.isfinite(r).all()


# --------------------------------------------------------------------- jumps
def test_jump_current_moves_the_active_target():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 2, "targets": np.array([[0.3, 0.7, 0.0]]), "preview": 2,
               "jump_kind": JUMP_CURRENT, "jump_step": 5, "jump_delta": 0.3})
    before = env.targets[0, 0]
    for _ in range(6):
        env.step(np.zeros(1))
    assert env.jump_applied[0] and env.jump_slot[0] == 0
    assert np.isclose(env.targets[0, 0], before + 0.3)


def test_jump_future_leaves_current_target_untouched():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 2, "targets": np.array([[0.3, 0.5, 0.0]]), "preview": 2,
               "jump_kind": JUMP_FUTURE, "jump_step": 5, "jump_delta": 0.3})
    t0 = env.targets[0, 0]
    for _ in range(6):
        env.step(np.zeros(1))
    assert env.jump_applied[0] and env.jump_slot[0] == 1
    assert np.isclose(env.targets[0, 0], t0), "T1 must be unchanged by a future-jump"
    assert np.isclose(env.targets[0, 1], 0.8)


def test_jump_is_skipped_when_target_not_visible():
    """A hidden target cannot 'jump' -- that would be resampling, not a perturbation."""
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 2, "targets": np.array([[0.3, 0.5, 0.0]]), "preview": 1,
               "jump_kind": JUMP_FUTURE, "jump_step": 5, "jump_delta": 0.3})
    for _ in range(6):
        env.step(np.zeros(1))
    assert not env.jump_applied[0]
    assert np.isclose(env.targets[0, 1], 0.5)


# ---------------------------------------------------------- plumbing / parity
def test_autoreset_is_same_step_and_reports_final_obs():
    cfg = quiet_cfg(**{"task.max_steps": 10})
    env = BatchedSeqReach1D(cfg, 3, seed=0)
    env.reset()
    for _ in range(9):
        _, _, term, trunc, info = env.step(np.zeros(3))
    obs, _, term, trunc, info = env.step(np.zeros(3))
    assert trunc.all() and not term.any()
    assert info["final_obs"].shape == (3, env.obs_dim)
    assert np.all(env.t == 0), "envs must be reset in the same step (CleanRL semantics)"


def test_seeding_is_deterministic():
    cfg = Config()
    a, b = BatchedSeqReach1D(cfg, 6, seed=42), BatchedSeqReach1D(cfg, 6, seed=42)
    oa, ob = a.reset(), b.reset()
    assert np.array_equal(oa, ob)
    for _ in range(50):
        u = np.random.RandomState(0).uniform(-8, 8, 6)
        oa, ra, *_ = a.step(u)
        ob, rb, *_ = b.step(u)
        assert np.array_equal(oa, ob) and np.array_equal(ra, rb)


def test_gym_adapter_matches_batched_env():
    cfg = quiet_cfg()
    g = GymSeqReach1D(cfg, seed=7)
    b = BatchedSeqReach1D(cfg, 1, seed=7)
    o1, _ = g.reset()
    o2 = b.reset()
    assert np.allclose(o1, o2[0])
    for i in range(120):
        u = np.array([4.0 * np.sin(i / 7)])
        o1, r1, t1, c1, _ = g.step(u)
        o2, r2, t2, c2, _ = b.step(u)
        assert np.allclose(o1, o2[0]) and np.isclose(r1, r2[0])
        assert t1 == t2[0] and c1 == c2[0]


def test_sensory_delay_returns_stale_state():
    cfg = quiet_cfg(**{"plant.sensory_delay": 5})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[0.9, 0.0, 0.0]]), "preview": 1})
    for _ in range(4):
        obs, *_ = env.step(np.array([8.0]))
    assert env.x[0] > 0.0
    assert obs[0, 0] == 0.0, "position feedback must still be at the pre-movement value"
    for _ in range(4):
        obs, *_ = env.step(np.array([8.0]))
    assert obs[0, 0] > 0.0


def test_pbrs_shaping_is_zero_sum_over_a_completed_episode():
    """Telescoping check: PBRS must not change the total return of a fixed trajectory
    beyond the -Phi(s0) boundary term, which is what makes it policy-invariant."""
    ret = {}
    for shaping in ("none", "pbrs"):
        cfg = quiet_cfg(**{"reward.shaping": shaping, "task.dwell_final": 3})
        env = BatchedSeqReach1D(cfg, 1, seed=0)
        env.reset({"n_targets": 1, "targets": np.array([[0.3, 0.0, 0.0]]), "preview": 1})
        total, gamma, disc = 0.0, cfg.ppo.gamma, 1.0
        for _ in range(400):
            _, r, term, trunc, _ = env.step(np.clip(30 * (0.3 - env.x) - 8 * env.v, -8, 8))
            total += disc * r[0]
            disc *= gamma
            if term[0] or trunc[0]:
                break
        ret[shaping] = total
    phi_s0 = -cfg.reward.shaping_coef * 0.3  # Phi at the start state
    assert np.isclose(ret["pbrs"] - ret["none"], -phi_s0, atol=1e-6)


# ------------------------------------------------------- workspace boundary
def test_boundary_stops_the_mass_instead_of_letting_it_slide():
    """Contact is inelastic: position clamps *and* velocity is zeroed.

    The old behaviour clipped position only, so the mass kept its velocity while pinned
    against the limit and the boundary became a free place to rest.
    """
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({"n_targets": 1, "targets": np.array([[-0.5, 0.0, 0.0]]), "preview": 1})
    for _ in range(200):
        env.step(np.array([cfg.plant.u_max]))
        if abs(env.x[0]) >= cfg.plant.x_limit:
            break
    assert np.isclose(env.x[0], cfg.plant.x_limit)
    assert env.v[0] == 0.0


def test_boundary_contact_is_charged_for():
    """Parking at the limit must cost strictly more than the same trial without it.

    This is the fix for the failure mode that trapped most of the first ablation grid:
    with w_wall = 0 the mass sits at the limit for free while the shaping term pays it.
    """
    spec = {"n_targets": 1, "targets": np.array([[-0.5, 0.0, 0.0]]), "preview": 1}
    returns = {}
    for w_wall in (0.0, 0.1):
        cfg = quiet_cfg(**{"reward.w_wall": w_wall})
        env = BatchedSeqReach1D(cfg, 1, seed=0)
        env.reset(spec)
        total = 0.0
        for _ in range(100):  # drive away from the target, into the far boundary
            _, r, term, trunc, _ = env.step(np.array([cfg.plant.u_max]))
            total += r[0]
            if term[0] or trunc[0]:
                break
        returns[w_wall] = total
    assert returns[0.1] < returns[0.0]


def test_boundary_is_unreachable_for_an_on_task_policy():
    """The boundary rule must be inert for a competent agent, or it is not a free fix.

    Targets live within +-target_range, so a policy that goes where it is told never
    makes contact and never pays w_wall.
    """
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 64, seed=3)
    env.reset()
    for _ in range(cfg.task.max_steps):
        cur = np.take_along_axis(env.targets, env.cur[:, None].clip(0, env.K - 1), 1)[:, 0]
        env.step(np.clip(-60.0 * (env.x - cur) - 15.0 * env.v, -8, 8))
    assert env.ep_wall.sum() == 0


# ------------------------------------------------------------- curriculum
def test_curriculum_interpolates_from_start_to_final_weights():
    cfg = quiet_cfg(**{"task.curriculum_steps": 1000,
                       "task.curriculum_start_weights": (1.0, 0.0, 0.0),
                       "task.n_target_weights": (0.2, 0.3, 0.5)})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    assert np.allclose(env.curriculum_weights(0), [1.0, 0.0, 0.0])
    assert np.allclose(env.curriculum_weights(500), [0.6, 0.15, 0.25])
    assert np.allclose(env.curriculum_weights(1000), [0.2, 0.3, 0.5])
    # Held, not extrapolated, past the end of the schedule.
    assert np.allclose(env.curriculum_weights(10_000), [0.2, 0.3, 0.5])


def test_curriculum_disabled_gives_final_weights_throughout():
    cfg = quiet_cfg(**{"task.curriculum_steps": 0})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    assert np.allclose(env.curriculum_weights(0), cfg.task.n_target_weights)


def test_a_fresh_env_always_samples_the_final_mixture():
    """Probes and evaluation build their own env, and must never inherit a curriculum
    stage -- the task being measured is the task as specified."""
    cfg = quiet_cfg(**{"task.curriculum_steps": 10_000_000})
    env = BatchedSeqReach1D(cfg, 4096, seed=0)
    env.reset()
    assert env.n_targets.max() == cfg.task.max_targets
    assert np.isclose(env.n_targets.mean(), 2.3, atol=0.05)  # E[n] under (0.2, 0.3, 0.5)


def test_set_n_target_weights_changes_sampling():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 512, seed=0)
    env.set_n_target_weights([1.0, 0.0, 0.0])
    env.reset()
    assert (env.n_targets == 1).all()
    env.set_n_target_weights([0.0, 0.0, 1.0])
    env.reset()
    assert (env.n_targets == 3).all()


def test_set_n_target_weights_rejects_bad_input():
    env = BatchedSeqReach1D(quiet_cfg(), 1, seed=0)
    with pytest.raises(ValueError):
        env.set_n_target_weights([1.0, 0.0])          # wrong length
    with pytest.raises(ValueError):
        env.set_n_target_weights([0.0, 0.0, 0.0])     # sums to zero
    with pytest.raises(ValueError):
        env.set_n_target_weights([-1.0, 1.0, 1.0])    # negative


# ------------------------------------------------------------- return split
def test_task_return_excludes_shaping():
    """`r_task` must be identical with and without PBRS; `r` must not be.

    The gamma=0.90 arm of the first grid logged the highest episodic return in the whole
    campaign at 0.000 success, purely on shaping income.  This is the metric that would
    have caught it.
    """
    spec = {"n_targets": 1, "targets": np.array([[0.3, 0.0, 0.0]]), "preview": 1}
    got = {}
    for shaping in ("none", "pbrs"):
        cfg = quiet_cfg(**{"reward.shaping": shaping, "task.dwell_final": 3})
        env = BatchedSeqReach1D(cfg, 1, seed=0)
        env.reset(spec)
        for _ in range(400):
            _, _, term, trunc, info = env.step(
                np.clip(30 * (0.3 - env.x) - 8 * env.v, -8, 8))
            if term[0] or trunc[0]:
                got[shaping] = (info["episode"]["r"][0], info["episode"]["r_task"][0])
                break
    assert np.isclose(got["none"][1], got["pbrs"][1])      # task return unaffected
    assert not np.isclose(got["none"][0], got["pbrs"][0])  # total return is
