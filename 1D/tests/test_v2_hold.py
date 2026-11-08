"""Tests for the v2 hold-period options: random start, catch trials, cue delay, start
observation.  Every option is off by default, so the first test pins the v1 layout."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach.config import Config
from seqreach.env import BatchedSeqReach1D
from seqreach import probes


def quiet_cfg(**over):
    base = {"plant.sigma_sdn": 0.0, "plant.tau_act": 0.0, "task.use_delay": True,
            "task.p_jump": 0.0, "reward.shaping": "none"}
    base.update(over)
    return Config().override(base)


ONE = {"n_targets": 1, "targets": np.array([[0.5, 0.0, 0.0]]), "preview": 1}


def test_defaults_keep_v1_layout():
    env = BatchedSeqReach1D(Config(), 64, seed=0)
    assert env.obs_dim == 16
    assert np.all(env.x0 == 0.0) and np.all(env.x == 0.0)
    assert not env.catch.any()


def test_random_start_in_range_and_hold_relative_to_it():
    cfg = quiet_cfg(**{"task.start_range": 0.5})
    env = BatchedSeqReach1D(cfg, 512, seed=1)
    env.reset()
    assert np.all(np.abs(env.x0) <= 0.5) and env.x0.std() > 0.2
    assert np.allclose(env.x, env.x0)
    # targets keep min separation from the start
    assert np.all(np.abs(env.targets[:, 0] - env.x0) >= cfg.task.min_separation - 1e-12)
    # holding still at an off-centre start is free and not a false start
    _, rew, term, _, info = env.step(np.zeros(512))
    assert not term.any()
    assert np.allclose(rew, 0.0)


def test_spec_start_overrides_start_range():
    cfg = quiet_cfg(**{"task.start_range": 0.5})
    env = BatchedSeqReach1D(cfg, 3, seed=0)
    env.reset({**ONE, "start_x": np.array([-0.3, 0.0, 0.2]), "delay_steps": 10})
    assert np.allclose(env.x0, [-0.3, 0.0, 0.2])


def test_catch_trial_never_shows_go_and_pays_on_hold():
    cfg = quiet_cfg()
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    obs = env.reset({**ONE, "catch": True})
    end = int(env.catch_end[0])
    assert cfg.task.delay_min_s / cfg.plant.dt <= end <= cfg.task.catch_max_s / cfg.plant.dt
    total, last = 0.0, None
    for t in range(end):
        assert obs[0, 3] == 0.0, "go flag must never fire on a catch trial"
        obs, rew, term, trunc, info = env.step(np.zeros(1))
        total += float(rew[0])
        last = info
        if term[0]:
            break
    assert term[0] and t == end - 1
    ep = last["episode"]
    assert ep["catch"][0] and ep["catch_held"][0] and not ep["success"][0]
    assert np.isclose(total, cfg.reward.r_catch)


def test_catch_trial_false_start_is_not_paid():
    cfg = quiet_cfg(**{"plant.damping": 0.0})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({**ONE, "catch": True})
    for _ in range(200):
        _, rew, term, _, info = env.step(np.full(1, 8.0))
        if term[0]:
            break
    ep = info["episode"]
    assert ep["false_start"][0] and not ep["catch_held"][0]
    assert rew[0] < 0


def test_p_catch_rate():
    env = BatchedSeqReach1D(quiet_cfg(**{"task.p_catch": 0.25}), 4000, seed=0)
    assert abs(env.catch.mean() - 0.25) < 0.03


def test_cue_delay_shifts_go_flag():
    cfg = quiet_cfg(**{"plant.cue_delay": 3})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    obs = env.reset({**ONE, "delay_steps": 10})
    flags = [obs[0, 3]]
    for _ in range(16):
        obs, *_ = env.step(np.zeros(1))
        flags.append(obs[0, 3])
    first = int(np.argmax(np.array(flags) > 0.5))
    assert first == 13, "go flag must appear cue_delay steps after the go step"


def test_cue_delay_does_not_delay_acquisition_rules():
    """The cue is late to the agent, not to the task: targets are acquirable from go."""
    cfg = quiet_cfg(**{"plant.cue_delay": 3, "plant.damping": 0.0})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    env.reset({**ONE, "targets": np.array([[0.0, 0.0, 0.0]]), "start_x": 0.0,
               "delay_steps": 5})
    # target on the start: acquisition only needs the dwell after the true go step
    fired = []
    for _ in range(5 + cfg.task.dwell_final + 2):
        _, _, term, _, _ = env.step(np.zeros(1))
        fired.append(bool(term[0]))
    assert fired.index(True) == 5 + cfg.task.dwell_final - 1


def test_start_observation_appended_and_gated_by_perceived_go():
    cfg = quiet_cfg(**{"task.obs_start": True, "plant.cue_delay": 2})
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    assert env.obs_dim == 17
    obs = env.reset({**ONE, "start_x": -0.3, "delay_steps": 4})
    assert np.isclose(obs[0, -1], 0.0)  # at the start, offset to start is zero
    seen = []
    for _ in range(8):
        obs, *_ = env.step(np.full(1, 0.5))  # drift a little
        seen.append((obs[0, 3], obs[0, -1]))
    for go, off in seen:
        if go > 0.5:
            assert off == 0.0, "start offset must vanish once the cue is perceived"
    assert any(go < 0.5 and off < 0 for go, off in seen), "offset points back to the start"


def test_probe_config_removes_v2_sampling():
    cfg = Config().override({"task.p_catch": 0.3, "task.start_range": 0.5})
    pc = probes.probe_config(cfg)
    assert pc.task.p_catch == 0.0 and pc.task.start_range == 0.0
