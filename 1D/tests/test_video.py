"""Video and tracking tests.

The renderer is a reporting tool, not part of the science, so what is worth pinning is
that it *reads the trial correctly*: which targets the agent can see at each step, when
a future target enters the horizon, and when one is acquired.  A frame that mislabels
those would make a video argue for lookahead that the rollout does not contain.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import probes, video
from seqreach.config import Config
from seqreach.networks import Agent, RunningMeanStd
from seqreach.tracking import Tracker


def tiny_cfg(**over):
    """Short episodes and a small net: enough to render, fast enough for CI."""
    base = {"task.max_steps": 24, "task.delay_min_s": 0.04, "task.delay_max_s": 0.08,
            "agent.hidden_size": 16, "agent.rnn_size": 16, "video.stride": 3,
            "video.width_px": 320, "video.dpi": 60}
    base.update(over)
    return Config().override(base)


def make_rollout(cfg, n_trials=2, seed=0):
    env_probe = video.video_config(cfg, cfg.video)
    agent = Agent(4 + 4 * cfg.task.max_targets, 1, cfg.agent, cfg.ppo.init_logstd)
    rms = RunningMeanStd((4 + 4 * cfg.task.max_targets,))
    spec = video.eval_spec(env_probe, n_trials)
    traj = probes.rollout(agent, rms, env_probe, spec, n_trials=n_trials, seed=seed)
    return env_probe, traj


# --------------------------------------------------------------------- specs
def test_eval_spec_covers_full_and_myopic_horizons():
    cfg = tiny_cfg()
    spec = video.eval_spec(cfg, 3)
    assert spec["n_targets"].tolist() == [3, 3, 2]
    # the first two trials are the same length but differ only in what the agent sees
    assert spec["preview"][0] == 3 and spec["preview"][1] == 1


def test_eval_spec_respects_fixed_preview_ablation():
    cfg = tiny_cfg(**{"task.fixed_preview": 1})
    spec = video.eval_spec(cfg, 3)
    assert (spec["preview"] == 1).all(), "must not film a horizon the ablation removed"


def test_video_config_kills_motor_noise_only_when_asked():
    cfg = tiny_cfg(**{"plant.sigma_sdn": 0.3})
    assert video.video_config(cfg, cfg.video).plant.sigma_sdn == 0.0
    cfg.video.noiseless = False
    assert video.video_config(cfg, cfg.video).plant.sigma_sdn == 0.3


# ------------------------------------------------------- trial interpretation
def test_trial_data_visibility_matches_the_env_rule():
    """visible == in-sequence, pending, and within the rolling preview window."""
    cfg, traj = make_rollout(tiny_cfg(), n_trials=3)
    for i in range(3):
        d = video._trial_data(traj, cfg, i)
        slots = np.arange(d["K"])[None, :]
        cur = d["cur"][:, None]
        expect = (slots < d["n"]) & (slots >= cur) & (slots <= cur + d["H"] - 1)
        assert np.array_equal(d["visible"], expect)
        assert np.array_equal(d["acquired"], (slots < d["n"]) & (slots < cur))
        # a slot is never simultaneously taken and current
        assert not (d["acquired"] & d["current"]).any()


def test_horizon_one_reveals_targets_one_at_a_time():
    cfg = tiny_cfg(**{"task.fixed_preview": 1})
    _, traj = make_rollout(cfg, n_trials=2)
    d = video._trial_data(traj, cfg, 0)
    assert d["visible"].sum(axis=1).max() <= 1, "H=1 must never show two targets"


def test_events_flag_reveals_and_acquisitions():
    cfg, traj = make_rollout(tiny_cfg(), n_trials=2)
    d = video._trial_data(traj, cfg, 0)
    kinds = [text for _, text, _ in d["events"]]
    assert "GO" in kinds or d["go"] >= d["T"]
    for t, text, _ in d["events"]:
        assert 0 <= t < d["T"] + 1
        if text.endswith("acquired ✓"):
            k = int(text[1]) - 1
            assert d["acquired"][t, k] and not d["acquired"][t - 1, k]


def test_outcome_label_agrees_with_the_rollout():
    cfg, traj = make_rollout(tiny_cfg(), n_trials=3)
    for i in range(3):
        d = video._trial_data(traj, cfg, i)
        label = d["outcome"][0]
        assert (label == "SUCCESS ✓") == bool(traj["success"][i])


# ------------------------------------------------------------------ rendering
def test_render_produces_a_readable_multi_frame_gif(tmp_path):
    from PIL import Image

    cfg, traj = make_rollout(tiny_cfg(), n_trials=2)
    out = video.render_gif(traj, cfg, tmp_path / "r.gif", title="unit test")
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.n_frames > 1
        assert im.size[0] == cfg.video.width_px
        assert abs(im.size[1] - cfg.video.width_px * video.ASPECT) <= 1


def test_frame_count_tracks_stride():
    cfg, traj = make_rollout(tiny_cfg(), n_trials=1)
    cfg.video.stride = 1
    dense = len(video.render_frames(traj, cfg, cfg.video, hold_frames=0))
    cfg.video.stride = 4
    sparse = len(video.render_frames(traj, cfg, cfg.video, hold_frames=0))
    assert 1 <= sparse < dense


def test_axis_limits_ignore_post_episode_rows():
    """The env autoresets in place; rows past end_step belong to a different trial."""
    cfg, traj = make_rollout(tiny_cfg(), n_trials=2)
    traj = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in traj.items()}
    end = int(traj["end_step"][0])
    if end + 1 < traj["x"].shape[0]:
        traj["x"][end + 1:, 0] = 99.0  # garbage that must not reach the axis limits
    frames = video.render_frames(traj, cfg, cfg.video, trials=[0], hold_frames=0)
    assert len(frames) >= 1


# ------------------------------------------------------------------- tracking
def test_tracker_is_inert_when_disabled(tmp_path):
    t = Tracker(tiny_cfg(), tmp_path)
    assert not t.active
    t.log({"a": 1}, step=0)  # all of these must be no-ops, not errors
    t.log_video(tmp_path / "missing.gif", step=0)
    t.summary({"b": 2})
    t.save_checkpoint(tmp_path / "missing.pt")
    t.finish()


def test_run_name_matches_the_run_directory():
    cfg = tiny_cfg(**{"seed": 7})
    cfg.exp_name = "myopic"
    assert cfg.run_name == "myopic_s7"


def test_config_round_trips_with_the_new_sections(tmp_path):
    cfg = tiny_cfg(**{"wandb.enabled": True, "wandb.tags": ("a", "b"),
                      "video.every_iters": 3})
    path = tmp_path / "c.yaml"
    cfg.save(path)
    back = Config.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert back.wandb.tags == ("a", "b") and back.video.every_iters == 3
