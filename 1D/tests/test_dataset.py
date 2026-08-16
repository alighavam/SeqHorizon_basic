"""Tests for the experiment/dataset layer.

The point of these tables is that an analysis done on the CSV agrees with an analysis
done on the raw rollout.  So the tests check correspondence, not just shape: every
number that lands in a column is compared against the trajectory it came from.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import dataset, experiments, probes
from seqreach.config import Config
from seqreach.networks import RunningMeanStd
from seqreach.env import BatchedSeqReach1D


class ScriptedAgent:
    """A PD controller wearing the ``Agent`` interface.

    An *untrained* network never reaches a target, so every event column would be NaN and
    the tests would pass vacuously.  This stand-in actually completes trials -- targets
    get entered and acquired, trials terminate early -- which is what makes the
    correspondence checks below meaningful.  It reads the observation the same way the
    real policy does, so the observation layout is exercised too.
    """

    recurrent = False

    def initial_state(self, num_envs, device):
        return torch.zeros(1, num_envs, 1, device=device)

    def get_action_and_value(self, x, state, done, action=None, deterministic=False):
        obs = x.numpy()
        v = obs[:, 1] * 2.0                      # de-scale: _V_SCALE
        go = obs[:, 3] > 0.5
        # Relative offset of whichever slot is flagged current *and* visible.
        rel = np.zeros(len(obs))
        n_slots = (obs.shape[1] - 4) // 4
        for k in range(n_slots):
            b = 4 + 4 * k
            sel = (obs[:, b + 2] > 0.5) & (obs[:, b] > 0.5)
            rel = np.where(sel, obs[:, b + 3], rel)
        u = np.clip(6.0 * rel - 1.6 * v, -1.0, 1.0) * go
        act = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
        zero = torch.zeros(len(obs))
        return act, zero, zero, zero, state


@pytest.fixture(scope="module")
def rollout():
    cfg = probes.probe_config(Config(), noiseless=True)
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    design = experiments.build("preview_sweep", cfg, n_t2=7)
    traj = probes.rollout(ScriptedAgent(), RunningMeanStd((env.obs_dim,)), cfg,
                          design.spec, n_trials=design.n_trials, seed=3)
    assert traj["success"].mean() > 0.5, "fixture controller must solve most trials"
    return traj, design, cfg


# ------------------------------------------------------------------ registry
def test_every_registered_experiment_builds_and_is_consistent():
    cfg = Config()
    for name in experiments.EXPERIMENTS:
        d = experiments.build(name, cfg)
        assert d.n_trials > 0, name
        assert d.name == name
        n = d.n_trials
        for key, val in d.spec.items():
            assert len(val) == n, f"{name}: spec['{key}'] has wrong length"
        for key, val in d.factors.items():
            assert len(val) == n, f"{name}: factor '{key}' has wrong length"
        # Targets must be acquirable: within range and correctly counted.
        assert (d.spec["n_targets"] >= 1).all()
        assert (d.spec["n_targets"] <= cfg.task.max_targets).all()
        assert (d.spec["preview"] >= 1).all()


def test_unknown_experiment_names_the_alternatives():
    with pytest.raises(KeyError, match="preview_sweep"):
        experiments.build("does_not_exist", Config())


def test_experiment_params_are_forwarded():
    d = experiments.build("preview_sweep", Config(), n_t2=5, both_directions=False)
    assert d.params["n_t2"] == 5
    assert set(np.unique(d.factors["direction"])) == {1.0}


def test_trial_list_rejects_impossible_target_counts():
    cfg = Config()
    tl = experiments.TrialList(cfg)
    with pytest.raises(ValueError):
        tl.add([0.1, 0.2, 0.3, 0.4])  # more than max_targets
    with pytest.raises(ValueError):
        tl.add([])


def test_add_grid_crosses_and_records_factors():
    cfg = Config()
    tl = experiments.TrialList(cfg)
    tl.add_grid(targets=[0.4, 0.5], preview=[1, 2], delay_steps=25)
    d = tl.build()
    assert d.n_trials == 2  # targets is scalar-per-trial here, preview varies
    assert set(np.unique(d.factors["preview"])) == {1, 2}


# ---------------------------------------------------------------- long table
def test_timeseries_matches_the_trajectory_sample_for_sample(rollout):
    traj, design, _ = rollout
    df = dataset.timeseries_frame(traj, design, "unit_test")
    for tn in (0, 3, design.n_trials - 1):
        sub = df[df.TN == tn].sort_values("sample")
        n = len(sub)
        assert np.allclose(sub["x"], traj["x"][:n, tn])
        assert np.allclose(sub["v"], traj["v"][:n, tn])
        assert np.allclose(sub["a"], traj["a"][:n, tn])
        assert np.allclose(sub["u"], traj["u"][:n, tn])


def test_time_columns_are_in_milliseconds_and_anchored_to_the_go_cue(rollout):
    traj, design, cfg = rollout
    df = dataset.timeseries_frame(traj, design, "unit_test")
    dt_ms = cfg.plant.dt * 1000.0
    assert np.allclose(df["t_ms"], df["sample"] * dt_ms)
    # t_from_go_ms is negative during the hold and exactly 0 on the go sample.
    for tn, go in enumerate(traj["go_step"][:5]):
        sub = df[df.TN == tn]
        row = sub[sub["sample"] == go]
        assert row["t_from_go_ms"].iloc[0] == pytest.approx(0.0)
        assert (sub[sub["sample"] < go]["t_from_go_ms"] < 0).all()
    assert (df.loc[df["go"], "t_from_go_ms"] >= 0).all()


def test_dead_samples_are_dropped_by_default(rollout):
    traj, design, _ = rollout
    kept = dataset.timeseries_frame(traj, design, "unit_test")
    everything = dataset.timeseries_frame(traj, design, "unit_test", include_dead=True)
    assert len(kept) < len(everything)
    assert "alive" not in kept.columns
    # Each trial keeps exactly the samples it was alive for.
    counts = kept.groupby("TN").size()
    for tn, n in counts.items():
        assert n == int(traj["alive"][:, tn].sum())


def test_unused_target_slots_are_nan_not_zero():
    """A 2-target trial must not look like it has a third target at the origin."""
    cfg = probes.probe_config(Config(), noiseless=True)
    tl = experiments.TrialList(cfg)
    tl.add([0.4, -0.5], preview=2)
    d = tl.build(name="two_only")
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    traj = probes.rollout(ScriptedAgent(), RunningMeanStd((env.obs_dim,)), cfg,
                          d.spec, n_trials=1, seed=0)
    ts = dataset.timeseries_frame(traj, d, "unit_test")
    tr = dataset.trial_frame(traj, d, "unit_test")
    assert ts["T3_now"].isna().all()
    assert tr["T3"].isna().all()
    assert not ts["T2_now"].isna().any()


# --------------------------------------------------------------- trial table
def test_trial_table_has_one_row_per_trial_and_carries_the_design(rollout):
    traj, design, _ = rollout
    df = dataset.trial_frame(traj, design, "unit_test")
    assert len(df) == design.n_trials
    assert list(df["TN"]) == list(range(design.n_trials))
    for key in ("direction", "t1", "t2", "d_next", "preview"):
        assert key in df.columns
    assert np.allclose(df["preview"], design.spec["preview"])


def test_entry_and_acquisition_events_match_the_trajectory(rollout):
    traj, design, cfg = rollout
    df = dataset.trial_frame(traj, design, "unit_test")
    tol, dt = cfg.task.tolerance, cfg.plant.dt
    for tn in range(design.n_trials):
        e = traj["entry_step"][tn, 0]
        if e < 0:
            assert np.isnan(df["T1_v_entry"].iloc[tn])
            assert df["T1_entry_sample"].iloc[tn] == -1
            continue
        assert df["T1_entry_sample"].iloc[tn] == e
        assert df["T1_entry_ms"].iloc[tn] == pytest.approx(e * dt * 1000)
        assert df["T1_v_entry"].iloc[tn] == pytest.approx(traj["v"][e, tn])
        # Entry means the mass really is inside the band at that sample.
        assert abs(traj["x"][e, tn] - design.factors["T1"][tn]) <= tol + 1e-9


def test_direction_folding_makes_the_two_directions_comparable(rollout):
    traj, design, _ = rollout
    df = dataset.trial_frame(traj, design, "unit_test")
    assert np.allclose(df["fold"], np.sign(df["T1"]))
    ok = df["T1_v_entry"].notna()
    assert ok.any(), "fixture produced no T1 entries"
    assert np.allclose(df.loc[ok, "T1_v_entry_out"],
                       df.loc[ok, "T1_v_entry"] * df.loc[ok, "fold"])
    # d_next > 0 must mean "T2 lies onward past T1" in both directions.
    right = df[df["direction"] > 0]
    assert np.allclose(right["d_next"], right["T2"] - right["T1"])
    left = df[df["direction"] < 0]
    assert np.allclose(left["d_next"], -(left["T2"] - left["T1"]))


def test_peak_velocity_is_the_extreme_of_the_first_segment(rollout):
    traj, design, _ = rollout
    df = dataset.trial_frame(traj, design, "unit_test")
    for tn in range(min(10, design.n_trials)):
        e = traj["entry_step"][tn, 0]
        if e < 0:
            continue
        seg = traj["v"][traj["go_step"][tn]:e + 1, tn]
        assert df["T1_peak_v"].iloc[tn] == pytest.approx(seg[np.argmax(np.abs(seg))])


# -------------------------------------------------------------------- saving
def test_save_writes_both_tables_plus_metadata(tmp_path, rollout):
    traj, design, cfg = rollout
    paths = dataset.save_experiment(traj, design, cfg, "agent_x", data_dir=tmp_path)
    assert set(paths) == {"timeseries", "trials", "meta"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    # Filename carries experiment, agent and acquisition time.
    assert paths["trials"].name.startswith("preview_sweep_agent_x_")
    meta = json.loads(paths["meta"].read_text())
    assert meta["netName"] == "agent_x"
    assert meta["experiment"] == "preview_sweep"
    assert meta["n_trials"] == design.n_trials
    assert meta["protocol"] == {"noiseless": True, "deterministic": True, "seed": 0}
    assert meta["config"]["plant"]["dt"] == cfg.plant.dt


def test_saved_csv_round_trips_without_losing_precision(tmp_path, rollout):
    """The preview-off control is exactly zero only if full precision survives."""
    traj, design, cfg = rollout
    paths = dataset.save_experiment(traj, design, cfg, "agent_x", data_dir=tmp_path)
    tr = pd.read_csv(paths["trials"])
    ref = dataset.trial_frame(traj, design, "agent_x")
    for col in ("T1_v_entry", "T1_peak_v", "d_next"):
        a, b = tr[col].to_numpy(), ref[col].to_numpy()
        ok = ~np.isnan(b)
        assert np.allclose(a[ok], b[ok], rtol=0, atol=1e-12), col


def test_save_hidden_is_opt_in(tmp_path, rollout):
    traj, design, cfg = rollout
    plain = dataset.save_experiment(traj, design, cfg, "a", data_dir=tmp_path)
    assert "hidden" not in plain
    withh = dataset.save_experiment(traj, design, cfg, "b", data_dir=tmp_path,
                                    save_hidden=True)
    assert withh["hidden"].exists()
    import pickle
    blob = pickle.loads(withh["hidden"].read_bytes())
    assert blob["h"].shape[0] == traj["x"].shape[0]


def test_load_experiment_recovers_the_metadata(tmp_path, rollout):
    traj, design, cfg = rollout
    paths = dataset.save_experiment(traj, design, cfg, "agent_x", data_dir=tmp_path)
    df, meta = dataset.load_experiment(paths["trials"])
    assert len(df) == design.n_trials
    assert meta["experiment"] == "preview_sweep"


# ------------------------------------------------- the result it has to carry
def test_preview_off_control_is_exactly_zero_in_the_saved_table(rollout):
    """The causal control: with T2 hidden the observation does not depend on T2, so
    trajectories must be bit-identical across T2 and the slope exactly 0.  This is the
    property the whole co-articulation claim rests on, checked through the dataframe."""
    traj, design, _ = rollout
    df = dataset.trial_frame(traj, design, "unit_test")
    off = df[(df["preview"] == 1) & df["T1_v_entry"].notna()]
    for _, grp in off.groupby("direction"):
        assert grp["T1_v_entry"].nunique() == 1, "hidden T2 changed the trajectory"
