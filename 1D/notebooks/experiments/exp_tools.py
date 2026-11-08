"""Shared code for the experiment notebooks in this folder.

Puts ``1D/``, ``1D/notebooks/`` and ``1D/scripts/`` on the import path, re-exports the
lesson plot style, and holds probe functions that more than one experiment uses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # .../SeqHorizon_basic/1D
for p in (ROOT, ROOT / "notebooks", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lesson_tools import (AQUA, BLUE, DIVERGING, GRID, INK, INK_SOFT, ORANGE,  # noqa: E402,F401
                          use_style)
from seqreach import probes  # noqa: E402
from seqreach.ppo import load_agent  # noqa: E402

RUNS = ROOT / "runs"
SEEDS = range(1, 9)


def rel(path) -> str:
    """Path relative to the notebook's working directory (for IPython.display.Video)."""
    return os.path.relpath(Path(path), Path.cwd())


# --------------------------------------------------------------- hold probe
def target_side(x0: float, target: float) -> str:
    """Where the target sits relative to the start, seen from the workspace centre.

    ``outward``: same side as the start and further from the centre, so "towards the
    target" and "towards the centre" point in *opposite* directions.  These are the
    conditions that can tell the two explanations of hold drift apart.
    ``inward``: anything else off-centre (target nearer the centre, or across it), where
    the two directions coincide.  ``centre``: the start is the trained start, x = 0.
    """
    if x0 == 0:
        return "centre"
    if np.sign(target) == np.sign(x0) and abs(target) > abs(x0):
        return "outward"
    return "inward"


def hold_probe(run, starts=None, offsets=(-0.4, -0.2, 0.2, 0.4), hold_s: float = 3.0,
               target_range: float = 0.9):
    """Hold the mass at a chosen start with one visible target, and watch it drift.

    The false-start abort is switched off (``hold_abort_dist`` = 10) and the hold lasts
    ``hold_s`` seconds, so the trajectory shows where the policy *takes* the mass during
    the hold rather than stopping at the 0.15 abort line.  Noiseless plant, mean action.

    Returns ``(df, traj, dt, hold)``: one row per condition, the raw rollout, the step
    length and the hold length in steps.
    """
    if starts is None:
        starts = np.round(np.arange(-0.8, 0.81, 0.2), 2)
    agent, obs_rms, cfg = load_agent(run)
    hold = int(round(hold_s / cfg.plant.dt))
    pcfg = probes.probe_config(cfg).override(
        {"task.hold_abort_dist": 10.0, "task.max_steps": hold + 250})

    rows = [(x0, round(x0 + d, 3)) for x0 in starts for d in offsets
            if abs(x0 + d) <= target_range + 1e-9]
    x0 = np.array([r[0] for r in rows], dtype=float)
    tg = np.array([r[1] for r in rows], dtype=float)
    K = cfg.task.max_targets
    targets = np.repeat(tg[:, None], K, axis=1)  # slots past n_targets are never used
    spec = {"n_targets": 1, "preview": 1, "start_x": x0, "targets": targets,
            "delay_steps": hold}
    traj = probes.rollout(agent, obs_rms, pcfg, spec, n_trials=len(rows), seed=0)
    dt = cfg.plant.dt

    def at(arr, s):
        return arr[min(int(round(s / dt)) - 1, hold - 1)]

    recs = []
    for i in range(len(rows)):
        x, v, u = traj["x"][:hold, i], traj["v"][:hold, i], traj["u"][:hold, i]
        recs.append({
            "run": Path(run).name, "x0": x0[i], "target": tg[i],
            "offset": round(tg[i] - x0[i], 3), "side": target_side(x0[i], tg[i]),
            "u_early": float(u[:5].mean()),  # mean command over the first 100 ms
            "v_0.3s": float(at(v, 0.3)), "v_1s": float(at(v, 1.0)),
            "x_1s": float(at(x, 1.0)), "x_end": float(x[-1]),
            "false_start_by_1s": bool((np.abs(x[: int(1.0 / dt)] - x0[i]) > 0.15).any()),
        })
    df = pd.DataFrame(recs)
    df["drift_to_centre"] = np.where(df.x0 != 0, -np.sign(df.x0) * np.sign(df["v_0.3s"]) > 0,
                                     np.nan)
    df["drift_to_target"] = np.sign(df.offset) * np.sign(df["v_0.3s"]) > 0
    return df, traj, dt, hold


def drift_fit(df: pd.DataFrame, col: str = "v_0.3s") -> dict:
    """Least-squares split of hold drift into a centre term and a target term.

        v = b_centre * x0  +  b_target * (target - x0)  +  c

    ``b_centre`` < 0 means the mass is pushed back towards x = 0 in proportion to how far
    it starts from it; ``b_target`` > 0 means it is pulled towards the target in
    proportion to the target's distance.  Both in 1/s.
    """
    X = np.column_stack([df.x0, df.offset, np.ones(len(df))])
    y = df[col].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"b_centre": coef[0], "b_target": coef[1], "c": coef[2], "R2": r2}


# ------------------------------------------------------------ run evaluation
def competence(run, start_range: float = 0.0, n: int = 512, seed: int = 123):
    """Competence on the training distribution of targets, from the centre (start_range
    = 0) or from random starts in [-start_range, start_range].  Also returns the
    pre-go drift measured on the same trials (true go step, before the agent can have
    seen the cue): displacement and velocity, signed so + = towards the first target."""
    from seqreach import probes as _p
    agent, obs_rms, cfg = load_agent(run)
    pcfg = _p.probe_config(cfg)
    spec = {}
    if start_range > 0:
        spec["start_x"] = np.random.default_rng(seed).uniform(-start_range, start_range, n)
    tr = _p.rollout(agent, obs_rms, pcfg, spec, n_trials=n, seed=seed)
    out = _p.outcome_breakdown(tr)
    x0 = tr["x"][0] - tr["v"][0] * tr["dt"]
    go = tr["go_step"]
    ok = go - 1 < tr["end_step"]  # reached the go cue without a false start
    i = np.flatnonzero(ok)
    sgn = np.sign(tr["targets0"][i, 0] - x0[i])
    dx = sgn * (tr["x"][go[i] - 1, i] - x0[i])
    dv = sgn * tr["v"][go[i] - 1, i]
    out.update(drift_x_to_target=float(dx.mean()), drift_abs=float(np.abs(dx).mean()),
               drift_v_to_target=float(dv.mean()), drift_v_abs=float(np.abs(dv).mean()))
    return out


def reaction_times_u(run, threshold: float = 2.0, n_per: int = 40):
    """RT from the force command: first step at/after the go step where |u| exceeds
    ``threshold`` (default 2 = 25% of u_max).  Also the old velocity-based RT.  By hold
    duration, on delay_probe trials (full preview, 2 targets, centre start)."""
    from seqreach import metrics as _m, probes as _p
    agent, obs_rms, cfg = load_agent(run)
    pcfg = _p.probe_config(cfg)
    spec, design = _p.delay_probe(pcfg, n_per=n_per)
    tr = _p.rollout(agent, obs_rms, pcfg, spec, n_trials=len(design["delay_steps"]), seed=5)
    dt = tr["dt"]
    rows = []
    kin = _m.segment_kinematics(tr, slot=0)
    rt_v = np.full(len(design["delay_steps"]), np.nan)
    rt_v[kin["trials"]] = kin["rt_s"]
    for i in range(len(design["delay_steps"])):
        g, e = int(tr["go_step"][i]), int(tr["end_step"][i])
        if e < g:
            continue  # false start
        u = np.abs(tr["u"][g:e + 1, i])
        k = np.flatnonzero(u >= threshold)
        rows.append({"delay_s": design["delay_steps"][i] * dt,
                     "rt_u_ms": k[0] * dt * 1000 if k.size else np.nan,
                     "rt_v_ms": rt_v[i] * 1000})
    return pd.DataFrame(rows)


def lookahead(run):
    """Co-articulation index with T2 visible and hidden (the v1 headline measure)."""
    from seqreach import metrics as _m, probes as _p
    agent, obs_rms, cfg = load_agent(run)
    pcfg = _p.probe_config(cfg)
    spec, design = _p.preview_probe(pcfg)
    tr = _p.rollout(agent, obs_rms, pcfg, spec, n_trials=len(design["t1"]), seed=7)
    res = _m.co_articulation_by_preview(tr, design)
    dep = _m.t2_dependence(_m._subset(tr, design["preview"] == 2),
                           {k: v[design["preview"] == 2] for k, v in design.items()})
    return {"ci": res["preview_on"]["ci"], "ci_p": res["preview_on"]["ci_p"],
            "ci_control": res["preview_off"]["ci"],
            "t2_v_spread": dep.get("v_spread_rms", np.nan)}
