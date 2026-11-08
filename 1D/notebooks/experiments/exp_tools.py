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


# ------------------------------------------------------------- horizon probe
# (n_targets, preview H) conditions.  H = n is full preview: every target of the
# sequence is visible from trial onset.  H = 1 shows only the current target, but the
# "pending" flags still tell the agent how many targets remain -- so H1 trials know the
# sequence length without knowing where the later targets are.
HORIZON_CONDS = ((1, 1), (2, 2), (3, 3), (2, 1), (3, 1))


def cond_label(n: int, h: int) -> str:
    return f"{n} tgt" if h == n else f"{n} tgt, H1"


def _poly_basis(a: np.ndarray, b: np.ndarray, deg: int = 3) -> np.ndarray:
    """All monomials a^i b^j with i + j <= deg (includes the constant)."""
    return np.column_stack([a ** i * b ** j for i in range(deg + 1)
                            for j in range(deg + 1 - i)])


def _coef(y: np.ndarray, X: np.ndarray, col: int) -> float:
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() <= X.shape[1] + 2:
        return np.nan
    c, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return float(c[col])


def horizon_probe(run, n_geom: int = 400, start_range: float = 0.5, seed: int = 11,
                  rt_threshold: float = 2.0, conds=HORIZON_CONDS, return_traj: bool = False):
    """Sequences of 1, 2 or 3 random targets from random starts, geometry matched.

    ``n_geom`` random geometries are drawn once -- start x0 ~ U[-start_range,
    start_range], targets T1..T3 uniform in the target range with the training
    min-separation, and a go time from the training hold distribution.  Each geometry
    is then replayed in every condition of ``conds``.  A 2-target trial is therefore the
    1-target trial plus T2, and so on: any change in RT or in the first reach comes
    from the extra targets alone, not from a different first reach.

    Per trial:
      * ``rt_ms``  -- force-onset RT: when the force on the mass (the command after the
        50 ms actuator lag) first reaches ``rt_threshold`` (25% of u_max), interpolated
        between steps, in ms from the true go step (so it includes the 80 ms cue delay).
        Force is signed towards T1.  ``rt_u_ms`` is the command RT (first step with
        u >= threshold towards T1), quantised to the 20 ms step; ``wrong_way`` flags a
        command of the same size in the wrong direction before it.
      * ``v_out``  -- velocity on first entering T1, signed so + = still moving the way
        the first reach went; ``d_next`` = (T2 - T1) in the same frame (+ = T2 continues
        onward).  ``v_out2`` / ``d_next2`` are the same at T2 (3-target trials).
      * first-reach peak speed and movement time to T1 entry.

    Returns ``(trials_df, geometry)``, plus the raw rollout if ``return_traj``.
    Noiseless plant, mean action.
    """
    _, _, cfg = load_agent(run)
    t, dt, K = cfg.task, cfg.plant.dt, cfg.task.max_targets
    rng = np.random.default_rng(seed)

    x0 = rng.uniform(-start_range, start_range, n_geom)
    tg = np.zeros((n_geom, K))
    prev = x0.copy()
    for k in range(K):
        cand = rng.uniform(-t.target_range, t.target_range, n_geom)
        for _ in range(200):
            bad = np.abs(cand - prev) < t.min_separation
            if not bad.any():
                break
            cand[bad] = rng.uniform(-t.target_range, t.target_range, int(bad.sum()))
        tg[:, k] = cand
        prev = cand
    go = rng.integers(int(t.delay_min_s / dt), int(t.delay_max_s / dt) + 1, n_geom)

    C = len(conds)
    n_arr = np.repeat([c[0] for c in conds], n_geom)
    h_arr = np.repeat([c[1] for c in conds], n_geom)
    gi = np.tile(np.arange(n_geom), C)
    df, tr = measure_trials(run, x0[gi], tg[gi], go[gi], n_arr, h_arr, seed=seed,
                            rt_threshold=rt_threshold)
    df.insert(0, "geom", gi)
    df.insert(1, "cond", [cond_label(n, h) for n, h in zip(n_arr, h_arr)])
    geo = {"x0": x0, "targets": tg, "go": go, "dt": dt}
    return (df, geo, tr) if return_traj else (df, geo)


def measure_trials(run, x0, tg, go, n_targets, preview, seed: int = 0,
                   rt_threshold: float = 2.0, noiseless: bool = True):
    """Run one trial per row (start ``x0``, targets ``tg`` (N, K), go step ``go``,
    ``n_targets``, ``preview``) and measure RT, the first reach and the T1/T2 exits.

    The measures are documented in ``horizon_probe``, plus two that separate *when* the
    response starts from *how hard* it starts: ``onset_ms``, the first step (once the cue
    can be seen) at which the command moves >= 0.2 from its pre-cue value, and
    ``u_first``, the first post-cue command, signed towards T1.  ``noiseless=False`` turns the
    plant's motor noise back on (the policy still takes its mean action).
    Returns ``(trials_df, raw_rollout)``.
    """
    agent, obs_rms, cfg = load_agent(run)
    pcfg = probes.probe_config(cfg, noiseless=noiseless)
    dt, cue = cfg.plant.dt, int(cfg.plant.cue_delay)
    x0, tg = np.asarray(x0, float), np.asarray(tg, float)
    n_arr, h_arr = np.asarray(n_targets), np.asarray(preview)
    spec = {"n_targets": n_arr, "preview": h_arr, "start_x": x0, "targets": tg,
            "delay_steps": np.asarray(go), "catch": np.zeros(len(x0), dtype=bool)}
    tr = probes.rollout(agent, obs_rms, pcfg, spec, n_trials=len(x0), seed=seed)

    entry = tr["entry_step"]
    rows = []
    for i in range(len(x0)):
        g, e = int(tr["go_step"][i]), int(tr["end_step"][i])
        a, b = x0[i], tg[i]
        s1 = np.sign(b[0] - a)
        rec = {"n_targets": int(n_arr[i]), "preview": int(h_arr[i]),
               "x0": a, "T1": b[0], "T2": b[1],
               "T3": b[2], "go_s": g * dt, "success": bool(tr["success"][i]),
               "false_start": bool(tr["false_start"][i]), "rt_ms": np.nan, "rt_u_ms": np.nan,
               "onset_ms": np.nan, "u_first": np.nan, "wrong_way": np.nan, "peak_v1": np.nan, "mt1_ms": np.nan,
               "v_out": np.nan, "d_next": (b[1] - b[0]) * s1, "dist1": abs(b[0] - a),
               "v_out2": np.nan, "d_next2": np.nan}
        if e >= g:
            # Response onset (latency): the first step, once the cue can be seen, at which
            # the command moves >= 0.2 (2.5% of u_max) in either direction from the last
            # pre-cue command.  Command k after go is issued at k * dt.  ``u_first`` is the
            # first post-cue command towards T1 (the vigour of the response).
            us = tr["u"][g:e + 1, i] * s1
            base = us[cue - 1] if cue > 0 else tr["u"][g - 1, i] * s1
            dev = np.flatnonzero(np.abs(us[cue:] - base) >= 0.2)
            rec["onset_ms"] = (cue + int(dev[0])) * dt * 1000 if dev.size else np.nan
            rec["u_first"] = float(us[cue]) if len(us) > cue else np.nan
            # RT counts force *towards T1* (signed), as in force-onset RT in people.
            # A brief push the wrong way first is recorded separately (``wrong_way``).
            # Command RT: the policy's output is held for a whole step, so it is
            # quantised to 20 ms.  Step g + k is the k-th step after the go step.
            u = tr["u"][g:e + 1, i] * s1
            k = np.flatnonzero(u >= rt_threshold)
            if k.size:
                rec["rt_u_ms"] = int(k[0]) * dt * 1000
                rec["wrong_way"] = bool((u[:k[0]] <= -rt_threshold).any())
            # Force RT: the force on the mass is the command through the 50 ms
            # actuator lag, a continuous signal, so its threshold crossing can be
            # interpolated.  Recorded state g + k is the end of step g + k, i.e.
            # (k + 1) dt after the go step began.  A strong command crosses within the
            # first step after it is issued, a weak one takes longer.
            f = tr["a"][g:e + 1, i] * s1
            k = np.flatnonzero(f >= rt_threshold)
            if k.size:
                k0 = int(k[0])
                f_prev = f[k0 - 1] if k0 else 0.0
                frac = (rt_threshold - f_prev) / (f[k0] - f_prev)
                rec["rt_ms"] = (k0 + frac) * dt * 1000
            e1 = int(entry[i, 0])
            if e1 >= g:
                seg = tr["v"][g:e1 + 1, i]
                rec["peak_v1"] = float(np.abs(seg).max())
                rec["mt1_ms"] = (e1 + 1 - g) * dt * 1000 - rec["rt_ms"]
                rec["v_out"] = float(tr["v"][e1, i] * s1)
            e2 = int(entry[i, 1])
            if n_arr[i] >= 3 and e2 > e1 >= g:
                s2 = np.sign(b[1] - b[0])
                rec["v_out2"] = float(tr["v"][e2, i] * s2)
                rec["d_next2"] = (b[2] - b[1]) * s2
        rows.append(rec)
    return pd.DataFrame(rows), tr


def horizon_ci(df: pd.DataFrame, deg: int = 3) -> dict:
    """Lookahead from random-geometry trials.

    With T1 and the start random, the speed at T1 depends strongly on the first reach
    itself (its length, how close T1 is to the wall).  So the co-articulation index is
    the partial slope of v_out on d_next, controlling for a cubic polynomial in
    (x0, T1).  The H1 condition is the empirical null: there T2 is invisible until T1 is
    entered, so the true slope is exactly 0.

    For 3-target trials also: ``ci_T3`` (does T3 change the exit speed at T1, beyond
    T2?) and ``ci_2`` (lookahead at T2 towards T3, controlling for T1 and T2).
    """
    d = df[df.success]
    out = {}
    for cond, g in d.groupby("cond"):
        if g.n_targets.iloc[0] < 2:
            continue
        base = _poly_basis(g.x0.to_numpy(), g.T1.to_numpy(), deg)
        X = np.column_stack([g.d_next, base])
        res = {"ci": _coef(g.v_out.to_numpy(), X, 0), "n_ci": int(g.v_out.notna().sum())}
        if g.n_targets.iloc[0] == 3:
            s1 = np.sign(g.T1 - g.x0).to_numpy()
            d3 = ((g.T3 - g.T2) * s1).to_numpy()
            X3 = np.column_stack([g.d_next, d3, base])
            res["ci_T3"] = _coef(g.v_out.to_numpy(), X3, 1)
            base2 = _poly_basis(g.T1.to_numpy(), g.T2.to_numpy(), deg)
            X2 = np.column_stack([g.d_next2, base2])
            res["ci_2"] = _coef(g.v_out2.to_numpy(), X2, 0)
        out[cond] = res
    return out


# ----------------------------------------------------------- direction probe
def direction_probe(run, n_geom: int = 300, d1=(0.2, 0.3), d2=(0.4, 0.6),
                    start_range: float = 0.5, seed: int = 21, noiseless: bool = True,
                    rt_threshold: float = 2.0, return_traj: bool = False):
    """Two-target trials in which only the *direction* of T2 changes.

    Each geometry draws a start x0 ~ U[-start_range, start_range], a reach direction s,
    a first-reach length d1 and a second-reach length d2 (uniform in the given
    ranges), and a go time from the training hold distribution.  Then
        T1 = x0 + s d1,   T2 = T1 + s d2 (``continues``)  or  T1 - s d2 (``reverses``),
    redrawn until all three points lie inside the target range.  Each geometry is run in
    four conditions: continues / reverses x T2 visible (H = 2) / hidden (H = 1).  With T2
    hidden the two directions are *the same trial* up to T1 entry, so their RT
    difference is exactly 0: the built-in null.

    Returns a trial DataFrame with ``geom``, ``T2_dir`` and ``visible`` columns (and the
    raw rollout, row-aligned with it, if ``return_traj``).
    """
    _, _, cfg = load_agent(run)
    t, dt, K = cfg.task, cfg.plant.dt, cfg.task.max_targets
    rng = np.random.default_rng(seed)
    R = t.target_range
    x0, s, a, b = (np.empty(n_geom) for _ in range(4))
    for i in range(n_geom):
        while True:
            x0[i] = rng.uniform(-start_range, start_range)
            s[i] = rng.choice([-1.0, 1.0])
            a[i], b[i] = rng.uniform(*d1), rng.uniform(*d2)
            T1 = x0[i] + s[i] * a[i]
            if max(abs(T1), abs(T1 + s[i] * b[i]), abs(T1 - s[i] * b[i])) <= R:
                break
    go = rng.integers(int(t.delay_min_s / dt), int(t.delay_max_s / dt) + 1, n_geom)
    T1 = x0 + s * a

    rows_x0, rows_tg, rows_go, labels = [], [], [], []
    for direction, sign in (("continues", 1.0), ("reverses", -1.0)):
        T2 = T1 + sign * s * b
        tg = np.zeros((n_geom, K))
        tg[:, 0], tg[:, 1:] = T1, T2[:, None]
        for vis, H in (("visible", 2), ("hidden", 1)):
            rows_x0.append(x0), rows_tg.append(tg), rows_go.append(go)
            labels += [(g, direction, vis, H) for g in range(n_geom)]
    H = np.array([l[3] for l in labels])
    df, tr = measure_trials(run, np.concatenate(rows_x0), np.concatenate(rows_tg),
                           np.concatenate(rows_go), np.full(len(H), 2), H, seed=seed,
                           rt_threshold=rt_threshold, noiseless=noiseless)
    df.insert(0, "geom", [l[0] for l in labels])
    df.insert(1, "T2_dir", [l[1] for l in labels])
    df.insert(2, "visible", [l[2] for l in labels])
    df["d1"], df["d2"] = np.tile(a, 4), np.tile(b, 4)
    return (df, tr) if return_traj else df
