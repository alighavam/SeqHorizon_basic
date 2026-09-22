"""Post-hoc measurements.

None of these quantities appears in the reward.  They are read off finished
trajectories exactly the way one would read them off a behavioural experiment.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _at(arr: np.ndarray, step: np.ndarray, trial: np.ndarray) -> np.ndarray:
    """arr[(T, N)] sampled at (step, trial) pairs."""
    return arr[step, trial]


def segment_kinematics(traj: dict, slot: int = 0) -> dict:
    """Kinematics of the reach to target ``slot``.

    Timed at **first entry** into the target's tolerance band, not at acquisition. The
    two coincide for pass-through targets (``dwell_inter=1``) but diverge sharply when a
    dwell is required, where acquisition is registered only after the dwell completes
    and velocity there is ~0 by construction.  Using entry keeps the co-articulation
    metric comparable across dwell settings -- which matters, because dwell is the
    ablation axis that the separability argument predicts should abolish the effect.
    """
    acq = traj.get("entry_step", traj["acq_step"])[:, slot]
    ok = acq >= 0
    trials = np.flatnonzero(ok)
    acq_ok = acq[ok]
    go = traj["go_step"][ok]

    v_at = _at(traj["v"], acq_ok, trials)
    x_at = _at(traj["x"], acq_ok, trials)

    peak_v = np.zeros(len(trials))
    onset = np.full(len(trials), np.nan)
    for i, (n, g, a) in enumerate(zip(trials, go, acq_ok)):
        seg = traj["v"][g:a + 1, n]
        if seg.size == 0:
            continue
        j = np.argmax(np.abs(seg))
        peak_v[i] = seg[j]
        thr = 0.05 * np.abs(seg[j])
        above = np.flatnonzero(np.abs(seg) >= thr)
        if above.size:
            onset[i] = above[0]

    return {
        "trials": trials,
        "acq_step": acq_ok,
        "v_at_acq": v_at,
        "x_at_acq": x_at,
        "peak_v": peak_v,
        "duration_s": (acq_ok - go) * traj["dt"],
        "rt_s": onset * traj["dt"],
    }


def co_articulation(traj: dict, design: dict, slot: int = 0,
                    group: str = "direction") -> dict:
    """Regress velocity at target-``slot`` acquisition on the *next* target's offset.

    Signs are folded by reach direction so both directions pool:
        v_out = v * sign(T1 - start)      (positive = still moving outward)
        d_next = (T2 - T1) * sign(T1 - start)   (positive = T2 continues onward)

    CI = d v_out / d d_next.  CI > 0 means the agent leaves T1 already carrying
    momentum toward T2 -- co-articulation.
    """
    k = segment_kinematics(traj, slot)
    tr = k["trials"]
    t1, t2 = design["t1"][tr], design["t2"][tr]
    sign = np.sign(t1)  # start_x is 0 in every probe
    v_out = k["v_at_acq"] * sign
    d_next = (t2 - t1) * sign

    out = {"d_next": d_next, "v_out": v_out, "peak_v_out": k["peak_v"] * sign,
           "duration_s": k["duration_s"], "trials": tr, "n": len(tr)}

    # Leftward and rightward reaches have different intercepts (the learned policy is
    # not mirror-symmetric), and the two directions do not span the same range of
    # d_next.  Regressing on the pooled cloud therefore picks up a between-direction
    # component that has nothing to do with T2 -- it put a spurious -0.007 on a control
    # whose true value is exactly 0.  Demeaning within direction (a fixed-effects
    # estimator) removes it and leaves the within-direction slope we actually want.
    groups = np.asarray(design[group])[tr] if group in design else np.zeros(len(tr))
    dn, vv = _demean_within(d_next, groups), _demean_within(v_out, groups)
    pv = _demean_within(k["peak_v"] * sign, groups)
    du = _demean_within(k["duration_s"], groups)

    if len(tr) >= 3 and np.ptp(dn) > 0:
        out.update(**_slope(dn, vv, prefix="ci"))
        out["peak_slope"] = _slope(dn, pv)["slope"]
        out["peak_p"] = _slope(dn, pv)["p"]
        out["dur_slope"] = _slope(dn, du)["slope"]
        out["dur_p"] = _slope(dn, du)["p"]
    else:
        out.update(ci=np.nan, ci_se=np.nan, ci_p=np.nan, ci_r=np.nan,
                   peak_slope=np.nan, peak_p=np.nan, dur_slope=np.nan, dur_p=np.nan)

    # Balanced-window refit.  The probe fixes T1 at +-0.45 while T2 sweeps +-0.9, so
    # d_next spans [-1.35, +0.45]: three quarters of every slope is measured on the
    # reversing side.  Any curvature is then converted into a spurious *linear* slope
    # whose sign is set by which arm dominates the sample.  Refitting on the widest
    # window symmetric about zero costs no extra rollouts and removes that asymmetry;
    # a large gap between `ci` and `ci_balanced` means the relation is not linear and
    # the single slope should not be read as a signed effect.
    d_max = min(dn.max(), -dn.min()) if len(dn) else 0.0
    bal = np.abs(dn) <= d_max
    if bal.sum() >= 3 and np.ptp(dn[bal]) > 0:
        b = _slope(dn[bal], vv[bal])
        out["ci_balanced"] = b["slope"]
        out["ci_balanced_se"] = b["se"]
        out["ci_balanced_p"] = b["p"]
        out["n_balanced"] = int(bal.sum())
    else:
        out.update(ci_balanced=np.nan, ci_balanced_se=np.nan,
                   ci_balanced_p=np.nan, n_balanced=int(bal.sum()))

    # Per-direction slopes, reported alongside because they differ substantially.
    out["by_group"] = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 3 and np.ptp(d_next[m]) > 0:
            out["by_group"][float(g)] = _slope(d_next[m], v_out[m])["slope"]
    # A signed CI is only meaningful if the two reach directions agree on the sign.
    bg = list(out["by_group"].values())
    out["dir_agree"] = bool(len(bg) >= 2 and (np.sign(bg[0]) == np.sign(bg[1])))
    return out


def _demean_within(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean()
    return out


def _slope(x: np.ndarray, y: np.ndarray, prefix: str | None = None) -> dict:
    """OLS slope, tolerant of the degenerate constant-y case (which is meaningful here:
    with the next target hidden, y is *exactly* constant)."""
    if np.ptp(x) == 0 or len(x) < 3:
        res = {"slope": np.nan, "se": np.nan, "p": np.nan, "r": np.nan}
    elif np.ptp(y) == 0:
        res = {"slope": 0.0, "se": 0.0, "p": 1.0, "r": 0.0}
    else:
        reg = stats.linregress(x, y)
        res = {"slope": reg.slope, "se": reg.stderr, "p": reg.pvalue, "r": reg.rvalue}
    if prefix:
        return {prefix: res["slope"], f"{prefix}_se": res["se"],
                f"{prefix}_p": res["p"], f"{prefix}_r": res["r"]}
    return res


def co_articulation_by_preview(traj: dict, design: dict) -> dict:
    """CI computed separately for preview-on and preview-off trials.

    Preview-off is the causal control: with T2 hidden the observation is identical for
    every T2, so any non-zero slope there would indicate a broken pipeline.
    """
    res = {}
    for name, level in (("preview_on", 2), ("preview_off", 1)):
        m = design["preview"] == level
        sub_design = {k: v[m] for k, v in design.items()}
        sub_traj = _subset(traj, m)
        res[name] = co_articulation(sub_traj, sub_design)
    res["delta_ci"] = res["preview_on"]["ci"] - res["preview_off"]["ci"]
    return res


def t2_dependence(traj: dict, design: dict, slot: int = 0,
                  group: str = "direction", tol: float = 0.05) -> dict:
    """Sign-agnostic magnitude of T2's influence on the reach to target ``slot``.

    ``co_articulation`` asks *which way* the reach bends with T2, and its sign is only
    interpretable when the underlying relation is monotonic and consistent across reach
    directions -- which is not guaranteed (a dwell requirement, or an agent whose two
    directions disagree, both break it).  This asks the weaker but unambiguous question:
    does the trajectory to T1 depend on T2 **at all**?

    Within one reach direction the probe trials differ only in T2, so at each timestep
    the spread of ``x`` across them is exactly the influence of T2.  With T2 hidden the
    observation is identical across those trials and the spread is *exactly* zero --
    a hard null, not an estimated one -- so the preview-off value doubles as a pipeline
    check.

    Measured from the go cue to the earliest first-entry into T1's band, so it describes
    the reach to T1 only and never the movement toward T2.
    """
    k = segment_kinematics(traj, slot)
    tr = k["trials"]
    nan = {"spread_rms": np.nan, "spread_peak": np.nan,
           "spread_rms_tol": np.nan, "spread_peak_tol": np.nan,
           "v_spread_rms": np.nan, "v_spread_peak": np.nan, "n": len(tr)}
    if len(tr) < 3:
        return nan
    groups = np.asarray(design[group])[tr] if group in design else np.zeros(len(tr))
    go, entry = traj["go_step"][tr], k["acq_step"]

    rms, peak, vrms, vpeak = [], [], [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3:
            continue
        g0, g1 = int(np.max(go[m])), int(np.min(entry[m]))
        if g1 <= g0:
            continue
        for key, acc_rms, acc_pk in (("x", rms, peak), ("v", vrms, vpeak)):
            A = np.stack([traj[key][g0:g1, n] for n in tr[m]], axis=1)
            sd = A.std(axis=1, ddof=1)
            acc_rms.append(float(np.sqrt((sd ** 2).mean())))
            acc_pk.append(float(sd.max()))
    if not rms:
        return nan
    # Position spread is reported in tolerance units; in 1D the path is a line, so most
    # of the lookahead lives in the velocity profile rather than in where the mass goes.
    return {"spread_rms": float(np.mean(rms)), "spread_peak": float(np.mean(peak)),
            "spread_rms_tol": float(np.mean(rms) / tol),
            "spread_peak_tol": float(np.mean(peak) / tol),
            "v_spread_rms": float(np.mean(vrms)), "v_spread_peak": float(np.mean(vpeak)),
            "n": len(tr)}


def t2_dependence_by_preview(traj: dict, design: dict, tol: float = 0.05) -> dict:
    """``t2_dependence`` for preview-on and preview-off trials separately."""
    res = {}
    for name, level in (("preview_on", 2), ("preview_off", 1)):
        m = design["preview"] == level
        res[name] = t2_dependence(_subset(traj, m),
                                  {k: v[m] for k, v in design.items()}, tol=tol)
    return res


def _subset(traj: dict, mask: np.ndarray) -> dict:
    """Select a subset of trials from a rollout record.

    Time-major arrays are identified by matching the recorded number of timesteps, not
    just by rank -- otherwise a per-trial (N, K) array would be mistaken for time-major
    whenever N happened to equal the trial count.
    """
    n_steps = traj["x"].shape[0]
    out = {}
    for k, v in traj.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
        elif v.ndim >= 2 and v.shape[0] == n_steps and v.shape[1] == mask.size:
            out[k] = v[:, mask]  # (T, N, ...)
        elif v.shape[0] == mask.size:
            out[k] = v[mask]  # (N, ...)
        else:
            out[k] = v
    return out


def trajectory_divergence(traj: dict, design: dict, tol: float = 2e-3) -> dict:
    """Earliest post-go time at which reaches to the *same* T1 differ across T2.

    Answers "how early does the future target start shaping the movement?"
    """
    m = design["preview"] == 2
    sub = _subset(traj, m)
    d = design["direction"][m] if "direction" in design else np.ones(m.sum())
    res = {}
    for dir_val in np.unique(d):
        sel = d == dir_val
        x = sub["x"][:, sel]
        alive = sub["alive"][:, sel]
        go = int(np.median(sub["go_step"][sel]))
        spread = np.where(alive.all(1), x.max(1) - x.min(1), np.nan)
        # The agent adopts a slightly T2-dependent *holding* posture, so position
        # already differs before the go cue.  Divergence of the movement must therefore
        # be measured against that pre-existing offset, not against zero -- otherwise
        # the postural bias alone reads as "diverged at movement onset".
        pre_go = float(spread[max(go - 1, 0)])
        idx = np.flatnonzero(spread[go:] > pre_go + tol)
        res[float(dir_val)] = {
            "divergence_step": int(idx[0]) if idx.size else None,
            "divergence_s": float(idx[0] * traj["dt"]) if idx.size else None,
            "pre_go_spread": pre_go,
            "spread": spread,
            "go_step": go,
        }
    return res


def reaction_times(traj: dict, design: dict) -> dict:
    """RT from go cue to movement onset (5% of that trial's peak speed), by delay."""
    k = segment_kinematics(traj, slot=0)
    tr = k["trials"]
    delays = design["delay_steps"][tr]
    rt = k["rt_s"]
    good = np.isfinite(rt)
    rows = []
    for d in np.unique(delays):
        m = (delays == d) & good
        if m.sum() == 0:
            continue
        rows.append({"delay_s": float(d * traj["dt"]), "n": int(m.sum()),
                     "rt_mean": float(rt[m].mean()),
                     "rt_sem": float(stats.sem(rt[m])) if m.sum() > 1 else 0.0})
    out = {"by_delay": rows}
    if len(rows) >= 3:
        reg = stats.linregress([r["delay_s"] for r in rows], [r["rt_mean"] for r in rows])
        out.update(slope=reg.slope, p=reg.pvalue, r=reg.rvalue)
    return out


def jump_response(traj: dict, design: dict) -> dict:
    """Compare jump trials against their matched no-jump control.

    The future-target rows are the ones that matter: if reaching to T1 changes when
    only T2 moved, the ongoing movement was being planned through T2.
    """
    kinds, signs = design["jump_kind"], design["jump_sign"]
    dt = traj["dt"]
    out = []
    for sgn in np.unique(signs):
        base_i = np.flatnonzero((kinds == 0) & (signs == sgn))[0]
        xb = traj["x"][:, base_i]
        for kind, label in ((1, "jump_current"), (2, "jump_future")):
            sel = np.flatnonzero((kinds == kind) & (signs == sgn))
            if sel.size == 0:
                continue
            i = sel[0]
            xj = traj["x"][:, i]
            n = min(int(traj["end_step"][i]), int(traj["end_step"][base_i]))
            diff = np.abs(xj[:n] - xb[:n])
            js = int(traj["jump_step"][i])
            after = np.flatnonzero(diff[js:] > 2e-3)
            acq0 = int(traj["acq_step"][i, 0])
            out.append({
                "kind": label,
                "sign": float(sgn),
                "jump_step": js,
                "latency_s": float(after[0] * dt) if after.size else None,
                "max_dev": float(diff.max()) if diff.size else 0.0,
                # deviation confined to the *first* segment (before T1 is acquired)
                "dev_before_t1": float(diff[js:acq0].max()) if acq0 > js else 0.0,
                "t1_acq_shift_s": float((traj["acq_step"][i, 0]
                                         - traj["acq_step"][base_i, 0]) * dt),
            })
    return {"rows": out}


# ------------------------------------------------------------- neural analysis
def _cv_ridge_r2(X: np.ndarray, y: np.ndarray, k: int = 5,
                 alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0)) -> float:
    """Cross-validated ridge R².

    Hidden state is high-dimensional relative to trial count, so an unregularised
    in-sample fit reads R²=1.0 regardless of whether anything is encoded.  Held-out
    prediction is the only honest test.
    """
    n = len(y)
    if n < 2 * k or np.ptp(y) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    best = -np.inf
    for a in alphas:
        pred = np.empty(n)
        for f in folds:
            tr = np.setdiff1d(order, f)
            Xt, yt = X[tr], y[tr]
            mu_x, mu_y = Xt.mean(0), yt.mean()
            Xc, yc = Xt - mu_x, yt - mu_y
            w = np.linalg.solve(Xc.T @ Xc + a * np.eye(X.shape[1]), Xc.T @ yc)
            pred[f] = (X[f] - mu_x) @ w + mu_y
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        best = max(best, 1 - ss_res / ss_tot)
    return float(best)



def preparatory_analysis(traj: dict, design: dict, readout_w: np.ndarray,
                         window: int = 5, extra: dict | None = None) -> dict:
    """Delay-period structure in the recurrent state.

    Read the decoding numbers with care.  T1 and T2 are *present in the observation*
    throughout the hold period, so a high R² for them is a pass-through check, not
    evidence of preparation -- it would be alarming if it were low.  The two
    informative quantities are:

    1. *Upcoming movement parameters* (``extra``, e.g. peak velocity of the first
       reach): these are not in the input, so decoding them from the pre-go state means
       the movement has been specified before the go cue.
    2. *Output-null geometry.*  The mean action is a linear readout ``u = w.h``, so
       ``w`` spans the 1-D potent space.  Preparation that specifies a movement without
       executing it should put its across-trial variance *off* ``w`` -- i.e. variance
       along ``w`` below that of a typical random direction (ratio < 1).
    """
    h_all = traj["h"]  # (T, N, H)
    go = traj["go_step"]
    N = h_all.shape[1]
    idx = np.stack([np.clip(go - 1 - k, 0, h_all.shape[0] - 1) for k in range(window)])
    H = h_all[idx, np.arange(N)[None, :], :].mean(0)  # (N, H) mean prep state

    w = np.asarray(readout_w).reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)
    Hc = H - H.mean(0, keepdims=True)

    var_potent = float(np.var(Hc @ w))
    total_var = float(np.trace(np.cov(Hc, rowvar=False)))
    dim = Hc.shape[1]
    var_random = total_var / dim  # expected variance along an arbitrary direction

    rng = np.random.default_rng(0)
    rand_dirs = rng.standard_normal((dim, 2000))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
    null_dist = np.var(Hc @ rand_dirs, axis=0)
    pct = float((null_dist < var_potent).mean())

    dec = {}
    for name in ("t1", "t2"):
        if name not in design:
            continue
        dec[f"r2_{name}"] = _cv_ridge_r2(Hc, np.asarray(design[name])[:N])
    for name, vals in (extra or {}).items():
        vals = np.asarray(vals, dtype=float)
        trials = np.asarray((extra or {}).get(f"{name}__trials",
                                              np.arange(len(vals))), dtype=int)
        if name.endswith("__trials"):
            continue
        ok = np.isfinite(vals) & (trials < N)
        if ok.sum() > 20:
            dec[f"r2_{name}"] = _cv_ridge_r2(Hc[trials[ok]], vals[ok])

    # Variance explained by the top principal components of the preparatory state.
    if Hc.shape[0] > 2:
        s = np.linalg.svd(Hc, compute_uv=False)
        ev = s**2 / (s**2).sum()
        dec["pc_var_ratio"] = ev[:5].tolist()

    return {
        "var_potent": var_potent,
        "var_random_expected": var_random,
        "potent_ratio": var_potent / (var_random + 1e-12),
        "potent_percentile": pct,
        "total_var": total_var,
        **dec,
    }
