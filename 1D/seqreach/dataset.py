"""Turn a frozen-agent rollout into analysis-ready tables on disk.

Two tables come out of every experiment, in the shape behavioural data usually wants:

**timeseries** -- long format, one row per (trial, sample). This is the movement itself:
``t_ms``, ``x``, ``v``, ``a``, ``u``, plus which target is currently being pursued and
whether the go cue has fired. Trial-level columns (targets, preview, your factors) are
repeated down each trial's rows so the table stands alone -- you can group by condition
without joining anything.

**trials** -- one row per trial: the design, the outcome, and the per-target event times
and kinematics (entry/acquisition sample, velocity at entry, peak velocity, duration).
This is what most analyses actually regress on.

Both are flat, so both are written as CSV. The GRU hidden state is not flat -- it is
(samples x trials x 128) -- so it goes to a pickle only when asked for.

Sign conventions, stated once because they decide what the numbers mean:

* ``x``, ``v``, ``u`` are signed in world coordinates: positive is rightward.
* ``direction`` (when an experiment defines it) is the sign of T1, so a leftward trial
  has direction = -1.
* ``*_out`` columns are folded by direction (multiplied by ``sign(T1)``) so that
  positive always means "still travelling towards the next target". Fold before pooling
  across directions; the policy is not mirror-symmetric.
* ``d_next`` is likewise folded: positive means T2 lies beyond T1, onward from the start.

Time is reported as ``t_ms`` (milliseconds from trial onset, i.e. sample * dt * 1000)
and ``t_from_go_ms`` (negative during the hold, 0 at the go cue).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .experiments import Design


# ---------------------------------------------------------------- long table
def timeseries_frame(traj: dict, design: Design, net_name: str,
                     include_dead: bool = False) -> pd.DataFrame:
    """One row per (trial, sample), carrying the trial's *design* down its rows.

    What is carried and what is not is a deliberate split. The condition columns (the
    factors, the targets, preview, delay) are repeated here so the table stands alone and
    you can group by condition without a join. The *derived* per-target measures
    (entry/acquisition times, velocities, peak speeds) are not repeated -- they belong to
    the trial, not to the sample, and duplicating ~40 of them down every row triples the
    file for no gain. Merge them in when you want them::

        ts = pd.read_csv("..._timeseries.csv")
        tr = pd.read_csv("..._trials.csv")
        df = ts.merge(tr[["TN", "T1_v_entry_out"]], on="TN")

    The environment autoresets in place, so samples after a trial ends belong to the
    *next* trial and are dropped by default. ``include_dead=True`` keeps them, flagged
    by ``alive``, which is occasionally useful for debugging but never for analysis.
    """
    dt_ms = traj["dt"] * 1000.0
    n_samples, n_trials = traj["x"].shape
    sample = np.repeat(np.arange(n_samples)[:, None], n_trials, axis=1)
    trial = np.repeat(np.arange(n_trials)[None, :], n_samples, axis=0)
    go = np.broadcast_to(traj["go_step"][None, :], (n_samples, n_trials))
    n_tg = np.broadcast_to(design.spec["n_targets"][None, :], (n_samples, n_trials))

    cols = {
        "netName": net_name,
        "TN": trial.ravel(order="F"),
        "sample": sample.ravel(order="F"),
        "t_ms": (sample * dt_ms).ravel(order="F"),
        "t_from_go_ms": ((sample - go) * dt_ms).ravel(order="F"),
        "x": traj["x"].ravel(order="F"),
        "v": traj["v"].ravel(order="F"),
        "a": traj["a"].ravel(order="F"),
        "u": traj["u"].ravel(order="F"),
        "cur": traj["cur"].ravel(order="F"),
        "go": (sample >= go).ravel(order="F"),
        "alive": traj["alive"].ravel(order="F"),
    }
    # Target positions as they stand at each sample -- they move on jump trials, so this
    # is per-sample rather than per-trial.  Unused slots are NaN, not 0, so a missing
    # third target cannot be mistaken for a target at the origin.
    for k in range(traj["targets"].shape[2]):
        pos = np.where(n_tg > k, traj["targets"][:, :, k], np.nan)
        cols[f"T{k + 1}_now"] = pos.ravel(order="F")

    df = pd.DataFrame(cols)
    if not include_dead:
        df = df[df["alive"]].drop(columns="alive").reset_index(drop=True)

    trial_df = trial_frame(traj, design, net_name)
    carry = ["TN", "experiment", "success", "fold", *design.factors.keys()]
    carry = [c for c in dict.fromkeys(carry) if c in trial_df.columns]
    return df.merge(trial_df[carry], on="TN", how="left", validate="many_to_one")


# --------------------------------------------------------------- trial table
def trial_frame(traj: dict, design: Design, net_name: str) -> pd.DataFrame:
    """One row per trial: design, outcome, and per-target events."""
    dt = traj["dt"]
    n_trials = traj["x"].shape[1]
    K = traj["acq_step"].shape[1]

    out: dict = {
        "netName": net_name,
        "TN": np.arange(n_trials),
        "experiment": design.name,
    }
    # Design factors, exactly as the experiment declared them.
    for k, v in design.factors.items():
        out[k] = v

    # Unused target slots are NaN rather than 0, matching the timeseries table.
    for k in range(K):
        key = f"T{k + 1}"
        if key in out:
            out[key] = np.where(design.spec["n_targets"] > k, out[key], np.nan)

    out["go_sample"] = traj["go_step"]
    out["go_ms"] = traj["go_step"] * dt * 1000.0
    out["end_sample"] = traj["end_step"]
    out["duration_ms"] = (traj["end_step"] + 1) * dt * 1000.0
    out["success"] = traj["success"]
    out["timeout"] = traj["timeout"]
    out["false_start"] = traj["false_start"]
    out["wall_steps"] = traj["wall_steps"]
    out["n_acquired"] = (traj["acq_step"] >= 0).sum(1)

    # Per-target events.  entry = first sample inside the tolerance band (the
    # pass-through moment); acq = the sample the target was registered as taken, which
    # differs from entry whenever a dwell is required.  See doc §5.1.
    for k in range(K):
        entry, acq = traj["entry_step"][:, k], traj["acq_step"][:, k]
        out[f"T{k + 1}_entry_sample"] = np.where(entry >= 0, entry, -1)
        out[f"T{k + 1}_acq_sample"] = np.where(acq >= 0, acq, -1)
        out[f"T{k + 1}_entry_ms"] = np.where(entry >= 0, entry * dt * 1000.0, np.nan)
        out[f"T{k + 1}_acq_ms"] = np.where(acq >= 0, acq * dt * 1000.0, np.nan)
        out[f"T{k + 1}_rt_ms"] = np.where(
            entry >= 0, (entry - traj["go_step"]) * dt * 1000.0, np.nan)
        out[f"T{k + 1}_v_entry"] = _sample_at(traj["v"], entry)
        out[f"T{k + 1}_x_entry"] = _sample_at(traj["x"], entry)
        out[f"T{k + 1}_v_acq"] = _sample_at(traj["v"], acq)
        out[f"T{k + 1}_x_acq"] = _sample_at(traj["x"], acq)
        out[f"T{k + 1}_peak_v"] = _peak_v(traj, k)

    df = pd.DataFrame(out)

    # Direction-folded copies, so the two reach directions can be pooled.  sign(T1) is
    # the fold: after it, positive velocity means "still heading onward".
    if "T1" in df.columns:
        fold = np.sign(df["T1"].to_numpy())
        fold[fold == 0] = 1.0
        df["fold"] = fold
        for k in range(K):
            for base in (f"T{k + 1}_v_entry", f"T{k + 1}_v_acq", f"T{k + 1}_peak_v"):
                if base in df:
                    df[f"{base}_out"] = df[base] * fold
        if "T2" in df.columns and "d_next" not in df.columns:
            df["d_next"] = (df["T2"] - df["T1"]) * fold
    return df


def _sample_at(arr: np.ndarray, step: np.ndarray) -> np.ndarray:
    """arr[(samples, trials)] read at a per-trial sample index; NaN where index < 0."""
    ok = step >= 0
    out = np.full(len(step), np.nan)
    idx = np.flatnonzero(ok)
    if idx.size:
        out[idx] = arr[step[idx], idx]
    return out


def _peak_v(traj: dict, slot: int) -> np.ndarray:
    """Signed velocity of largest magnitude between the go cue and entry to target k."""
    entry, go = traj["entry_step"][:, slot], traj["go_step"]
    start = traj["entry_step"][:, slot - 1] if slot > 0 else go
    out = np.full(len(entry), np.nan)
    for n, (s, e) in enumerate(zip(start, entry)):
        if e < 0:
            continue
        lo = max(int(go[n]), int(s) if s >= 0 else int(go[n]))
        seg = traj["v"][lo:e + 1, n]
        if seg.size:
            out[n] = seg[np.argmax(np.abs(seg))]
    return out


# -------------------------------------------------------------------- saving
def default_data_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root else Path(__file__).resolve().parents[2] / "data"


def stamp(when: datetime | None = None) -> str:
    """Acquisition time, sortable and filename-safe: 20260816-143005."""
    return (when or datetime.now()).strftime("%Y%m%d-%H%M%S")


def save_experiment(traj: dict, design: Design, cfg: Config, net_name: str, *,
                    data_dir: Path | str | None = None, noiseless: bool = True,
                    deterministic: bool = True, seed: int = 0,
                    run_dir: str | None = None, save_hidden: bool = False,
                    include_dead: bool = False) -> dict[str, Path]:
    """Write the tables plus a metadata sidecar; return the paths written.

    Filenames are ``<experiment>_<agent>_<YYYYmmdd-HHMMSS>_<table>.<ext>`` -- experiment
    first so a directory listing groups by protocol, timestamp before the table name so
    the files of one acquisition sort together.
    """
    out_dir = default_data_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{design.name or 'experiment'}_{net_name}_{stamp()}"

    ts = timeseries_frame(traj, design, net_name, include_dead=include_dead)
    tr = trial_frame(traj, design, net_name)

    paths = {"timeseries": out_dir / f"{base}_timeseries.csv",
             "trials": out_dir / f"{base}_trials.csv",
             "meta": out_dir / f"{base}_meta.json"}
    ts.to_csv(paths["timeseries"], index=False)
    tr.to_csv(paths["trials"], index=False)

    if save_hidden and "h" in traj:
        # (samples, layers, trials, units) -- not flat, so pickle rather than CSV.
        import pickle

        paths["hidden"] = out_dir / f"{base}_hidden.pkl"
        with paths["hidden"].open("wb") as fh:
            pickle.dump({"netName": net_name, "experiment": design.name,
                         "h": np.asarray(traj["h"]), "dt": traj["dt"],
                         "alive": traj["alive"], "note": "axes: (sample, layer, trial, unit)"},
                        fh, protocol=4)

    meta = {
        "netName": net_name,
        "experiment": design.name,
        "description": design.description,
        "params": _jsonable(design.params),
        "n_trials": int(design.n_trials),
        "acquired": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "protocol": {"noiseless": noiseless, "deterministic": deterministic,
                     "seed": seed},
        "dt_s": cfg.plant.dt,
        "columns": {"timeseries": list(ts.columns), "trials": list(tr.columns)},
        "config": cfg.to_dict(),
        "code_version": _git_describe(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, default=str))
    return paths


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _git_describe() -> str | None:
    """Best-effort provenance.  The project is not a git repo today; if it becomes one,
    every dataset gains a commit stamp for free."""
    try:
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------- convenience
def load_experiment(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Load one saved table plus its metadata sidecar, if present."""
    path = Path(path)
    df = pd.read_csv(path)
    meta_path = path.parent / (path.name.rsplit("_", 1)[0] + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta
