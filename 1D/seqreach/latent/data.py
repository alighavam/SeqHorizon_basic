"""From our rollouts to the arrays the latent-analysis methods take.

This replaces the original toolbox's spike-train side (readers, binning, ``align_fr``):
the GRU state needs none of it.  The pieces:

* ``record``: run trials and keep the hidden state ``h`` (T, N, 128) together with
  what is needed to interpret it (cue delay, readout weights, dt).
* ``align``: cut a window around a task event for every trial -> (trials, time, units),
  NaN where a trial had not started or had already ended.
* ``balanced_trials`` / ``utils.get_condition_mean``: single trials -> condition
  averages, or equal repeats per condition for the cross-validated methods.
* ``readout``: the output direction.  The mean command is ``u = clip(w.h + b)``
  exactly, so ``w`` spans the 1-D output-potent space and everything orthogonal to it
  is output-null.

Timing.  ``h[k]`` is the state computed on step ``k`` from that step's observation; the
command sent on step ``k`` is read out from it.  The go cue appears on ``go_step`` but
reaches the agent ``cue_delay`` steps later, so ``h[go_step + cue_delay]`` is the first
state that has seen it (event ``"cue"``).  In ``align`` lag 0 is the state recorded on the
event step.
"""

from __future__ import annotations

import numpy as np

from .. import probes
from ..ppo import load_agent

EVENTS = ("go", "cue", "end", "entry1", "entry2", "entry3", "acq1", "acq2", "acq3")


def readout(agent, cfg):
    """Output weights in command units: ``u = clip(w @ h + b, -u_max, u_max)``."""
    u_max = cfg.plant.u_max
    w = agent.actor_mean.weight.detach().numpy().reshape(-1) * u_max
    b = float(agent.actor_mean.bias.detach().numpy().reshape(-1)[0]) * u_max
    return w, b


def record(run, spec: dict, n_trials: int | None = None, seed: int = 0,
           noiseless: bool = True, deterministic: bool = True) -> dict:
    """Run the trials in ``spec`` (as ``probes.rollout``) and return the rollout with
    the GRU state ``h`` plus ``cue_delay``, ``u_max``, ``readout_w`` and ``readout_b``.

    ``noiseless=False`` turns the plant's motor noise on: needed whenever a method
    splits repeats of a condition (cross-validated dPCA, alignment index or
    dissimilarity), because noiseless repeats are identical.
    """
    agent, obs_rms, cfg = load_agent(run)
    if not cfg.agent.recurrent:
        raise ValueError("this run has no recurrent state")
    if n_trials is None:
        lens = [len(v) for v in spec.values() if np.ndim(v) >= 1]
        if not lens:
            raise ValueError("pass n_trials: the spec has no per-trial arrays")
        n_trials = lens[0]
    pcfg = probes.probe_config(cfg, noiseless=noiseless)
    tr = probes.rollout(agent, obs_rms, pcfg, spec, n_trials=n_trials, seed=seed,
                        deterministic=deterministic)
    w, b = readout(agent, cfg)
    tr.update(cue_delay=int(cfg.plant.cue_delay), u_max=float(cfg.plant.u_max),
              readout_w=w, readout_b=b)
    return tr


def event_steps(traj: dict, event: str, cue_delay: int | None = None) -> np.ndarray:
    """Step index of ``event`` in each trial (-1 where it never happened)."""
    if event == "go":
        return np.asarray(traj["go_step"], dtype=int)
    if event == "cue":
        cd = traj.get("cue_delay") if cue_delay is None else cue_delay
        if cd is None:
            raise ValueError("pass cue_delay (or use a rollout from data.record)")
        return np.asarray(traj["go_step"], dtype=int) + int(cd)
    if event == "end":
        return np.asarray(traj["end_step"], dtype=int)
    for name, key in (("entry", "entry_step"), ("acq", "acq_step")):
        if event.startswith(name):
            k = int(event[len(name):]) - 1
            return np.asarray(traj[key][:, k], dtype=int)
    raise ValueError(f"unknown event {event!r}; one of {EVENTS}")


def align(traj: dict, event: str, window=(10, 20), field: str = "h",
          cue_delay: int | None = None):
    """Cut ``field`` around ``event`` for every trial.

    ``window = (before, after)`` in steps: lags ``-before .. after`` inclusive.
    Returns ``(X, lags)`` with ``X`` of shape (trials, lags, units) -- or (trials, lags)
    for scalar fields such as ``"u"`` or ``"x"`` -- and NaN wherever the step falls
    before the trial's first step, after its last (``end_step``; later steps belong to
    the next trial after the env's auto-reset), or the event never happened.
    Multiply ``lags`` by ``traj["dt"]`` for seconds.
    """
    data = np.asarray(traj[field], dtype=float)                   # (T, N[, H])
    T, N = data.shape[:2]
    ev = event_steps(traj, event, cue_delay)
    lags = np.arange(-window[0], window[1] + 1)
    k = ev[:, None] + lags[None, :]                                 # (N, L)
    ok = (ev[:, None] >= 0) & (k >= 0) & (k < T) & (k <= np.asarray(traj["end_step"])[:, None])
    X = data[np.clip(k, 0, T - 1), np.arange(N)[:, None]]         # (N, L[, H])
    X[~ok] = np.nan
    return X, lags


def balanced_trials(X, labels, n_per: int | None = None, seed: int = 0):
    """Arrange single trials ``X`` (trials, ...) into ``(repeats, conditions, ...)``
    with the same number of repeats per condition.

    As the original ``CVdPCA``, conditions with fewer trials are topped up by drawing
    some of their trials again (without replacement where possible); ``n_per``
    defaults to the largest count.  Returns ``(array, levels)``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    levels, inv = np.unique(np.asarray(labels).reshape(-1), return_inverse=True)
    counts = np.bincount(inv)
    n_per = int(counts.max()) if n_per is None else int(n_per)
    out = np.empty((n_per, len(levels)) + X.shape[1:])
    for c in range(len(levels)):
        idx = np.flatnonzero(inv == c)
        if len(idx) >= n_per:
            pick = rng.choice(idx, n_per, replace=False)
        else:
            extra = rng.choice(idx, n_per - len(idx), replace=n_per - len(idx) > len(idx))
            pick = np.concatenate([idx, extra])
        out[:, c] = X[pick]
    return out, levels


def potent_fraction(X, w) -> float:
    """Share of the variance of ``X`` (..., units) along the readout direction ``w``.
    A random direction would get ``1 / n_units`` on average."""
    from .utils import subspace_variance
    return subspace_variance(X, np.asarray(w).reshape(-1, 1))
