"""Test experiments: named, reproducible protocols run on a frozen agent.

An *experiment* here means what it means in the lab: a trial list with a factorial
design, run on a subject that is not learning during it. The agent's weights are
loaded from a checkpoint and never updated; each experiment declares its own trials
and the factor levels that label them.

Nothing in this file is used during training.

---------------------------------------------------------------------------------
Adding your own experiment
---------------------------------------------------------------------------------

Write a function that returns a ``Design``, and decorate it::

    @experiment("my_probe", "one line describing what it manipulates")
    def my_probe(cfg, *, spacing: float = 0.3):
        trials = TrialList(cfg)
        for direction in (+1.0, -1.0):
            for t2 in np.linspace(-0.9, 0.9, 25):
                trials.add(
                    targets=[0.45 * direction, t2],   # T1, T2, ...
                    preview=2,                        # how many targets are visible
                    delay_steps=25,                   # hold duration before the go cue
                    factors={"direction": direction, "t2": t2},
                )
        return trials.build()

It is then runnable as ``run_experiment.py --exp my_probe``, and any keyword argument
you declare is settable from the command line as ``--param spacing=0.4``.

``factors`` is free-form: whatever you put there becomes a column in the output
dataframe, one value per trial. Put the independent variables of your design there —
that is what you will group by during analysis.

---------------------------------------------------------------------------------
Conventions worth knowing
---------------------------------------------------------------------------------

* Trials are **deterministic** unless you ask otherwise: motor noise off, policy at its
  mean. Two trials with identical settings are then bit-identical, so any difference
  between conditions is causal. ``--noisy`` and ``--stochastic`` relax that.
* ``preview`` is a *count*, not a flag: 1 = only the current target is visible, 2 = the
  current one and the next, and so on. It is the manipulation for anything about
  advance information.
* ``delay_steps`` is the hold period before the go cue, in simulation steps (dt = 20 ms),
  so 25 steps = 0.5 s. The agent must stay near the start until then.
* Targets are acquired **in order**, and the trial ends when the last one is taken (or at
  ``max_steps``). Give as many targets as the config's ``max_targets`` allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import Config
from .env import JUMP_CURRENT, JUMP_FUTURE, JUMP_NONE

__all__ = ["Design", "TrialList", "experiment", "EXPERIMENTS", "build",
           "JUMP_NONE", "JUMP_CURRENT", "JUMP_FUTURE"]


@dataclass
class Design:
    """A built trial list: what the environment needs, plus how trials are labelled.

    ``spec`` is consumed by the environment; ``factors`` is metadata that rides along
    into the output dataframe and is never seen by the agent.
    """

    spec: dict
    factors: dict
    name: str = ""
    description: str = ""
    params: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.spec["n_targets"])


class TrialList:
    """Accumulates trials one at a time, then packs them into a ``Design``.

    Deliberately un-clever: a Python loop appending dicts. Experiments are read far
    more often than they are run, and a readable design is worth more than a fast one.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.K = cfg.task.max_targets
        self._rows: list[dict] = []

    def add(self, targets, *, preview: int | None = None, delay_steps: int = 25,
            jump_kind: int = JUMP_NONE, jump_step: int | None = None,
            jump_delta: float = 0.0, factors: dict | None = None,
            repeats: int = 1) -> "TrialList":
        """Append one trial (or ``repeats`` copies of it).

        targets      list of target positions, in acquisition order (1..max_targets)
        preview      how many targets are visible at once; default = all of them
        delay_steps  hold duration before the go cue, in steps (dt = 20 ms)
        jump_*       optional mid-trial target displacement; see ``jump_probe``
        factors      per-trial labels that become dataframe columns
        """
        targets = list(np.atleast_1d(np.asarray(targets, dtype=float)))
        if not 1 <= len(targets) <= self.K:
            raise ValueError(f"need 1..{self.K} targets, got {len(targets)}")
        row = np.zeros(self.K)
        row[: len(targets)] = targets
        for _ in range(repeats):
            self._rows.append({
                "targets": row.copy(),
                "n_targets": len(targets),
                "preview": len(targets) if preview is None else int(preview),
                "delay_steps": int(delay_steps),
                "jump_kind": int(jump_kind),
                "jump_step": int(delay_steps + 8 if jump_step is None else jump_step),
                "jump_delta": float(jump_delta),
                "factors": dict(factors or {}),
            })
        return self

    def add_grid(self, **kwargs) -> "TrialList":
        """Convenience: ``add`` with every combination of the list-valued arguments.

        Scalars are held fixed, sequences are crossed.  Crossed values are automatically
        recorded as factors, so ``add_grid(t2=[...], preview=[1, 2])`` gives you ``t2``
        and ``preview`` columns without listing them twice.
        """
        import itertools

        varying = {k: list(v) for k, v in kwargs.items()
                   if k != "factors" and isinstance(v, (list, tuple, np.ndarray))
                   and k != "targets"}
        fixed = {k: v for k, v in kwargs.items() if k not in varying}
        base_factors = fixed.pop("factors", {}) or {}
        if not varying:
            return self.add(factors=base_factors, **fixed)
        keys = list(varying)
        for combo in itertools.product(*(varying[k] for k in keys)):
            kv = dict(zip(keys, combo))
            self.add(factors={**base_factors, **kv}, **kv, **fixed)
        return self

    def build(self, name: str = "", description: str = "",
              params: dict | None = None) -> Design:
        if not self._rows:
            raise ValueError("experiment produced no trials")
        n = len(self._rows)
        spec = {
            "targets": np.stack([r["targets"] for r in self._rows]),
            "n_targets": np.array([r["n_targets"] for r in self._rows]),
            "preview": np.array([r["preview"] for r in self._rows]),
            "delay_steps": np.array([r["delay_steps"] for r in self._rows]),
            "jump_kind": np.array([r["jump_kind"] for r in self._rows]),
            "jump_step": np.array([r["jump_step"] for r in self._rows]),
            "jump_delta": np.array([r["jump_delta"] for r in self._rows]),
        }
        # Union of factor names, so trials that omit one get NaN rather than failing.
        names: list[str] = []
        for r in self._rows:
            for k in r["factors"]:
                if k not in names:
                    names.append(k)
        factors = {}
        for k in names:
            vals = [r["factors"].get(k, np.nan) for r in self._rows]
            factors[k] = (np.array(vals) if all(isinstance(v, (int, float, np.number))
                                                for v in vals)
                          else np.array(vals, dtype=object))
        factors["n_targets"] = spec["n_targets"]
        factors["preview"] = spec["preview"]
        factors["delay_steps"] = spec["delay_steps"]
        for k in range(self.K):
            factors[f"T{k + 1}"] = np.where(spec["n_targets"] > k,
                                            spec["targets"][:, k], np.nan)
        assert all(len(v) == n for v in factors.values())
        return Design(spec=spec, factors=factors, name=name, description=description,
                      params=dict(params or {}))


# --------------------------------------------------------------------- registry
EXPERIMENTS: dict[str, dict] = {}


def experiment(name: str, description: str = ""):
    """Register an experiment builder under ``name``."""

    def deco(fn: Callable[..., Design]):
        EXPERIMENTS[name] = {"fn": fn, "description": description or (fn.__doc__ or "")}
        return fn

    return deco


def build(name: str, cfg: Config, **params) -> Design:
    """Build a registered experiment, passing ``params`` to its builder."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{name}'. "
                       f"available: {', '.join(sorted(EXPERIMENTS))}")
    entry = EXPERIMENTS[name]
    design = entry["fn"](cfg, **params)
    design.name = name
    design.description = entry["description"].strip().splitlines()[0] if entry["description"] else ""
    design.params = params
    return design


# ===================================================================== library
# The protocols the analysis in `analyze.py` already relies on, plus a few obvious
# extensions.  Copy one of these as the template for your own.


@experiment("preview_sweep",
            "Fix T1, sweep T2, crossed with T2 visible vs hidden and reach direction.")
def preview_sweep(cfg: Config, *, t1: float = 0.45, n_t2: int = 25,
                  delay_steps: int = 25, repeats: int = 1,
                  both_directions: bool = True) -> Design:
    """The co-articulation protocol.

    The preview contrast is the causal control: with T2 hidden the observation is
    identical for every T2, so trajectories are bit-identical and any T2-dependence
    there would mean the pipeline is broken.
    """
    t = cfg.task
    trials = TrialList(cfg)
    grid = np.linspace(-t.target_range, t.target_range, n_t2)
    for direction in ([1.0, -1.0] if both_directions else [1.0]):
        a = t1 * direction
        for b in grid:
            if abs(b - a) < t.min_separation or abs(b - t.start_x) < t.min_separation:
                continue
            for h in (2, 1):  # 2 = T2 visible from onset, 1 = hidden until T1 is taken
                trials.add([a, b], preview=h, delay_steps=delay_steps, repeats=repeats,
                           factors={"direction": direction, "t1": a, "t2": b,
                                    # Signed offset of T2 folded by reach direction:
                                    # > 0 means T2 continues onward past T1.
                                    "d_next": (b - a) * np.sign(a)})
    return trials.build(params=dict(t1=t1, n_t2=n_t2, delay_steps=delay_steps,
                                    repeats=repeats, both_directions=both_directions))


@experiment("delay_sweep",
            "Random 2-target trials crossed with the duration of the pre-go hold.")
def delay_sweep(cfg: Config, *, delays: tuple[int, ...] = (5, 12, 25, 37, 50),
                n_per: int = 40, seed: int = 0) -> Design:
    """Reaction time and preparatory geometry vs preparation time."""
    rng = np.random.default_rng(seed)
    t = cfg.task
    trials = TrialList(cfg)
    for d in delays:
        for _ in range(n_per):
            while True:
                a, b = rng.uniform(-t.target_range, t.target_range, 2)
                if (abs(a - t.start_x) > t.min_separation
                        and abs(b - a) > t.min_separation):
                    break
            trials.add([a, b], preview=2, delay_steps=int(d),
                       factors={"delay_s": d * cfg.plant.dt, "t1": a, "t2": b,
                                "direction": float(np.sign(a))})
    return trials.build(params=dict(delays=list(delays), n_per=n_per, seed=seed))


@experiment("target_jump",
            "Displace the current or a future target mid-movement (double-step).")
def target_jump(cfg: Config, *, t1: float = 0.45, t2: float = -0.45,
                delta: float = 0.35, jump_after_go: int = 8,
                delay_steps: int = 25, repeats: int = 1) -> Design:
    """Online replanning.

    ``jump_future`` is the sharp condition: only T2 moves, so any change in the ongoing
    reach to T1 is evidence the agent was planning through T2.  The no-jump rows are the
    matched controls -- identical trials with the perturbation switched off.
    """
    trials = TrialList(cfg)
    for kind, label in ((JUMP_NONE, "none"), (JUMP_CURRENT, "current"),
                        (JUMP_FUTURE, "future")):
        for sign in (1.0, -1.0):
            trials.add([t1, t2], preview=2, delay_steps=delay_steps,
                       jump_kind=kind, jump_step=delay_steps + jump_after_go,
                       jump_delta=sign * delta, repeats=repeats,
                       factors={"jump_kind": label, "jump_sign": sign,
                                "jump_delta": sign * delta,
                                "jump_step": delay_steps + jump_after_go})
    return trials.build(params=dict(t1=t1, t2=t2, delta=delta,
                                    jump_after_go=jump_after_go,
                                    delay_steps=delay_steps, repeats=repeats))


@experiment("random_task",
            "The training distribution itself -- competence and failure modes.")
def random_task(cfg: Config, *, n_trials: int = 512, seed: int = 123) -> Design:
    """Trials drawn exactly as training draws them.

    Use this for success rates and failure profiles; the structured experiments above
    fix T1 and so cannot speak to competence on the task as posed.
    """
    rng = np.random.default_rng(seed)
    t, p = cfg.task, cfg.plant
    w = np.asarray(t.n_target_weights, dtype=float)
    w = w / w.sum()
    lo, hi = int(t.delay_min_s / p.dt), int(t.delay_max_s / p.dt)
    trials = TrialList(cfg)
    for _ in range(n_trials):
        n_tg = int(rng.choice(len(w), p=w) + 1)
        prev, tgs = t.start_x, []
        for _ in range(n_tg):
            while True:
                c = rng.uniform(-t.target_range, t.target_range)
                if abs(c - prev) >= t.min_separation:
                    break
            tgs.append(c)
            prev = c
        full = rng.random() < t.full_preview_prob
        h = n_tg if full else int(rng.integers(1, max(n_tg, 2)))
        trials.add(tgs, preview=min(h, n_tg),
                   delay_steps=int(rng.integers(lo, hi + 1)) if t.use_delay else 0,
                   factors={"full_preview": bool(full)})
    return trials.build(params=dict(n_trials=n_trials, seed=seed))


@experiment("sequence_length",
            "Matched T1, varying how many targets follow it (1 vs 2 vs 3).")
def sequence_length(cfg: Config, *, t1: float = 0.45, n_rest: int = 12,
                    delay_steps: int = 25, seed: int = 0) -> Design:
    """Does the first reach change with how much sequence remains after it?

    Not part of the original battery.  Included as a worked second example of a design
    the co-articulation protocol cannot express, since that one fixes the trial at two
    targets.
    """
    rng = np.random.default_rng(seed)
    t = cfg.task
    trials = TrialList(cfg)
    for direction in (1.0, -1.0):
        a = t1 * direction
        for n_tg in (1, 2, 3):
            for _ in range(n_rest):
                tgs, prev = [a], a
                for _ in range(n_tg - 1):
                    while True:
                        c = rng.uniform(-t.target_range, t.target_range)
                        if abs(c - prev) >= t.min_separation:
                            break
                    tgs.append(c)
                    prev = c
                trials.add(tgs, preview=n_tg, delay_steps=delay_steps,
                           factors={"direction": direction, "t1": a,
                                    "n_following": n_tg - 1})
    return trials.build(params=dict(t1=t1, n_rest=n_rest, delay_steps=delay_steps,
                                    seed=seed))
