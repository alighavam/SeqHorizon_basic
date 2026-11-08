"""Small helpers shared by the lesson notebooks.

None of this is research code.  It is plumbing that would otherwise be copy-pasted into
every lesson: putting ``1D/`` on the import path, a plot style, two hand-written
controllers, and ``run_trial`` -- the env.reset / env.step loop that lesson 1 writes out
by hand, packaged so later lessons can reuse it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # .../SeqHorizon_basic/1D
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqreach.env import BatchedSeqReach1D  # noqa: E402

RUNS = ROOT / "runs"

# ------------------------------------------------------------------ plot style
# Categorical colours in a fixed order (colour-blind-safe for up to three series).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BAND = "#f0efec"  # target tolerance bands
# Diverging pair for signed quantities (red = one way, blue = the other, grey = zero).
DIVERGING = ["#c23b3a", "#e88a89", "#d9d8d4", "#6da7ec", "#1c5cab"]


def use_style() -> None:
    """Plot style for the lessons.  Call it after the imports.

    ``seqreach/plots.py`` and ``seqreach/video.py`` switch matplotlib to its file-only
    "Agg" backend when imported (right for scripts, which only save figures).  In a
    notebook that silently stops figures from showing, so switch inline display back on.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
    plt.rcParams.update({
        "figure.figsize": (8, 3.2), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA]),
    })


def show(section) -> pd.DataFrame:
    """A config section (e.g. ``cfg.reward``) as a two-column table."""
    return pd.DataFrame({"value": vars(section)}).rename_axis("key")


# ----------------------------------------------------------------- controllers
# A controller is any function  state -> force.  ``state`` is a dict with the current
# x, v, a, t (step), go (has the go cue fired?), targets, cur (index of the target
# being reached for) and n_targets.  These are *hand-written*, not learned: they exist
# so you can see what the environment and reward do before any learning happens.

def pd_controller(kp: float = 60.0, kd: float = 14.0):
    """Wait for the go cue, then spring-and-damper towards the current target.

    It arrives at each target (nearly) at rest -- a "stopper".
    """
    def ctrl(s):
        if not s["go"]:
            return 0.0
        return kp * (s["targets"][s["cur"]] - s["x"]) - kd * s["v"]
    return ctrl


def passer_controller(kp: float = 60.0, kd: float = 14.0):
    """Like ``pd_controller``, but if the *next* target lies further along the same
    direction, aim straight at it -- so the current target is crossed at speed.

    This is the simplest possible form of lookahead: the reach to T1 depends on T2.
    """
    def ctrl(s):
        if not s["go"]:
            return 0.0
        cur, tg = s["cur"], s["targets"]
        aim = tg[cur]
        if cur + 1 < s["n_targets"] and (tg[cur + 1] - tg[cur]) * (tg[cur] - s["x"]) > 0:
            aim = tg[cur + 1]
        return kp * (aim - s["x"]) - kd * s["v"]
    return ctrl


# ------------------------------------------------------------------ one trial
def run_trial(cfg, targets, controller, preview=None, delay_steps: int = 25,
              seed: int = 0):
    """Run ONE trial of the environment under ``controller``.

    Returns ``(df, episode)``: a DataFrame with one row per simulation step, and the
    environment's end-of-trial summary (``info["episode"]``).  This is exactly the
    reset/step loop written out in lesson 1, section 4.
    """
    targets = list(targets)
    K = cfg.task.max_targets
    row = np.zeros(K)
    row[:len(targets)] = targets
    env = BatchedSeqReach1D(cfg, num_envs=1, seed=seed)
    env.reset({"n_targets": [len(targets)], "targets": [row],
               "preview": [len(targets) if preview is None else preview],
               "delay_steps": [delay_steps], "jump_kind": [0]})
    u_max = cfg.plant.u_max
    rows, episode = [], None
    for _ in range(cfg.task.max_steps):
        t, cur = int(env.t[0]), int(env.cur[0])
        state = {"x": env.x[0], "v": env.v[0], "a": env.a[0], "t": t,
                 "go": t >= env.go_step[0], "targets": env.targets[0].copy(),
                 "cur": cur, "n_targets": int(env.n_targets[0])}
        u = float(np.clip(controller(state), -u_max, u_max))
        target_now = env.targets[0, cur]
        obs, r, term, trunc, info = env.step(np.array([u]))
        done = bool(term[0] or trunc[0])
        if done:  # the env already reset itself; read the final state from info
            fs, ep = info["final_state"], info["episode"]
            x, v, a, cur_after = fs["x"][0], fs["v"][0], fs["a"][0], int(ep["cur"][0])
            episode = {k: (val[0].item() if hasattr(val[0], "item") else val[0])
                       for k, val in ep.items()}
        else:
            x, v, a, cur_after = env.x[0], env.v[0], env.a[0], int(env.cur[0])
        rows.append({"step": t, "time_s": t * cfg.plant.dt, "go": state["go"],
                     "u": u, "a": a, "v": v, "x": x, "target_now": target_now,
                     "cur": cur, "acquired": cur_after > cur, "reward": float(r[0])})
        if done:
            break
    return pd.DataFrame(rows), episode


def plot_trial(df, targets, cfg, title: str = "", axes=None, color=BLUE, label=None):
    """Position (with target tolerance bands) and velocity against time."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    ax_x, ax_v = axes
    tol = cfg.task.tolerance
    for k, tg in enumerate(targets):
        ax_x.axhspan(tg - tol, tg + tol, color=BAND, zorder=0)
        ax_x.text(0.002, tg, f" T{k + 1}", transform=ax_x.get_yaxis_transform(),
                  va="center", color=INK_SOFT, fontsize=9)
    go_s = df.loc[df.go, "time_s"].min() if df.go.any() else None
    for ax in axes:
        if go_s is not None:
            ax.axvline(go_s, color=INK_SOFT, lw=1, ls="--")
    ax_x.plot(df.time_s, df.x, color=color, label=label)
    hit = df[df.acquired]
    ax_x.plot(hit.time_s, hit.x, "o", color=color, ms=7, mec="white", mew=1.5)
    ax_v.plot(df.time_s, df.v, color=color, label=label)
    ax_v.axhline(0, color=INK_SOFT, lw=0.8)
    ax_x.set_ylabel("position x")
    ax_v.set_ylabel("velocity v")
    ax_v.set_xlabel("time (s)")
    if title:
        ax_x.set_title(title)
    if go_s is not None:
        ax_x.text(go_s, 0.03, " go", transform=ax_x.get_xaxis_transform(),
                  va="bottom", color=INK_SOFT, fontsize=9)
    return axes
