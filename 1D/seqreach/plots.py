"""Figures for the 1D testbed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.frameon": False,
})


def _alive_slice(traj, i):
    n = int(traj["end_step"][i]) + 1
    return slice(0, max(n, 1))


def plot_training(run_dirs: dict[str, Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), constrained_layout=True)
    for label, rd in run_dirs.items():
        log = Path(rd) / "train_log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        s = [r["global_step"] for r in rows]
        for ax, key, name in zip(
            axes,
            ("episodic_return", "success_rate", "hit_fraction"),
            ("episodic return", "sequence success rate", "fraction of targets hit"),
        ):
            y = [r[key] for r in rows]
            ax.plot(s, y, lw=1.0, label=label)
            ax.set_xlabel("environment steps")
            ax.set_title(name)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_example_trials(traj: dict, out: Path, n: int = 6) -> None:
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.6), sharex=True,
                             constrained_layout=True)
    dt = traj["dt"]
    for j in range(n):
        i = j
        sl = _alive_slice(traj, i)
        t = np.arange(sl.stop) * dt
        ax = axes[0, j]
        ax.plot(t, traj["x"][sl, i], "k", lw=1.2)
        for k in range(int(traj["n_targets"][i])):
            ax.axhline(traj["targets"][sl.stop - 1, i, k], color=f"C{k}", ls="--", lw=0.8)
        ax.axvline(traj["go_step"][i] * dt, color="g", lw=0.8)
        ax.set_title(f"trial {i}", fontsize=7)
        if j == 0:
            ax.set_ylabel("position")
        axes[1, j].plot(t, traj["v"][sl, i], "k", lw=1.0)
        axes[1, j].axvline(traj["go_step"][i] * dt, color="g", lw=0.8)
        axes[1, j].set_xlabel("time (s)")
        if j == 0:
            axes[1, j].set_ylabel("velocity")
    fig.suptitle("example trials (green = go cue, dashed = targets)", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_coarticulation(traj: dict, design: dict, res: dict, out: Path,
                        direction: float = 1.0) -> None:
    """The headline figure: does T2 reshape the reach to T1?"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), constrained_layout=True)
    dt = traj["dt"]

    for col, (level, name) in enumerate(((2, "T2 visible"), (1, "T2 hidden"))):
        ax = axes[col]
        sel = np.flatnonzero((design["preview"] == level)
                             & (design["direction"] == direction))
        t2 = design["t2"][sel]
        norm = plt.Normalize(t2.min(), t2.max())
        cmap = plt.cm.coolwarm
        for i, tt in zip(sel, t2):
            a = int(traj.get("entry_step", traj["acq_step"])[i, 0])
            if a < 0:
                continue
            g = int(traj["go_step"][i])
            ax.plot(np.arange(g, a + 1) * dt - g * dt, traj["v"][g:a + 1, i],
                    color=cmap(norm(tt)), lw=0.8, alpha=0.9)
        ax.set_title(f"{name}\nvelocity during reach to T1")
        ax.set_xlabel("time from go (s)")
        ax.set_ylabel("velocity" if col == 0 else "")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="T2 position", pad=0.02)

    ax = axes[2]
    for name, key, color in (("T2 visible", "preview_on", "C3"),
                             ("T2 hidden", "preview_off", "C0")):
        r = res[key]
        ax.scatter(r["d_next"], r["v_out"], s=8, alpha=0.6, color=color, label=name)
        if np.isfinite(r["ci"]):
            xs = np.linspace(r["d_next"].min(), r["d_next"].max(), 10)
            ax.plot(xs, r["ci"] * xs + np.mean(r["v_out"] - r["ci"] * r["d_next"]),
                    color=color, lw=1.5)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("next-target offset  (T2−T1)·sign(T1)")
    ax.set_ylabel("velocity at T1 acquisition\n(outward positive)")
    ax.set_title(f"co-articulation index\nCI={res['preview_on']['ci']:.3f} "
                 f"(visible) vs {res['preview_off']['ci']:.3f} (hidden)")
    ax.legend(fontsize=7)

    ax = axes[3]
    div = metrics.trajectory_divergence(traj, design)
    for dir_val, d in div.items():
        sp = d["spread"]
        t = (np.arange(len(sp)) - d["go_step"]) * dt
        ax.plot(t, sp, lw=1.0, label=f"direction {dir_val:+.0f}")
        if d["divergence_s"] is not None:
            ax.axvline(d["divergence_s"], ls="--", lw=0.8, color="k")
    ax.axvline(0, color="g", lw=0.8)
    ax.set_xlabel("time from go (s)")
    ax.set_ylabel("spread of x across T2 conditions")
    ax.set_title("when does T2 start\nshaping the movement?")
    ax.legend(fontsize=7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_reaction_times(rt: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.6), constrained_layout=True)
    rows = rt["by_delay"]
    d = [r["delay_s"] for r in rows]
    m = [r["rt_mean"] for r in rows]
    e = [r["rt_sem"] for r in rows]
    ax.errorbar(d, m, yerr=e, marker="o", lw=1.2, capsize=2, color="k")
    ax.set_xlabel("hold duration before go cue (s)")
    ax.set_ylabel("reaction time (s)")
    ttl = "RT vs preparation time"
    if "slope" in rt:
        ttl += f"\nslope={rt['slope']*1000:.1f} ms/s, p={rt['p']:.3f}"
    ax.set_title(ttl)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_jump(traj: dict, design: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), constrained_layout=True)
    dt = traj["dt"]
    kinds, signs = design["jump_kind"], design["jump_sign"]
    labels = {0: "no jump", 1: "T1 jumps (current)", 2: "T2 jumps (future)"}
    colors = {0: "k", 1: "C1", 2: "C2"}

    for ax, sgn in zip(axes, np.unique(signs)):
        for kind in (0, 1, 2):
            sel = np.flatnonzero((kinds == kind) & (signs == sgn))
            if sel.size == 0:
                continue
            i = sel[0]
            sl = _alive_slice(traj, i)
            t = np.arange(sl.stop) * dt
            ax.plot(t, traj["x"][sl, i], color=colors[kind], lw=1.3, label=labels[kind])
            a0 = int(traj["acq_step"][i, 0])
            if a0 > 0:
                ax.plot(a0 * dt, traj["x"][a0, i], "o", ms=4, color=colors[kind])
        js = int(traj["jump_step"][np.flatnonzero(signs == sgn)[0]])
        ax.axvline(js * dt, color="r", ls=":", lw=1.0)
        ax.axvline(traj["go_step"][0] * dt, color="g", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position")
        ax.set_title(f"jump size {sgn:+.2f}·δ  (red = jump, dot = T1 acquired)")
        ax.legend(fontsize=7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_preparatory(traj: dict, design: dict, prep: dict, out: Path,
                     window: int = 5) -> None:
    h_all, go = traj["h"], traj["go_step"]
    N = h_all.shape[1]
    idx = np.stack([np.clip(go - 1 - k, 0, h_all.shape[0] - 1) for k in range(window)])
    H = h_all[idx, np.arange(N)[None, :], :].mean(0)
    Hc = H - H.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(Hc, full_matrices=False)
    pcs = Hc @ vt[:2].T

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), constrained_layout=True)
    for ax, key, name in zip(axes[:2], ("t1", "t2"), ("T1 position", "T2 position")):
        if key not in design:
            continue
        c = design[key][:N]
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, cmap="coolwarm", s=10)
        fig.colorbar(sc, ax=ax, label=name, pad=0.02)
        r2 = prep.get(f"r2_{key}", np.nan)
        ax.set_title(f"delay-period state (PC1-2)\ncoloured by {name}, decode R²={r2:.2f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    ax = axes[2]
    ax.bar([0, 1], [prep["var_potent"], prep["var_random_expected"]],
           color=["C3", "0.6"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["along output\ndirection", "typical\ndirection"])
    ax.set_ylabel("across-trial variance")
    ax.set_title(f"is preparation output-null?\nratio={prep['potent_ratio']:.2f}, "
                 f"pct={prep['potent_percentile']:.2f}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_ablations(rows: list[dict], out: Path) -> None:
    """Bar chart of co-articulation index across ablation conditions.

    Arms below the competence floor are drawn hatched and washed out, and the floor is
    marked on the competence panel.  A reader should not be able to take a CI bar off
    this figure without also seeing that the agent could not do the task -- the first
    version of this plot presented all nine arms identically, which is how a sign-flipped
    index from a 19%-success agent ended up being read as a result.
    """
    rows = [r for r in rows if r.get("ci") is not None and np.isfinite(r["ci"])]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(6, 1.0 * len(rows)), 3.2),
                             constrained_layout=True)
    names = [r["name"] for r in rows]
    ci = [r["ci"] for r in rows]
    se = [r.get("ci_se") or 0.0 for r in rows]
    # ``competent`` may be absent in summaries written before the gate existed; treat
    # unknown as competent so old files still plot, rather than silently greying out.
    good = [r.get("competent", True) for r in rows]
    floor = next((r.get("competence_floor") for r in rows
                  if r.get("competence_floor") is not None), 0.90)

    base = ["C3" if n == "baseline" else "C0" for n in names]
    colors = [c if g else "0.75" for c, g in zip(base, good)]
    hatch = ["" if g else "///" for g in good]

    bars = axes[0].bar(range(len(rows)), ci, yerr=se, color=colors, capsize=2)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel("co-articulation index")
    axes[0].set_title("lookahead across ablations")

    succ = [r.get("success_rate", np.nan) for r in rows]
    bars = axes[1].bar(range(len(rows)), succ, color=colors)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    axes[1].axhline(floor, color="k", ls="--", lw=0.8)
    axes[1].text(len(rows) - 0.4, floor, f" floor {floor:.2f}", va="bottom", ha="right",
                 fontsize=6)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("sequence success rate")
    axes[1].set_title("task competence (gate)")

    if not all(good):
        fig.suptitle("hatched / grey = below competence floor; CI not interpretable",
                     fontsize=7, y=1.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
