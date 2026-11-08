#!/usr/bin/env python
"""Film N single-target trials back to back, with control signals and RT on screen.

    python 1D/scripts/single_target_video.py --run 1D/runs/baseline_s1
    python 1D/scripts/single_target_video.py --run 1D/runs/baseline_s1 --trials 10 --seed 3

Each trial has exactly one target at a random position (task distribution, min
separation from the start) and a random go cue (task distribution, 0.25-1.0 s hold).
The policy runs deterministically (mean action) on a noiseless plant unless --noisy /
--stochastic are given.

Panels: the track (mass, target, speed-coloured tail); position vs time; velocity vs
time; the policy's force command u and the lagged actuator force a.  The side box shows
the trial's target, hold duration, reaction time, movement time and peak speed, each
revealed only once it has happened.

RT follows ``metrics.segment_kinematics``: time from the go cue to the first step at
which |v| reaches 5% of that reach's peak speed.  Steps are 20 ms, so RT is quantised
to 20 ms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from seqreach import probes, video
from seqreach.ppo import load_agent

CMAP = matplotlib.colormaps["coolwarm"]
GO = "#2f8f3f"
HOLD = "#c08a2a"
TGT = "#e0913a"
ONSET = "#7b3fb4"


def trial_info(traj, i, dt):
    """Per-trial timing: go, movement onset (RT), entry into target, end."""
    end = int(traj["end_step"][i])
    go = int(traj["go_step"][i])
    entry = int(traj["entry_step"][i, 0])
    v = traj["v"][: end + 1, i]
    onset = None
    peak = np.nan
    if entry >= go:
        seg = v[go: entry + 1]
        j = int(np.argmax(np.abs(seg)))
        peak = float(seg[j])
        above = np.flatnonzero(np.abs(seg) >= 0.05 * abs(peak))
        if above.size:
            onset = go + int(above[0])
    return {
        "T": end + 1, "go": go, "entry": entry if entry >= 0 else None,
        "start": float(traj["x"][0, i] - traj["v"][0, i] * dt),
        "false_start": bool(traj["false_start"][i]),
        "onset": onset, "peak": peak, "target": float(traj["targets0"][i, 0]),
        "success": bool(traj["success"][i]),
        "rt_ms": (onset - go) * dt * 1000 if onset is not None else None,
        "mt_ms": (entry - onset) * dt * 1000 if (onset is not None and entry >= 0) else None,
        "hold_ms": go * dt * 1000,
    }


def render(traj, cfg, title, stride, hold_frames, width_px):
    dt, tol, u_max = cfg.plant.dt, cfg.task.tolerance, cfg.plant.u_max
    N = traj["x"].shape[1]
    infos = [trial_info(traj, i, dt) for i in range(N)]
    T_max = max(d["T"] for d in infos)
    lim = cfg.task.target_range + 0.15
    vmax = max(1.0, float(np.nanmax([abs(d["peak"]) for d in infos])) * 1.15)
    speed_norm = plt.Normalize(0, vmax)

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, width_px / dpi * 0.78), dpi=dpi,
                     facecolor="white")
    L, W = 0.07, 0.62
    frames = []
    for i, d in enumerate(infos):
        T = d["T"]
        x = traj["x"][:T, i]
        v = traj["v"][:T, i]
        u = traj["u"][:T, i]
        a = traj["a"][:T, i]
        tt = np.arange(T) * dt
        steps = list(range(0, T, stride))
        if steps[-1] != T - 1:
            steps.append(T - 1)

        for k, t in enumerate(steps + [T - 1] * hold_frames):
            fig.clf()
            ax_s = fig.add_axes((L, 0.80, W, 0.10))
            ax_x = fig.add_axes((L, 0.555, W, 0.20))
            ax_v = fig.add_axes((L, 0.33, W, 0.18))
            ax_u = fig.add_axes((L, 0.085, W, 0.20))

            # --- header
            fig.text(L, 0.955, f"{title}   ·   trial {i + 1}/{N}   ·   1 target",
                     fontsize=10, ha="left", va="center", color="#222")
            fig.text(L + W, 0.955, f"t = {t * dt:4.2f} s", fontsize=11, ha="right",
                     va="center", family="monospace", color="#111")
            if t < d["go"]:
                phase, pc = "HOLD  (waiting for go cue)", HOLD
            elif d["entry"] is not None and t >= d["entry"]:
                phase, pc = ("IN TARGET  (final dwell 0.3 s)" if t < T - 1
                             else ("SUCCESS ✓" if d["success"] else "FAILED")), GO
            elif t == T - 1 and not d["success"]:
                phase, pc = ("FALSE START ✗" if d["false_start"] else "TIMEOUT ✗"), "#c0392b"
            else:
                phase, pc = "GO  →  reach", GO
            fig.text(L, 0.915, phase, fontsize=10, ha="left", va="center",
                     color=pc, fontweight="bold")

            # --- scene
            ax_s.set_xlim(-lim, lim)
            ax_s.set_ylim(-0.12, 0.12)
            ax_s.set_aspect("equal", adjustable="box")
            ax_s.set_yticks([])
            for s in ("top", "right", "left"):
                ax_s.spines[s].set_visible(False)
            ax_s.tick_params(axis="x", labelsize=7, length=2, colors="#888")
            ax_s.axhline(0, color="#dcdcdc", lw=3, zorder=1)
            ax_s.plot([d["start"]], [0], marker="|", ms=12, mew=1.5, color="#666",
                      zorder=2)
            ax_s.text(d["start"], -0.11, "start", fontsize=6.5, ha="center",
                      va="bottom", color="#666")
            if abs(d["start"]) > 1e-9:
                ax_s.plot([0], [0], marker="|", ms=7, color="#ccc", zorder=2)
            ax_s.add_patch(Circle((d["target"], 0), tol, facecolor=TGT + "33",
                                  edgecolor=TGT, lw=2, zorder=3))
            lo = max(0, t - 15)
            if t - lo >= 1:
                pts = np.stack([x[lo:t + 1], np.zeros(t + 1 - lo)], 1)
                lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap=CMAP,
                                    norm=speed_norm, linewidths=4, zorder=4)
                lc.set_array(np.abs(v[lo:t]))
                ax_s.add_collection(lc)
            ax_s.add_patch(Circle((x[t], 0), 0.03, facecolor=CMAP(speed_norm(abs(v[t]))),
                                  edgecolor="#111", lw=1, zorder=5))

            # --- time panels
            for ax in (ax_x, ax_v, ax_u):
                ax.set_xlim(0, (T_max - 1) * dt)
                ax.tick_params(labelsize=7, length=2)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                if t >= d["go"]:
                    ax.axvline(d["go"] * dt, color=GO, ls="--", lw=1)
                if d["onset"] is not None and t >= d["onset"]:
                    ax.axvline(d["onset"] * dt, color=ONSET, ls=":", lw=1.2)
                ax.axvline(t * dt, color="#bbb", lw=0.8)
            if t >= d["go"]:
                ax_x.text(d["go"] * dt, 1.02, " go", transform=ax_x.get_xaxis_transform(),
                          fontsize=7, color=GO, va="bottom")
            if d["onset"] is not None and t >= d["onset"]:
                ax_v.text(d["onset"] * dt, 0.97, " onset",
                          transform=ax_v.get_xaxis_transform(), fontsize=7,
                          color=ONSET, va="top", ha="left")

            ax_x.set_ylim(-lim, lim)
            ax_x.axhspan(d["target"] - tol, d["target"] + tol, color=TGT, alpha=0.18)
            ax_x.axhline(d["start"], color="#888", lw=0.8, ls=":")
            ax_x.plot(tt[:t + 1], x[:t + 1], color="#2b6cb0", lw=1.8)
            ax_x.set_ylabel("position x", fontsize=8)
            ax_x.set_xticklabels([])

            ax_v.set_ylim(-vmax, vmax)
            ax_v.axhline(0, color="#ddd", lw=0.8)
            ax_v.plot(tt[:t + 1], v[:t + 1], color="#c0392b", lw=1.8)
            ax_v.set_ylabel("velocity v", fontsize=8)
            ax_v.set_xticklabels([])

            ax_u.set_ylim(-u_max * 1.08, u_max * 1.08)
            ax_u.axhline(0, color="#ddd", lw=0.8)
            for yv in (-u_max, u_max):
                ax_u.axhline(yv, color="#bbb", lw=0.8, ls=":")
            ax_u.plot(tt[:t + 1], u[:t + 1], color="#555", lw=1.2, label="command u")
            ax_u.plot(tt[:t + 1], a[:t + 1], color="#2f7a4b", lw=1.8,
                      label="actuator force a (50 ms lag)")
            ax_u.set_ylabel("force", fontsize=8)
            ax_u.set_xlabel("time (s)", fontsize=8)
            ax_u.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.98),
                        frameon=False, ncol=2, borderaxespad=0)

            # --- info box
            def fmt(val, unit, shown):
                return f"{val:6.0f} {unit}" if (shown and val is not None) else "     —"
            onset_seen = d["onset"] is not None and t >= d["onset"]
            entry_seen = d["entry"] is not None and t >= d["entry"]
            lines = [
                ("start", f"{d['start']:+6.2f}"),
                ("target", f"{d['target']:+6.2f}"),
                ("distance", f"{abs(d['target'] - d['start']):6.2f}"),
                ("hold (go delay)", fmt(d["hold_ms"], "ms", t >= d["go"])),
                ("reaction time", fmt(d["rt_ms"], "ms", onset_seen)),
                ("movement time", fmt(d["mt_ms"], "ms", entry_seen)),
                ("peak speed", f"{abs(d['peak']):6.2f}" if entry_seen else "     —"),
                ("", ""),
                ("x", f"{x[t]:+6.3f}"),
                ("v", f"{v[t]:+6.3f}"),
                ("u", f"{u[t]:+6.2f}"),
            ]
            y0 = 0.74
            fig.text(0.735, y0 + 0.045, "this trial", fontsize=9, fontweight="bold",
                     color="#222")
            for j, (k_, val) in enumerate(lines):
                yy = y0 - j * 0.04
                col = ONSET if k_ == "reaction time" else "#333"
                fig.text(0.735, yy, k_, fontsize=8.5, color=col)
                fig.text(0.975, yy, val, fontsize=8.5, family="monospace", ha="right",
                         color=col)

            # running table of completed trials
            done = [infos[m] for m in range(i)] + ([d] if t == T - 1 else [])
            fig.text(0.735, 0.30, "so far", fontsize=9, fontweight="bold", color="#222")
            fig.text(0.735, 0.27, " #  start  target  RT    MT", fontsize=7.5,
                     family="monospace", color="#777")
            for m, dd in enumerate(done[-6:]):
                n0 = len(done) - len(done[-6:]) + m + 1
                rt = f"{dd['rt_ms']:4.0f}" if dd["rt_ms"] is not None else "   —"
                mt = f"{dd['mt_ms']:4.0f}" if dd["mt_ms"] is not None else "   —"
                fig.text(0.735, 0.245 - m * 0.025,
                         f"{n0:2d}  {dd['start']:+5.2f}  {dd['target']:+5.2f}  {rt}  {mt}",
                         fontsize=7.5, family="monospace", color="#333")
            fig.text(0.735, 0.06, "times in ms · RT: go → |v| ≥ 5% peak\n"
                     "MT: onset → target entry · 1 step = 20 ms",
                     fontsize=6.5, color="#888", linespacing=1.5)

            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames, infos


def save_mp4(frames, path, fps):
    """Pipe raw RGB frames to ffmpeg as H.264 (plays and pauses in any player)."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg not found on PATH; use --format gif")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def make_video(run, trials=10, seed=0, start_range=0.0, fmt="mp4", name="single_target",
               noisy=False, stochastic=False, stride=2, width=900, out=None):
    """Roll out ``trials`` single-target trials and write the video.

    Returns ``(path, traj, infos)`` so callers (notebooks) can analyse the same trials
    they are watching.
    """
    run_dir = Path(run)
    agent, obs_rms, cfg = load_agent(run_dir)
    vcfg = cfg.video
    vcfg.noiseless = not noisy
    ecfg = video.video_config(cfg, vcfg)

    # One target, preview 1; target position and go cue drawn from the task distribution.
    spec = {"n_targets": 1, "preview": 1}
    if start_range > 0:
        rng = np.random.default_rng(seed)
        spec["start_x"] = rng.uniform(-start_range, start_range, trials)
    traj = probes.rollout(agent, obs_rms, ecfg, spec, n_trials=trials, seed=seed,
                          deterministic=not stochastic)

    title = f"{run_dir.name}" + ("  ·  noisy" if noisy else "") + \
            ("  ·  stochastic" if stochastic else "")
    frames, infos = render(traj, ecfg, title, stride, hold_frames=15, width_px=width)

    out = Path(out) if out else run_dir / "videos" / f"{name}.{fmt}"
    ms = int(round(1000 * ecfg.plant.dt * stride))  # real time
    if out.suffix == ".mp4":
        save_mp4(frames, out, fps=1000 / ms)
    else:
        video.save_gif(frames, out, duration_ms=ms)
    return out, traj, infos


def outcome(d):
    return "ok" if d["success"] else ("FALSE START" if d["false_start"] else "TIMEOUT")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="1D/runs/baseline_s1")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-range", type=float, default=0.0,
                    help="start position drawn from U[-r, r] (0 = trained start)")
    ap.add_argument("--name", default="single_target", help="output file stem")
    ap.add_argument("--noisy", action="store_true", help="keep motor noise on")
    ap.add_argument("--stochastic", action="store_true", help="sample actions")
    ap.add_argument("--stride", type=int, default=2, help="render every s-th step")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--format", default="gif", choices=["gif", "mp4"],
                    help="mp4 needs ffmpeg on PATH; ignored if --out has a suffix")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out, _, infos = make_video(a.run, a.trials, a.seed, a.start_range, a.format, a.name,
                               a.noisy, a.stochastic, a.stride, a.width, a.out)
    print(f"[video] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(" #  start   target   hold_ms  RT_ms  MT_ms  peak_v  outcome")
    for i, d in enumerate(infos):
        rt = f"{d['rt_ms']:5.0f}" if d["rt_ms"] is not None else "    -"
        mt = f"{d['mt_ms']:5.0f}" if d["mt_ms"] is not None else "    -"
        print(f"{i + 1:2d}  {d['start']:+.3f}  {d['target']:+.3f}  {d['hold_ms']:7.0f}  "
              f"{rt}  {mt}  {d['peak']:+6.2f}  {outcome(d)}")


if __name__ == "__main__":
    main()
