"""Rendered rollouts: an animated GIF of the point mass doing the task.

The figure is deliberately two coupled panels, because neither alone is enough:

* **Scene** (top) -- the physical 1D workspace.  Targets are circles whose radius *is*
  the acquisition tolerance, numbered in sequence order and styled by what the agent
  currently knows about them (hidden beyond the preview horizon / visible / current /
  already acquired).  The point mass is a circle on the track dragging a short
  speed-coloured comet tail.  The tail is only the last ``trail_steps`` steps and fades
  with age: a full-trial trace on a line collapses into an unreadable smear as soon as
  the agent reverses direction.
* **Timeline** (bottom) -- position against time, drawn with the same blue->red speed
  colormap.  Here the whole trial *can* be shown, because time never doubles back, so
  the two panels answer different questions: "where is it now" and "what did the whole
  reach look like".

Everything else on the frame is there to make the emergent-lookahead story readable at a
glance: the moment a future target enters the horizon is flashed as an event, so it is
obvious whether the trajectory bends before or after the agent could possibly have known
where it was going.

Both training and offline evaluation go through the same path
(``probes.rollout`` -> ``render_frames`` -> ``save_gif``), so a training video and a
test-time video are the same object.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from . import probes
from .config import Config, VideoConfig

SPEED_CMAP = matplotlib.colormaps["coolwarm"]  # blue = slow, red = fast

# Target styling by what the agent knows right now.  Deliberately not one colour per
# target: the blue->red channel is spoken for by speed, and reusing it would be
# ambiguous.
_STYLE = {
    "hidden":   dict(edge="#c2c2c2", lw=0.9, ls=(0, (2, 2)), text="#b5b5b5"),
    "visible":  dict(edge="#2f2f2f", lw=1.5, ls="solid",     text="#2f2f2f"),
    "current":  dict(edge="#e0913a", lw=2.6, ls="solid",     text="#b26a12"),
    "acquired": dict(edge="#69a96b", lw=1.0, ls="solid",     text="#69a96b"),
}
_FACE_ALPHA = {"hidden": 0.0, "visible": 0.07, "current": 0.16, "acquired": 0.10}

_GO_COLOR = "#2f8f3f"
_HOLD_COLOR = "#c08a2a"
_FLASH_STEPS = 14  # how long an event caption stays on screen, in env steps
ASPECT = 0.658  # frame height / width


# --------------------------------------------------------------------- trials
def eval_spec(cfg: Config, n_trials: int) -> dict:
    """A fixed battery of trials to film, biased towards the informative cases.

    Trial 1 gets the full sequence visible from the start, trial 2 the same length with a
    horizon of 1 (each target appears only once the previous one is taken), so the two
    sit side by side in the same video.  Target positions and the go cue are still drawn
    from the task distribution -- only the sequence length and horizon are pinned.
    """
    K = cfg.task.max_targets
    lengths = [min(3, K), min(3, K), min(2, K), 1]
    horizons = [min(3, K), 1, 1, 1]
    n_tg = np.array([lengths[i % len(lengths)] for i in range(n_trials)])
    if cfg.task.fixed_preview is not None:  # respect the ablation, do not misrepresent
        prev = np.full(n_trials, int(cfg.task.fixed_preview))
    else:
        prev = np.array([horizons[i % len(horizons)] for i in range(n_trials)])
    return {"n_targets": n_tg, "preview": np.clip(prev, 1, K)}


def video_config(cfg: Config, vcfg: VideoConfig) -> Config:
    """Plant used for filming.  Noise off by default so the same seed replays the same
    trial at every checkpoint and successive videos are comparable frame for frame."""
    if not vcfg.noiseless:
        return cfg
    return cfg.override({"plant.sigma_sdn": 0.0, "plant.sigma_add": 0.0})


# ------------------------------------------------------------------ per-trial
def _trial_data(traj: dict, cfg: Config, i: int) -> dict:
    """Slice trial ``i`` out of a rollout and derive everything the frame needs."""
    T_avail = traj["x"].shape[0]
    T = int(min(traj["end_step"][i] + 1, T_avail))
    T = max(T, 1)
    K = cfg.task.max_targets
    n = int(traj["n_targets"][i])
    H = int(traj["preview"][i])
    go = int(traj["go_step"][i])

    cur = traj["cur"][:T, i].astype(int)
    slots = np.arange(K)[None, :]
    in_seq = slots < n
    acquired = in_seq & (slots < cur[:, None])
    current = in_seq & (slots == cur[:, None])
    visible = in_seq & (slots >= cur[:, None]) & (slots <= (cur + H - 1)[:, None])

    d = {
        "T": T, "n": n, "H": H, "go": go, "K": K,
        "x": traj["x"][:T, i], "v": traj["v"][:T, i],
        "tg": traj["targets"][:T, i, :],
        "cur": cur, "acquired": acquired, "current": current, "visible": visible,
        "cum_r": np.cumsum(traj["r"][:T, i]) if "r" in traj else np.zeros(T),
        "success": bool(traj["success"][i]),
    }

    # --- events, in the order they should appear as captions
    events: list[tuple[int, str, str]] = []
    if go < T:
        events.append((go, "GO", _GO_COLOR))
    for k in range(n):
        rev = np.flatnonzero(d["visible"][:, k])
        if rev.size and rev[0] > 0:
            events.append((int(rev[0]), f"T{k + 1} enters horizon", "#3d6fb4"))
        acq = np.flatnonzero(d["acquired"][:, k])
        if acq.size:
            events.append((int(acq[0]), f"T{k + 1} acquired ✓", "#69a96b"))
    moved = np.flatnonzero(np.any(np.abs(np.diff(d["tg"], axis=0)) > 1e-9, axis=1))
    for t in moved:
        events.append((int(t) + 1, "target jumped", "#c0392b"))
    d["events"] = sorted(events)

    end = T - 1
    if d["success"]:
        d["outcome"] = ("SUCCESS ✓", _GO_COLOR)
    elif end >= cfg.task.max_steps - 1:
        d["outcome"] = ("TIMEOUT", "#c0392b")
    else:
        d["outcome"] = ("FALSE START", "#c0392b")
    return d


def _param_lines(cfg: Config, vcfg: VideoConfig) -> str:
    p, t, r, a, o = cfg.plant, cfg.task, cfg.reward, cfg.agent, cfg.ppo
    net = f"GRU{a.rnn_size}" if a.recurrent else f"MLP{a.hidden_size}"
    delay = f"{t.delay_min_s:.2f}-{t.delay_max_s:.2f}s" if t.use_delay else "off"
    return "\n".join([
        f"plant   dt {p.dt:.3f}  mass {p.mass:g}  damp {p.damping:g}  u_max {p.u_max:g}  "
        f"tau_act {p.tau_act:g}  sig_sdn {p.sigma_sdn:g}  sens_delay {p.sensory_delay}",
        f"task    tol {t.tolerance:g}  dwell {t.dwell_inter}/{t.dwell_final}  "
        f"max_targets {t.max_targets}  range {t.target_range:g}  min_sep {t.min_separation:g}  "
        f"go-delay {delay}  p_jump {t.p_jump:g}",
        f"reward  hit {r.r_hit:g}  final {r.r_final:g}  w_eff {r.w_effort:g}  "
        f"w_rate {r.w_rate:g}  w_time {r.w_time:g}  shaping {r.shaping}"
        f"({r.shaping_coef:g})   |   ppo  gamma {o.gamma:g}  lam {o.gae_lambda:g}  "
        f"lr {o.learning_rate:g}  {net}",
        f"render  noiseless {vcfg.noiseless}  deterministic {vcfg.deterministic}  "
        f"seed {vcfg.seed}  trail {vcfg.trail_steps} steps  speed_max {vcfg.speed_max:g}",
    ])


class _Renderer:
    """Holds one matplotlib figure and mutates its artists frame by frame.

    Rebuilding the figure per frame is ~10x slower, which matters: this runs inside the
    training loop.
    """

    def __init__(self, cfg: Config, vcfg: VideoConfig, lim: float, title: str):
        self.cfg, self.vcfg, self.lim = cfg, vcfg, lim
        self.title_prefix = title
        w_in = vcfg.width_px / vcfg.dpi
        h_in = w_in * ASPECT
        self.fig = plt.figure(figsize=(w_in, h_in), dpi=vcfg.dpi, facecolor="white")

        scene_rect = (0.055, 0.640, 0.815, 0.265)
        time_rect = (0.055, 0.205, 0.815, 0.335)
        self.ax = self.fig.add_axes(scene_rect)
        self.axt = self.fig.add_axes(time_rect)
        cax = self.fig.add_axes((0.902, 0.205, 0.016, 0.700))

        # Equal aspect by construction: the y half-range is fixed from the axes shape so
        # a circle of radius = tolerance is drawn as an actual circle.
        self.y_half = lim * (scene_rect[3] * h_in) / (scene_rect[2] * w_in)
        self._build_scene()
        self._build_timeline()

        norm = plt.Normalize(0.0, vcfg.speed_max)
        cb = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SPEED_CMAP), cax=cax)
        cb.set_label("speed |v|", fontsize=6.5)
        cb.ax.tick_params(labelsize=6, length=2)
        self.norm = norm

        self.t_title = self.fig.text(0.055, 0.955, "", fontsize=8.5, ha="left",
                                     va="center", color="#222222")
        self.t_clock = self.fig.text(0.918, 0.955, "", fontsize=11, ha="right",
                                     va="center", family="monospace", color="#111111")
        self.t_status = self.fig.text(0.055, 0.583, "", fontsize=7.5, ha="left",
                                      va="center", family="monospace")
        self.fig.text(0.055, 0.118, _param_lines(cfg, vcfg), fontsize=5.4, ha="left",
                      va="top", family="monospace", color="#8a8a8a", linespacing=1.6)

        # Everything that changes within a trial.  The rest of the figure -- axes, ticks,
        # colorbar, the parameter block -- is captured once per trial and blitted back,
        # which is worth ~8x here because this renders inside the training loop.
        self._dynamic = [
            *self.bands, *self.band_labels, self.path, self.now_line, self.now_dot,
            self.trail, *self.rings, *self.dots, *self.labels, self.arrow, self.mass,
            self.t_flash, self.t_title, self.t_clock, self.t_status,
        ]
        self._bg = None

    # ------------------------------------------------------------ static setup
    def _build_scene(self):
        ax, lim, yh = self.ax, self.lim, self.y_half
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-yh, yh)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6.5, length=2, colors="#888888")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.axhline(0.0, color="#dcdcdc", lw=3.0, zorder=1, solid_capstyle="round")
        ax.plot([self.cfg.task.start_x], [0.0], marker="|", ms=9, mew=1.2,
                color="#9a9a9a", zorder=2)
        ax.text(self.cfg.task.start_x, -0.50 * yh, "start", fontsize=5.5, ha="center",
                va="top", color="#9a9a9a")

        self.trail = LineCollection([], cmap=SPEED_CMAP, linewidths=4.5, zorder=3,
                                    capstyle="round")
        ax.add_collection(self.trail)

        tol = self.cfg.task.tolerance
        self.rings, self.dots, self.labels = [], [], []
        for k in range(self.cfg.task.max_targets):
            ring = Circle((0, 0), tol, zorder=4, visible=False)
            ax.add_patch(ring)
            dot, = ax.plot([], [], marker="o", ms=2.0, color="#2f2f2f", zorder=5,
                           visible=False)
            lab = ax.text(0, 0.55 * yh, "", fontsize=8, ha="center", va="center",
                          fontweight="bold", zorder=6, visible=False)
            self.rings.append(ring), self.dots.append(dot), self.labels.append(lab)

        self.mass = Circle((0, 0), max(tol * 0.55, 0.012 * lim), zorder=8,
                           edgecolor="#111111", lw=1.1)
        ax.add_patch(self.mass)
        self.arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=7,
                                     lw=1.2, color="#444444", zorder=7, visible=False)
        ax.add_patch(self.arrow)
        # Event captions sit above the target labels, out of the way of everything that
        # moves, but still inside the panel where the eye already is.
        self.t_flash = ax.text(0.008, 0.97, "", transform=ax.transAxes, fontsize=8,
                               ha="left", va="top", fontweight="bold", zorder=9)

    def _build_timeline(self):
        ax, lim = self.axt, self.lim
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("position", fontsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.tick_params(labelsize=6.5, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        self.bands, self.band_labels = [], []
        for _ in range(self.cfg.task.max_targets):
            band = Rectangle((0, 0), 0, 0, zorder=2, visible=False)
            ax.add_patch(band)
            lab = ax.text(0, 0, "", fontsize=6, ha="left", va="center", zorder=3,
                          visible=False)
            self.bands.append(band), self.band_labels.append(lab)

        self.go_line = ax.axvline(0.0, color=_GO_COLOR, lw=1.0, ls="--", zorder=4)
        self.path = LineCollection([], cmap=SPEED_CMAP, linewidths=1.8, zorder=5)
        ax.add_collection(self.path)
        self.now_line = ax.axvline(0.0, color="#999999", lw=0.8, zorder=6)
        self.now_dot, = ax.plot([], [], marker="o", ms=4.5, mec="#111111", mew=0.9,
                                zorder=7)

    # ----------------------------------------------------------------- drawing
    def new_trial(self, d: dict, label: str):
        """Reset the per-trial artists (axis extent, target count)."""
        self.d, self.trial_label = d, label
        dt = self.cfg.plant.dt
        self.axt.set_xlim(0, max(d["T"] - 1, 1) * dt)
        self.go_line.set_xdata([d["go"] * dt, d["go"] * dt])
        for k in range(d["K"]):
            on = k < d["n"]
            self.rings[k].set_visible(on)
            self.dots[k].set_visible(on)
            self.labels[k].set_visible(on)
            self.bands[k].set_visible(on)
            self.band_labels[k].set_visible(on)
            if on:
                self.labels[k].set_text(str(k + 1))
                self.band_labels[k].set_text(f"T{k + 1}")
        self._capture_background()

    def _capture_background(self):
        """Render the static layer once (tick labels change with the trial's duration)."""
        vis = [a.get_visible() for a in self._dynamic]
        for a in self._dynamic:
            a.set_visible(False)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a, v in zip(self._dynamic, vis):
            a.set_visible(v)

    def draw(self, t: int) -> np.ndarray:
        d, cfg, vcfg = self.d, self.cfg, self.vcfg
        dt, tol = cfg.plant.dt, cfg.task.tolerance
        x, v = d["x"][t], d["v"][t]

        # --- targets, in both panels, styled by what the agent knows at this step
        for k in range(d["n"]):
            if d["acquired"][t, k]:
                state = "acquired"
            elif d["current"][t, k]:
                state = "current"
            elif d["visible"][t, k]:
                state = "visible"
            else:
                state = "hidden"
            s, pos = _STYLE[state], d["tg"][t, k]
            ring = self.rings[k]
            ring.set_center((pos, 0.0))
            ring.set_edgecolor(s["edge"])
            ring.set_facecolor(mcolors.to_rgba(s["edge"], _FACE_ALPHA[state]))
            ring.set_linewidth(s["lw"])
            ring.set_linestyle(s["ls"])
            self.dots[k].set_data([pos], [0.0])
            self.dots[k].set_color(s["text"])
            self.labels[k].set_position((pos, 0.55 * self.y_half))
            self.labels[k].set_color(s["text"])

            band = self.bands[k]
            band.set_bounds(0, pos - tol, self.axt.get_xlim()[1], 2 * tol)
            band.set_facecolor(mcolors.to_rgba(s["edge"], _FACE_ALPHA[state] + 0.05))
            band.set_edgecolor(s["edge"])
            band.set_linewidth(0.7)
            band.set_linestyle(s["ls"])
            self.band_labels[k].set_position((0.004 * self.axt.get_xlim()[1], pos))
            self.band_labels[k].set_color(s["text"])

        # --- point mass + comet tail (last trail_steps only; see module docstring)
        lo = max(0, t - vcfg.trail_steps)
        xs = d["x"][lo:t + 1]
        speed = np.abs(d["v"][lo:t + 1])
        if xs.size >= 2:
            pts = np.stack([xs, np.zeros_like(xs)], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            age = np.linspace(0.0, 1.0, len(segs))
            self.trail.set_segments(segs)
            self.trail.set_array(0.5 * (speed[:-1] + speed[1:]))
            self.trail.set_norm(self.norm)
            self.trail.set_alpha(0.12 + 0.78 * age**1.6)
        else:
            self.trail.set_segments([])

        self.mass.set_center((x, 0.0))
        self.mass.set_facecolor(SPEED_CMAP(self.norm(abs(v))))
        reach = 0.16 * self.lim * np.clip(abs(v) / vcfg.speed_max, 0, 1.4)
        off = np.sign(v) * (self.mass.radius + 0.006 * self.lim)
        y = 0.34 * self.y_half
        tail = np.clip(x + off, -self.lim, self.lim)
        tip = np.clip(x + off + np.sign(v) * reach, -self.lim, self.lim)
        if abs(tip - tail) > 0.012 * self.lim:
            self.arrow.set_positions((tail, y), (tip, y))
            self.arrow.set_visible(True)
        else:
            self.arrow.set_visible(False)

        # --- timeline: the whole trial so far, no fade (time cannot double back)
        tx, tv = d["x"][:t + 1], np.abs(d["v"][:t + 1])
        if tx.size >= 2:
            tt = np.arange(t + 1) * dt
            pts = np.stack([tt, tx], axis=1)
            self.path.set_segments(np.stack([pts[:-1], pts[1:]], axis=1))
            self.path.set_array(0.5 * (tv[:-1] + tv[1:]))
            self.path.set_norm(self.norm)
        else:
            self.path.set_segments([])
        self.now_line.set_xdata([t * dt, t * dt])
        self.now_dot.set_data([t * dt], [x])
        self.now_dot.set_color(SPEED_CMAP(self.norm(abs(v))))

        # --- header, status, event caption
        self.t_title.set_text(f"{self.title_prefix}   ·   {self.trial_label}")
        self.t_clock.set_text(f"t = {t * dt:5.2f} s")

        pre_go = t < d["go"]
        if t >= d["T"] - 1:
            phase, pcol = d["outcome"]
        elif pre_go:
            phase, pcol = "HOLD", _HOLD_COLOR
        else:
            phase, pcol = "GO", _GO_COLOR
        vis = [f"T{k + 1}" for k in range(d["n"]) if d["visible"][t, k]]
        tgt = min(int(d["cur"][t]) + 1, d["n"])
        self.t_status.set_text(
            f"{phase:<12s} horizon H={d['H']}   sees {','.join(vis) or '-':<9s} "
            f"going to T{tgt}/{d['n']}   x={x:+.3f}  v={v:+.3f}   "
            f"return={d['cum_r'][t]:+7.2f}"
        )
        self.t_status.set_color(pcol)

        self.t_flash.set_text("")
        for et, text, col in reversed(d["events"]):
            if et <= t < et + _FLASH_STEPS:
                self.t_flash.set_text(text)
                self.t_flash.set_color(col)
                self.t_flash.set_alpha(1.0 - 0.7 * (t - et) / _FLASH_STEPS)
                break

        canvas = self.fig.canvas
        canvas.restore_region(self._bg)
        for a in self._dynamic:
            if a.get_visible():
                (a.axes or self.fig).draw_artist(a)
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


# ------------------------------------------------------------------ rendering
def render_frames(traj: dict, cfg: Config, vcfg: VideoConfig | None = None,
                  trials: list[int] | None = None, title: str = "",
                  hold_frames: int = 12) -> list[np.ndarray]:
    """Render the given trials into one flat list of RGB frames."""
    vcfg = vcfg or cfg.video
    trials = list(range(traj["x"].shape[1])) if trials is None else list(trials)

    # Only the live portion of each trial counts: the env autoresets in place, so rows
    # past ``end_step`` belong to a different trial and would skew the axis limits.
    alive = traj["alive"][:, trials]
    reach = max(cfg.task.target_range,
                float(np.abs(traj["x"][:, trials][alive]).max(initial=0.0)),
                float(np.abs(traj["targets"][:, trials][alive]).max(initial=0.0)))
    lim = reach + max(3 * cfg.task.tolerance, 0.08)

    data = [_trial_data(traj, cfg, i) for i in trials]
    total = sum(len(range(0, d["T"], vcfg.stride)) for d in data)
    stride = vcfg.stride
    while total > 600 and stride < 8:  # keep the GIF (and peak memory) sane
        stride += 1
        total = sum(len(range(0, d["T"], stride)) for d in data)

    r = _Renderer(cfg, vcfg, lim, title)
    frames: list[np.ndarray] = []
    try:
        for j, (i, d) in enumerate(zip(trials, data)):
            r.new_trial(d, f"trial {j + 1}/{len(trials)}  "
                           f"({d['n']} targets, horizon {d['H']})")
            steps = list(range(0, d["T"], stride))
            if steps[-1] != d["T"] - 1:
                steps.append(d["T"] - 1)
            for t in steps:
                frames.append(r.draw(t))
            frames.extend([frames[-1]] * hold_frames)  # pause on the outcome
    finally:
        r.close()
    return frames


def save_gif(frames: list[np.ndarray], path: str | Path, duration_ms: int = 40,
             colors: int = 128) -> Path:
    """Write frames as an optimised GIF using one shared adaptive palette.

    A per-frame palette makes the colours shimmer between frames and inflates the file;
    quantising every frame against a palette built from samples across the whole clip
    avoids both.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    sample = imgs[:: max(1, len(imgs) // 12)][:12]
    palette = Image.fromarray(np.concatenate([np.asarray(s) for s in sample], axis=0))
    palette = palette.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quant = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    quant[0].save(path, save_all=True, append_images=quant[1:], loop=0,
                  duration=duration_ms, optimize=True, disposal=1)
    return path


def render_gif(traj: dict, cfg: Config, out_path: str | Path,
               vcfg: VideoConfig | None = None, trials: list[int] | None = None,
               title: str = "") -> Path:
    vcfg = vcfg or cfg.video
    frames = render_frames(traj, cfg, vcfg, trials=trials, title=title)
    if vcfg.real_time:
        ms = int(round(1000 * cfg.plant.dt * vcfg.stride))
    else:
        ms = int(round(1000 / max(vcfg.fps, 1)))
    return save_gif(frames, out_path, duration_ms=max(ms, 20))


def record_video(agent, obs_rms, cfg: Config, out_path: str | Path,
                 vcfg: VideoConfig | None = None, title: str = "",
                 n_trials: int | None = None) -> Path:
    """Roll out the pinned evaluation trials and write the GIF.  One call, one file."""
    vcfg = vcfg or cfg.video
    n = n_trials or vcfg.n_trials
    ecfg = video_config(cfg, vcfg)
    traj = probes.rollout(agent, obs_rms, ecfg, eval_spec(ecfg, n), n_trials=n,
                          seed=vcfg.seed, deterministic=vcfg.deterministic)
    name = title or f"{cfg.exp_name}_s{cfg.seed}"
    return render_gif(traj, ecfg, out_path, vcfg=vcfg, title=name)
