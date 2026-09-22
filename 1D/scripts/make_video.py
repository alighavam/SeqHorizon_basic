#!/usr/bin/env python
"""Render a GIF of a trained agent doing the task.

    python 1D/scripts/make_video.py --run 1D/runs/baseline_s1
    python 1D/scripts/make_video.py --run 1D/runs/baseline_s1 --probe jump
    python 1D/scripts/make_video.py --run 1D/runs/baseline_s1 --trials 6 --noisy

``--probe jump`` films the online-replanning protocol (a target is displaced
mid-movement); ``--probe delay`` films matched trials with different hold durations.
Both are the same protocols ``analyze.py`` measures, so the video and the numbers are
describing the same rollouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import probes, video
from seqreach.ppo import load_agent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory holding checkpoint.pt")
    ap.add_argument("--out", default=None, help="output .gif (default <run>/videos/)")
    ap.add_argument("--probe", default="task", choices=["task", "jump", "delay"])
    ap.add_argument("--trials", type=int, default=None,
                    help="how many trials to film (default: config for --probe task, "
                         "the whole protocol otherwise)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--noisy", action="store_true",
                    help="keep signal-dependent motor noise on")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of taking its mean")
    ap.add_argument("--stride", type=int, default=None, help="render every s-th env step")
    ap.add_argument("--trail", type=int, default=None, help="comet-tail length in steps")
    ap.add_argument("--width", type=int, default=None, help="frame width in pixels")
    args = ap.parse_args()

    run_dir = Path(args.run)
    agent, obs_rms, cfg = load_agent(run_dir)

    vcfg = cfg.video
    for attr, val in (("seed", args.seed), ("stride", args.stride),
                      ("trail_steps", args.trail), ("width_px", args.width)):
        if val is not None:
            setattr(vcfg, attr, val)
    vcfg.noiseless = not args.noisy
    vcfg.deterministic = not args.stochastic

    ecfg = video.video_config(cfg, vcfg)
    if args.probe == "jump":
        # All six matched conditions: no jump / current-target jump / future-target jump,
        # each in both directions.  Watching them back to back is the whole point.
        spec, _ = probes.jump_probe(ecfg)
        n_all = len(spec["jump_kind"])
    elif args.probe == "delay":
        spec, design = probes.delay_probe(ecfg, n_per=1)
        n_all = len(design["delay_steps"])
    else:
        n_all = args.trials or vcfg.n_trials
        spec = video.eval_spec(ecfg, n_all)

    traj = probes.rollout(agent, obs_rms, ecfg, spec, n_trials=n_all, seed=vcfg.seed,
                          deterministic=vcfg.deterministic)
    trials = list(range(min(n_all, args.trials or n_all)))

    out = Path(args.out) if args.out else run_dir / "videos" / f"{args.probe}.gif"
    title = f"{cfg.run_name}  ·  {args.probe}" + ("  ·  noisy" if args.noisy else "")
    path = video.render_gif(traj, ecfg, out, vcfg=vcfg, trials=trials, title=title)
    print(f"[video] wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
