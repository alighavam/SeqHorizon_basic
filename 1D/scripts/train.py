#!/usr/bin/env python
"""Train a PPO agent on SeqReach1D.

    python 1D/scripts/train.py --name baseline
    python 1D/scripts/train.py --name gamma097_sparse --set ppo.gamma=0.97 reward.shaping=none
    python 1D/scripts/train.py --config 1D/runs/baseline_s1/config.yaml --set task.dwell_inter=25

With Weights & Biases tracking (metrics, losses, and rollout GIFs during training):

    python 1D/scripts/train.py --name baseline --wandb --wandb-group pilot
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach.config import Config
from seqreach.ppo import train


def parse_value(s: str):
    """Parse a CLI override value as a Python literal, falling back to str."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s

def build_config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: parse_value(v) for k, v in overrides.items()}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.timesteps is not None:
        overrides["ppo.total_timesteps"] = args.timesteps

    # CLI flags win over --set, which wins over the YAML.
    if args.wandb:
        overrides["wandb.enabled"] = True
    for flag, key in (("wandb_project", "wandb.project"),
                      ("wandb_entity", "wandb.entity"),
                      ("wandb_group", "wandb.group"),
                      ("wandb_mode", "wandb.mode"),
                      ("video_every", "video.every_iters")):
        val = getattr(args, flag)
        if val is not None:
            overrides[key] = val
    if args.tags:
        overrides["wandb.tags"] = tuple(args.tags)
    if args.no_video:
        overrides["video.enabled"] = False

    cfg = cfg.override(overrides)
    cfg.exp_name = args.name
    return cfg

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="baseline", help="run name (directory under runs/)")
    ap.add_argument("--config", default=None, help="YAML config to start from")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="dotted config overrides, e.g. ppo.gamma=0.9")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")

    w = ap.add_argument_group("weights & biases")
    w.add_argument("--wandb", action="store_true", help="track this run on W&B")
    w.add_argument("--wandb-project", default=None)
    w.add_argument("--wandb-entity", default=None)
    w.add_argument("--wandb-group", default=None,
                   help="groups related runs, e.g. an ablation campaign")
    w.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    w.add_argument("--tags", nargs="*", default=None)

    v = ap.add_argument_group("video")
    v.add_argument("--video-every", type=int, default=None,
                   help="render a rollout GIF every N PPO iterations")
    v.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args)
    run_dir = Path(args.runs_dir) / cfg.run_name
    print(f"[train] {cfg.exp_name} -> {run_dir}")
    print(f"[train] recurrent={cfg.agent.recurrent} gamma={cfg.ppo.gamma} "
          f"steps={cfg.ppo.total_timesteps:,}")
    train(cfg, run_dir, log_every=args.log_every, quiet=args.quiet)
    print(f"[train] done -> {run_dir}/checkpoint.pt")

if __name__ == "__main__":
    main()
