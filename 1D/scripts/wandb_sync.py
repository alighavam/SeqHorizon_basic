#!/usr/bin/env python
"""Push already-finished runs in ``runs/`` to Weights & Biases.

Everything a live ``--wandb`` training run would have produced is reconstructed from
what is on disk: the training curves are replayed out of ``train_log.jsonl`` at their
original ``global_step``, the analysis figures and ``results.json`` are attached, a
rollout GIF is rendered from ``checkpoint.pt``, and the checkpoint itself is uploaded as
a versioned model artifact.

    python 1D/scripts/wandb_sync.py --all --group backfill
    python 1D/scripts/wandb_sync.py --runs 1D/runs/baseline_s1 --project seqhorizon-1d
    python 1D/scripts/wandb_sync.py --all --no-video          # metrics + artifacts only

Runs are named ``<exp_name>_s<seed>``, matching their directory.  Re-running this
creates a *new* W&B run each time unless you pass ``--skip-existing``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import video
from seqreach.config import Config
from seqreach.ppo import load_agent
from seqreach.tracking import Tracker, read_train_log

ROOT = Path(__file__).resolve().parents[1]


def existing_run_names(project: str, entity: str | None) -> set[str]:
    try:
        import wandb

        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        return {r.name for r in api.runs(path)}
    except Exception as exc:
        print(f"[sync] could not list existing runs ({exc!r}); syncing everything")
        return set()


def sync_one(run_dir: Path, args) -> bool:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"[sync] skip {run_dir.name}: no config.yaml")
        return False

    cfg = Config.load(cfg_path)
    cfg = cfg.override({
        "wandb.enabled": True,
        "wandb.project": args.project,
        "wandb.entity": args.entity,
        "wandb.group": args.group,
        "wandb.mode": args.mode,
        "wandb.tags": tuple(args.tags or ()) + ("backfill",),
    })
    tracker = Tracker(cfg, run_dir, job_type="backfill",
                      extra_config={"source_run_dir": str(run_dir)})
    if not tracker.active:
        return False

    rows = read_train_log(run_dir)
    for rec in rows:
        tracker.log({f"charts/{k}": v for k, v in rec.items()
                     if k not in ("iteration", "global_step")}
                    | {"iteration": rec["iteration"]},
                    step=rec["global_step"])
    last_step = rows[-1]["global_step"] if rows else None
    print(f"[sync] {run_dir.name}: replayed {len(rows)} iterations")

    # --- analysis outputs, if analyze.py has been run
    adir = run_dir / "analysis"
    res_path = adir / "results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        flat = {f"analysis/{k}": v for k, v in res.items()
                if isinstance(v, (int, float, bool, str))}
        co = res.get("coarticulation", {})
        flat.update({f"analysis/coarticulation.{k}": v for k, v in co.items()
                     if isinstance(v, (int, float))})
        tracker.summary(flat)
        tracker.log_images({f"analysis/{p.stem}": p for p in sorted(adir.glob("*.png"))},
                           step=last_step)
        tracker.save_files([res_path])
        print(f"[sync] {run_dir.name}: attached analysis "
              f"(CI={res.get('ci')}, success={res.get('success_rate')})")

    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists() and not args.no_video:
        try:
            agent, obs_rms, ccfg = load_agent(run_dir)
            out = run_dir / "videos" / "final.gif"
            video.record_video(agent, obs_rms, ccfg, out, vcfg=cfg.video,
                               title=f"{cfg.run_name}  ·  final"
                                     + (f"  ·  step {last_step:,}" if last_step else ""))
            tracker.log_video(out, step=last_step, key="rollout",
                              caption="final policy")
            print(f"[sync] {run_dir.name}: logged video {out}")
        except Exception as exc:
            print(f"[sync] {run_dir.name}: video failed ({exc!r})")

    if ckpt.exists() and not args.no_artifact:
        tracker.save_checkpoint(ckpt, step=last_step)

    tracker.finish()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run directories to sync")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--all", action="store_true", help="sync every run under --runs-dir")
    ap.add_argument("--project", default="seqhorizon-1d")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--mode", default="online", choices=["online", "offline"])
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose name already exists in the project")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-artifact", action="store_true")
    args = ap.parse_args()

    if args.all or not args.runs:
        dirs = sorted(p.parent for p in Path(args.runs_dir).glob("*/config.yaml"))
    else:
        dirs = [Path(r) for r in args.runs]
    if args.skip_existing and args.mode == "online":
        have = existing_run_names(args.project, args.entity)
        dirs = [d for d in dirs if d.name not in have]

    print(f"[sync] {len(dirs)} run(s) -> {args.project}: {[d.name for d in dirs]}")
    ok = sum(sync_one(d, args) for d in dirs)
    print(f"[sync] done: {ok}/{len(dirs)} synced")


if __name__ == "__main__":
    main()
