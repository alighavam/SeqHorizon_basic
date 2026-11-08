#!/usr/bin/env python
"""Train the v2 pilot grid: which hold design removes pre-go drift?

Every v2 arm shares the same task changes relative to v1:

* random start position in training (x0 ~ U[-0.5, 0.5]) with a visible start marker,
  so the hold is learned relative to the start rather than to x = 0;
* sensory delay of 2 steps (40 ms) on position/velocity and a cue delay of 4 steps
  (80 ms) on the go signal, so reaction time becomes a real quantity.

The arms differ only in how strictly the hold is enforced:

    v2_base   v1 hold rule (abort at 0.15, no catch trials)
    v2_tight  abort at 0.05 (= target tolerance)
    v2_catch  20% catch trials (go cue never comes; hold to be paid r_catch)
    v2_both   abort at 0.05 + 20% catch trials

    python 1D/scripts/run_v2_pilot.py --seeds 1 2 3 --parallel 10 --wandb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V2_COMMON = {
    "task.start_range": 0.5,
    "task.obs_start": True,
    "plant.sensory_delay": 2,
    "plant.cue_delay": 4,
}

ARMS: dict[str, tuple[dict, str]] = {
    "v2_base": ({}, "v1 hold rule"),
    "v2_tight": ({"task.hold_abort_dist": 0.05}, "tight abort (0.05)"),
    "v2_catch": ({"task.p_catch": 0.2}, "20% catch trials"),
    "v2_both": ({"task.hold_abort_dist": 0.05, "task.p_catch": 0.2},
                "tight abort + catch trials"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="v2-pilot")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    jobs = [(a, s) for a in (args.only or ARMS) for s in args.seeds
            if not (runs_dir / f"{a}_s{s}" / "checkpoint.pt").exists()]
    print(f"[v2] {len(jobs)} jobs: {[f'{a}_s{s}' for a, s in jobs]}", flush=True)

    queue, running = list(jobs), []
    while queue or running:
        while queue and len(running) < args.parallel:
            arm, seed = queue.pop(0)
            over = {**V2_COMMON, **ARMS[arm][0], "torch_threads": 1}
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", arm,
                   "--seed", str(seed), "--timesteps", str(args.timesteps),
                   "--runs-dir", str(runs_dir), "--quiet", "--no-video",
                   "--set", *[f"{k}={v}" for k, v in over.items()]]
            if args.wandb:
                cmd += ["--wandb", "--wandb-group", args.wandb_group, "--tags", "v2", arm]
            rd = runs_dir / f"{arm}_s{seed}"
            rd.mkdir(parents=True, exist_ok=True)
            fh = (rd / "stdout.log").open("w")
            running.append((f"{arm}_s{seed}", subprocess.Popen(cmd, stdout=fh, stderr=fh), fh))
            print(f"  -> {arm}_s{seed}", flush=True)
        time.sleep(10)
        for item in list(running):
            name, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                ok = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(f"  <- {name} {ok}", flush=True)
    print("[v2] all done", flush=True)


if __name__ == "__main__":
    main()
