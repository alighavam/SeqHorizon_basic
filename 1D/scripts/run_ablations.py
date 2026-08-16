#!/usr/bin/env python
"""Train and analyse the ablation grid, then summarise co-articulation across it.

Each entry removes exactly one candidate source of lookahead.  The point is not to
show the effect once, but to find which couplings it actually depends on.

    python 1D/scripts/run_ablations.py --timesteps 2000000 --parallel 6
    python 1D/scripts/run_ablations.py --only no_preview myopic --analyze-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import plots

ROOT = Path(__file__).resolve().parents[1]

# name -> (dotted overrides, what it tests)
ABLATIONS: dict[str, tuple[dict, str]] = {
    "baseline": ({}, "full task"),
    # `no_preview` (task.fixed_preview=1) was removed on 2026-08-16.  It cannot be
    # probed: trained with only one target ever visible, the agent is out of
    # distribution the moment a second one appears and reached T1 on 0 of 36 preview-on
    # trials.  The only condition it *can* be probed in is preview=1, where CI is
    # exactly zero by construction for any agent whatsoever -- with T2 absent from the
    # observation, trajectories are bit-identical across T2.  That measurement already
    # exists as the baseline's own within-agent preview-off control (`ci_control`), which
    # is strictly stronger because it holds weights and competence fixed.  See
    # doc/1D_sequential_reach.md §6.5 (C6).  `task.fixed_preview` remains available as a
    # config knob; it just is not an ablation arm.
    "myopic": (
        {"ppo.gamma": 0.90},
        "C7: discount horizon (~10 steps) shorter than a reach -- removes the incentive",
    ),
    # Matched sparse-reward gamma series.  PBRS contributes (gamma-1)*Phi per step, and
    # since Phi <= 0 that is a *positive* payment for staying far from the target:
    # negligible at gamma=0.995, worth ~+20/episode at gamma=0.90.  It stays
    # policy-invariant, but it is a learning trap that would confound a gamma sweep.
    # Holding shaping off makes the three points below differ only in gamma; the
    # gamma=0.995 arm of this series is `no_shaping`.
    "myopic_sparse": (
        {"ppo.gamma": 0.90, "reward.shaping": "none"},
        "C7 (matched): gamma=0.90, sparse reward",
    ),
    "gamma097_sparse": (
        {"ppo.gamma": 0.97, "reward.shaping": "none"},
        "C7 (matched): gamma=0.97 -- one reach fits the horizon, the next does not",
    ),
    # The dwell must be paid for in time, or this arm tests two things at once.
    # dwell_inter=25 adds 24 extra steps of *mandatory holding* at each intermediate
    # target -- 48 steps on a three-target trial -- out of a 200-step budget that a
    # near-time-optimal controller already needs 164 of (median; p90 186).  Slack at p90
    # collapses from baseline's 36% to 7%, and a learned policy is never time-optimal:
    # measured 0.00 success on three-target trials while the same run reaches 0.94 on
    # two-target ones.  Giving back exactly the 48 steps the manipulation costs restores
    # 26% slack, so the arm isolates separability instead of confounding it with a
    # 24% cut to the movement-time budget.
    "stop_at_targets": (
        {"task.dwell_inter": 25, "task.max_steps": 250},
        "C1: must come to rest at each target -- makes the problem separable "
        "(time budget compensated for the added dwell)",
    ),
    "no_effort": (
        {"reward.w_effort": 0.0, "reward.w_rate": 0.0},
        "C2: no effort cost -- arrival velocity stops being worth optimising",
    ),
    "no_shaping": (
        {"reward.shaping": "none"},
        "robustness: sparse reward only, no potential-based shaping",
    ),
    "with_jumps": (
        {"task.p_jump": 0.25},
        "trains on target jumps -- for the online-replanning comparison",
    ),
    "mlp": (
        {"agent.recurrent": False},
        "architecture: memoryless policy instead of a GRU",
    ),
}


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--wandb", action="store_true",
                    help="track every arm on W&B, grouped so the grid reads as one row")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default=None,
                    help="default: ablations-<timesteps>")
    ap.add_argument("--video-every", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    names = args.only or list(ABLATIONS)
    runs_dir = Path(args.runs_dir)
    py = sys.executable

    # Flags every arm inherits, so one campaign is one W&B group.
    common = []
    if args.wandb:
        group = args.wandb_group or f"ablations-{args.timesteps // 1000}k"
        common += ["--wandb", "--wandb-group", group, "--tags", "ablation"]
        if args.wandb_project:
            common += ["--wandb-project", args.wandb_project]
    if args.no_video:
        common.append("--no-video")
    elif args.video_every is not None:
        common += ["--video-every", str(args.video_every)]

    # ---------------------------------------------------------------- train
    if not args.analyze_only:
        pending = [n for n in names
                   if not (runs_dir / f"{n}_s{args.seed}" / "checkpoint.pt").exists()]
        print(f"[ablations] training {len(pending)}: {pending}")
        queue, running = list(pending), []
        while queue or running:
            while queue and len(running) < args.parallel:
                name = queue.pop(0)
                over, _ = ABLATIONS[name]
                cmd = [py, str(ROOT / "scripts" / "train.py"), "--name", name,
                       "--timesteps", str(args.timesteps), "--seed", str(args.seed),
                       "--runs-dir", str(runs_dir), "--quiet", *common,
                       "--set", f"torch_threads={args.threads}",
                       *[f"{k}={v}" for k, v in over.items()]]
                log = runs_dir / f"{name}_s{args.seed}"
                log.mkdir(parents=True, exist_ok=True)
                fh = (log / "stdout.log").open("w")
                running.append((name, subprocess.Popen(cmd, stdout=fh, stderr=fh), fh))
                print(f"  -> launched {name}")
            time.sleep(5)
            for item in list(running):
                name, proc, fh = item
                if proc.poll() is not None:
                    fh.close()
                    running.remove(item)
                    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                    print(f"  <- {name} {status}")

    # -------------------------------------------------------------- analyse
    rows = []
    for name in names:
        rd = runs_dir / f"{name}_s{args.seed}"
        if not (rd / "checkpoint.pt").exists():
            print(f"[ablations] skip {name}: no checkpoint")
            continue
        print(f"[ablations] analysing {name}")
        r = subprocess.run([py, str(ROOT / "scripts" / "analyze.py"), "--run", str(rd)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            continue
        res = json.loads((rd / "analysis" / "results.json").read_text())
        co = res.get("coarticulation", {})
        rows.append({
            "name": name,
            "tests": ABLATIONS[name][1],
            "ci": co.get("ci"),
            "ci_se": co.get("ci_se"),
            "ci_p": co.get("ci_p"),
            "ci_control": co.get("ci_control"),
            "success_rate": res.get("success_rate"),
            "hit_fraction": res.get("hit_fraction"),
            # Carried through so the summary can never present a CI without the
            # competence that licenses reading it.
            "competent": res.get("competent"),
            "competence_floor": res.get("competence_floor"),
            "success_multi_target": res.get("success_multi_target"),
            "wall_contact_rate": res.get("wall_contact_rate"),
            "n_on": co.get("n_on"),
            "n_expected": co.get("n_expected"),
        })

    out = runs_dir / "ablation_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    plots.plot_ablations(rows, runs_dir / "fig6_ablations.png")
    plots.plot_training({r["name"]: runs_dir / f"{r['name']}_s{args.seed}" for r in rows},
                        runs_dir / "fig7_training_curves.png")

    # Competent arms first: the summary should read as "here is what can be concluded",
    # with the rest below the line as diagnostics rather than results.
    ok = [r for r in rows if r.get("competent")]
    bad = [r for r in rows if not r.get("competent")]

    hdr = (f"{'condition':<17}{'CI':>10}{'±se':>9}{'control':>10}{'succ':>7}"
           f"{'multi':>7}{'wall':>6}  tests")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in ok:
        print(f"{r['name']:<17}{fmt(r['ci']):>10}{fmt(r['ci_se']):>9}"
              f"{fmt(r['ci_control']):>10}{fmt(r['success_rate']):>7}"
              f"{fmt(r['success_multi_target']):>7}{fmt(r['wall_contact_rate']):>6}"
              f"  {r['tests']}")
    if bad:
        print("-" * len(hdr))
        print("BELOW COMPETENCE FLOOR -- diagnostics only, do not read the CI column:")
        for r in bad:
            note = ""
            if r.get("n_on") is not None and r.get("n_expected"):
                if r["n_on"] < r["n_expected"]:
                    note = f"  [probe attrition {r['n_on']}/{r['n_expected']}]"
            print(f"{r['name']:<17}{fmt(r['ci']):>10}{fmt(r['ci_se']):>9}"
                  f"{fmt(r['ci_control']):>10}{fmt(r['success_rate']):>7}"
                  f"{fmt(r['success_multi_target']):>7}{fmt(r['wall_contact_rate']):>6}"
                  f"  {r['tests']}{note}")
        print("\n'multi' is success on multi-target trials only.  An arm near 1.00 "
              "overall\nsuccess on single targets and 0.00 here has learned one reach, "
              "not the task.")
    print(f"\n[ablations] wrote {out} and fig6_ablations.png")


if __name__ == "__main__":
    main()
