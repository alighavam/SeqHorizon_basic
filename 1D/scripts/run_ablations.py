#!/usr/bin/env python
"""Train and analyse the ablation grid, then summarise co-articulation across it.

Each entry changes exactly one thing relative to the baseline.  The point is not to
show the effect once, but to find which couplings it actually depends on.

    python 1D/scripts/run_ablations.py --timesteps 3000000 --parallel 6 --threads 1 --no-video --wandb
    python 1D/scripts/run_ablations.py --analyze-only --seed 1
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
    # C6 (information) has no arm.  An agent trained with only one target visible is out
    # of distribution the moment a second one appears, so it cannot be probed where it
    # matters.  The baseline's own preview-off control (`ci_control`) is the within-agent
    # version, and strictly stronger.  See doc/1D_task_and_design.md §6.
    #
    # C7 runs with shaping off.  PBRS contributes (gamma-1)*Phi per step, a positive
    # payment for staying far from the target that grows as gamma falls; holding shaping
    # off makes gamma the only difference from `no_shaping` (the gamma=0.995 point).
    # gamma=0.90 is not an arm: it shrinks the whole learnable signal, not just the
    # lookahead incentive, and removes the task (design doc §5.2).
    "gamma097_sparse": (
        {"ppo.gamma": 0.97, "reward.shaping": "none"},
        "C7: discount horizon -- one reach fits the horizon, the next does not",
    ),
    # The dwell must be paid for in time, or this arm tests two things at once.  Forcing
    # rest costs braking and re-acceleration as well as the dwell itself: the arm needs
    # ~220 steps converged and ~366 mid-training; a budget sweep found it competent
    # only at 700.
    "stop_at_targets": (
        {"task.dwell_inter": 25, "task.max_steps": 700},
        "C1: must come to rest at each target -- makes the problem separable",
    ),
    "no_effort": (
        {"reward.w_effort": 0.0, "reward.w_rate": 0.0},
        "C2: no effort cost -- arrival velocity stops being worth optimising",
    ),
    "no_shaping": (
        {"reward.shaping": "none"},
        "robustness: sparse reward only, no potential-based shaping",
    ),
    # Naive (non-policy-invariant) pull toward the current target, as a dose series.
    # At |x - T| = 0.5 the per-step cost is coef/100: coef 2 equals the time cost, and
    # at 60 the term exceeds the entire task return.
    **{f"greedy{c}": (
        {"reward.shaping": "greedy", "reward.shaping_coef": float(c)},
        f"confound: naive greedy shaping, coef {c}",
    ) for c in (2, 6, 20, 60)},
    "mlp": (
        {"agent.recurrent": False},
        "architecture: memoryless policy instead of a GRU",
    ),
}


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
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
            "ci_r": co.get("ci_r"),
            "ci_control": co.get("ci_control"),
            # A signed slope means nothing unless the relation is linear and the two
            # reach directions agree; these carry that check into the summary table.
            "ci_balanced": co.get("ci_balanced"),
            "dir_agree": co.get("dir_agree"),
            "by_group": co.get("by_group"),
            "t2_spread_rms": co.get("t2_spread_rms"),
            "t2_v_spread_rms": co.get("t2_v_spread_rms"),
            "budget_limited": res.get("budget_limited"),
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

    hdr = (f"{'condition':<17}{'CI':>10}{'±se':>9}{'r':>7}{'bal':>9}{'dir':>5}"
           f"{'|T2|v':>8}{'succ':>7}{'multi':>7}  tests")
    print("\n" + hdr)
    print("-" * len(hdr))
    def row(r):
        return (f"{r['name']:<17}{fmt(r['ci']):>10}{fmt(r['ci_se']):>9}"
                f"{fmt(r.get('ci_r')):>7}{fmt(r.get('ci_balanced')):>9}"
                f"{('ok' if r.get('dir_agree') else 'XX'):>5}"
                f"{fmt(r.get('t2_v_spread_rms')):>8}{fmt(r['success_rate']):>7}"
                f"{fmt(r['success_multi_target']):>7}  {r['tests']}")
    for r in ok:
        print(row(r))
    if bad:
        print("-" * len(hdr))
        print("BELOW COMPETENCE FLOOR -- diagnostics only, do not read the CI column:")
        for r in bad:
            note = ""
            if r.get("n_on") is not None and r.get("n_expected"):
                if r["n_on"] < r["n_expected"]:
                    note = f"  [probe attrition {r['n_on']}/{r['n_expected']}]"
            if r.get("budget_limited"):
                note += "  [BUDGET-LIMITED: failures are timeouts]"
            print(row(r) + note)
        print("\n'r' is the CI regression's correlation, 'bal' the refit on a d_next window\n"
              "symmetric about zero, 'dir' whether the two reach directions agree on the\n"
              "sign, '|T2|v' the sign-agnostic velocity spread induced by T2.  A CI with\n"
              "dir=XX or |bal - CI| large is not a signed effect.\n")
    print("'multi' is success on multi-target trials only.  An arm near 1.00 "
              "overall\nsuccess on single targets and 0.00 here has learned one reach, "
              "not the task.")
    print(f"\n[ablations] wrote {out} and fig6_ablations.png")


if __name__ == "__main__":
    main()
