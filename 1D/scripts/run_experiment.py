#!/usr/bin/env python
"""Run a frozen agent through a test experiment and save the data.

The agent is loaded from a checkpoint and never updated -- this is testing, not training.

    # what is available
    python 1D/scripts/run_experiment.py --list

    # one experiment on one agent
    python 1D/scripts/run_experiment.py --run 1D/runs/baseline_s1 --exp preview_sweep

    # every experiment on every trained agent
    python 1D/scripts/run_experiment.py --run 1D/runs/*_s1 --exp all

    # override an experiment's own parameters
    python 1D/scripts/run_experiment.py --run 1D/runs/baseline_s1 --exp preview_sweep \
        --param t1=0.6 --param n_t2=41

    # with motor noise and a sampling policy, repeated (trials stop being identical)
    python 1D/scripts/run_experiment.py --run 1D/runs/baseline_s1 --exp preview_sweep \
        --noisy --stochastic --repeats 20 --seed 3

Writes to ``data/`` by default:

    <experiment>_<agent>_<YYYYmmdd-HHMMSS>_timeseries.csv   one row per sample
    <experiment>_<agent>_<YYYYmmdd-HHMMSS>_trials.csv       one row per trial
    <experiment>_<agent>_<YYYYmmdd-HHMMSS>_meta.json        config + protocol + columns

Add ``--save-hidden`` for the GRU state as a pickle (it is not flat, so not a CSV).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import dataset, experiments, probes
from seqreach.ppo import load_agent


def parse_param(s: str):
    """``key=value`` with the value read as a Python literal where possible."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param needs key=value, got '{s}'")
    k, v = s.split("=", 1)
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[],
                    help="run director(ies) holding checkpoint.pt")
    ap.add_argument("--exp", nargs="*", default=["preview_sweep"],
                    help="experiment name(s), or 'all'")
    ap.add_argument("--param", type=parse_param, action="append", default=[],
                    metavar="KEY=VAL", help="override an experiment parameter")
    ap.add_argument("--data-dir", default=None,
                    help="output directory (default: <project>/data)")
    ap.add_argument("--name", default=None,
                    help="agent name in the data (default: the run directory name)")
    ap.add_argument("--seed", type=int, default=7, help="rollout seed")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat every trial N times; only meaningful with --noisy "
                         "or --stochastic, since otherwise repeats are identical")
    ap.add_argument("--noisy", action="store_true",
                    help="keep signal-dependent motor noise on")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of taking its mean")
    ap.add_argument("--save-hidden", action="store_true",
                    help="also pickle the GRU hidden state")
    ap.add_argument("--include-dead", action="store_true",
                    help="keep samples after a trial ends (flagged 'alive'); "
                         "debugging only")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    args = ap.parse_args()

    if args.list:
        print("available experiments:\n")
        for name in sorted(experiments.EXPERIMENTS):
            desc = experiments.EXPERIMENTS[name]["description"].strip()
            first = desc.splitlines()[0] if desc else ""
            print(f"  {name:<18} {first}")
        print(f"\ndefined in {Path(experiments.__file__).relative_to(Path.cwd())}"
              if Path(experiments.__file__).is_relative_to(Path.cwd())
              else f"\ndefined in {experiments.__file__}")
        return

    if not args.run:
        ap.error("--run is required (or use --list)")

    names = (sorted(experiments.EXPERIMENTS) if "all" in args.exp else args.exp)
    params = dict(args.param)

    for run_dir in args.run:
        run_dir = Path(run_dir)
        if not (run_dir / "checkpoint.pt").exists():
            print(f"[experiment] skip {run_dir}: no checkpoint.pt")
            continue
        agent, obs_rms, cfg = load_agent(run_dir)
        net_name = args.name or run_dir.name
        pcfg = probes.probe_config(cfg, noiseless=not args.noisy)

        for exp_name in names:
            design = experiments.build(exp_name, pcfg, **params)
            if args.repeats > 1:
                design = _tile(design, args.repeats)
            traj = probes.rollout(agent, obs_rms, pcfg, design.spec,
                                  n_trials=design.n_trials, seed=args.seed,
                                  deterministic=not args.stochastic)
            paths = dataset.save_experiment(
                traj, design, pcfg, net_name,
                data_dir=args.data_dir, noiseless=not args.noisy,
                deterministic=not args.stochastic, seed=args.seed,
                run_dir=str(run_dir), save_hidden=args.save_hidden,
                include_dead=args.include_dead)
            n_ok = int(traj["success"].sum())
            print(f"[experiment] {net_name:<24} {exp_name:<16} "
                  f"{design.n_trials:>5} trials  {n_ok:>5} completed")
            for key in ("timeseries", "trials", "hidden"):
                if key in paths:
                    print(f"    {key:<11} {paths[key]}")


def _tile(design, repeats: int):
    """Repeat every trial ``repeats`` times, tagging copies with a ``rep`` factor."""
    import numpy as np

    from seqreach.experiments import Design

    spec = {k: np.repeat(v, repeats, axis=0) for k, v in design.spec.items()}
    factors = {k: np.repeat(v, repeats, axis=0) for k, v in design.factors.items()}
    factors["rep"] = np.tile(np.arange(repeats), design.n_trials)
    return Design(spec=spec, factors=factors, name=design.name,
                  description=design.description,
                  params={**design.params, "repeats": repeats})


if __name__ == "__main__":
    main()
