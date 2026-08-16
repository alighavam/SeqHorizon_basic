#!/usr/bin/env python
"""Run the probe battery on a trained agent and write figures + results.json.

    python 1D/scripts/analyze.py --run 1D/runs/baseline_s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach import metrics, plots, probes
from seqreach.ppo import load_agent


def jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()
                if not (isinstance(v, np.ndarray) and v.size > 200)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--noisy", action="store_true",
                    help="keep signal-dependent motor noise on during probes")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--competence-floor", type=float, default=probes.COMPETENCE_FLOOR,
                    help="sequence success rate below which behavioural indices are "
                         "reported but flagged uninterpretable (0 disables the gate)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the noisy robustness check in its own files; it is a different protocol and
    # must not overwrite the primary (noiseless, deterministic) results.
    sfx = "_noisy" if args.noisy else ""

    agent, obs_rms, cfg = load_agent(run_dir)
    pcfg = probes.probe_config(cfg, noiseless=not args.noisy)
    results: dict = {"run": str(run_dir), "name": cfg.exp_name, "seed": cfg.seed,
                     "gamma": cfg.ppo.gamma, "recurrent": cfg.agent.recurrent}
    print(f"[analyze] {run_dir}")

    # -- 0. competence on the training distribution ---------------------------
    # This gates everything below it.  Each behavioural index describes the shape of a
    # movement the agent is making; if it is not making the movement, the index is not
    # measuring the thing it is named after.  See doc/1D_sequential_reach.md §6.5 for the
    # case that forced this -- an arm whose CI reversed sign once it learned the task.
    rand = probes.rollout(agent, obs_rms, pcfg, {}, n_trials=512, seed=123)
    outcome = probes.outcome_breakdown(rand, floor=args.competence_floor)
    results.update(outcome)
    competent = outcome["competent"]

    by_len = "  ".join(f"{k}tgt {v:.2f}" for k, v in
                       sorted(outcome["success_by_n_targets"].items()))
    print(f"  success rate {outcome['success_rate']:.3f}  "
          f"hit fraction {outcome['hit_fraction']:.3f}   [{by_len}]")
    print(f"  failures: false start {outcome['false_start_rate']:.2f}  "
          f"timeout {outcome['timeout_rate']:.2f}  "
          f"wall contact {outcome['wall_contact_rate']:.2f} "
          f"({outcome['mean_wall_steps']:.0f} steps/episode)")
    if not competent:
        print(f"  !! NOT COMPETENT (success {outcome['success_rate']:.2f} < floor "
              f"{args.competence_floor:.2f}).  Indices below are recorded but must not "
              f"be interpreted as measurements of lookahead.")
        if outcome["success_multi_target"] < 0.05 <= outcome["success_rate"]:
            print("     signature: solves single-target trials, fails every multi-target "
                  "one -- a single-reach policy, so there is no second movement to plan.")
    if not args.no_figures:
        plots.plot_example_trials(rand, out_dir / f"fig1_example_trials{sfx}.png")

    # -- 1. co-articulation: fix T1, sweep T2, preview on/off -----------------
    spec, design = probes.preview_probe(pcfg, repeats=args.repeats)
    traj = probes.rollout(agent, obs_rms, pcfg, spec, n_trials=len(design["t1"]), seed=7)
    co = metrics.co_articulation_by_preview(traj, design)
    results["coarticulation"] = {
        "ci": co["preview_on"]["ci"], "ci_se": co["preview_on"]["ci_se"],
        "ci_p": co["preview_on"]["ci_p"], "ci_r": co["preview_on"]["ci_r"],
        "ci_control": co["preview_off"]["ci"], "delta_ci": co["delta_ci"],
        "n_on": co["preview_on"]["n"], "n_off": co["preview_off"]["n"],
        "peak_slope": co["preview_on"]["peak_slope"],
        "peak_p": co["preview_on"]["peak_p"],
        "dur_slope": co["preview_on"]["dur_slope"],
        "dur_p": co["preview_on"]["dur_p"],
    }
    results["ci"] = co["preview_on"]["ci"]
    results["ci_se"] = co["preview_on"]["ci_se"]
    results["ci_interpretable"] = competent
    # The probe fixes T1 and sweeps T2, so every trial should contribute.  A short count
    # means the policy could not even reach T1 on the rest -- which is how the first
    # grid's `no_preview` arm produced a NaN index (0 of 36 preview-on trials) and how
    # two other arms produced indices fit to a single reach direction.
    n_expected = int((design["preview"] == 2).sum())
    results["coarticulation"]["n_expected"] = n_expected
    if co["preview_on"]["n"] < n_expected:
        print(f"  !! probe attrition: T1 reached on {co['preview_on']['n']}/{n_expected} "
              f"preview-on trials -- the index is fit to a biased subset")
    print(f"  CI (T2 visible) = {co['preview_on']['ci']:+.4f} "
          f"± {co['preview_on']['ci_se']:.4f}  p={co['preview_on']['ci_p']:.2e}"
          f"{'' if competent else '   [UNINTERPRETABLE: below competence floor]'}")
    print(f"  CI (T2 hidden)  = {co['preview_off']['ci']:+.4f}   [control]")

    div = metrics.trajectory_divergence(traj, design)
    results["divergence"] = {
        str(k): {"divergence_s": v["divergence_s"],
                 "pre_go_spread": v["pre_go_spread"]} for k, v in div.items()
    }
    print("  divergence: " + ", ".join(
        f"dir{k:+.0f} {v['divergence_s']}s (pre-go spread {v['pre_go_spread']:.4f})"
        for k, v in div.items()))
    if not args.no_figures:
        plots.plot_coarticulation(traj, design, co, out_dir / f"fig2_coarticulation{sfx}.png")

    # -- 2. delay period ------------------------------------------------------
    dspec, ddesign = probes.delay_probe(pcfg)
    dtraj = probes.rollout(agent, obs_rms, pcfg, dspec,
                           n_trials=len(ddesign["delay_steps"]), seed=11)
    rt = metrics.reaction_times(dtraj, ddesign)
    results["reaction_time"] = jsonable(rt)
    if rt.get("by_delay"):
        msg = ", ".join(f"{r['delay_s']:.2f}s->{r['rt_mean']*1000:.0f}ms"
                        for r in rt["by_delay"])
        print(f"  RT vs delay: {msg}")
        if "slope" in rt:
            print(f"  RT slope {rt['slope']*1000:+.1f} ms per s of hold "
                  f"(p={rt['p']:.3f})")
    if not args.no_figures and rt.get("by_delay"):
        plots.plot_reaction_times(rt, out_dir / f"fig3_reaction_time{sfx}.png")

    # -- 3. online replanning -------------------------------------------------
    jspec, jdesign = probes.jump_probe(pcfg)
    jtraj = probes.rollout(agent, obs_rms, pcfg, jspec,
                           n_trials=len(jdesign["jump_kind"]), seed=13)
    jr = metrics.jump_response(jtraj, jdesign)
    results["jump"] = jsonable(jr)
    for r in jr["rows"]:
        print(f"  {r['kind']:<13} sign {r['sign']:+.0f}  latency "
              f"{r['latency_s']}  dev-before-T1 {r['dev_before_t1']:.4f}")
    if not args.no_figures:
        plots.plot_jump(jtraj, jdesign, out_dir / f"fig4_jump{sfx}.png")

    # -- 4. preparatory dynamics ---------------------------------------------
    # Uses the delay-probe rollout: 200 trials with independently varying T1 and T2,
    # which the preview grid (fixed T1) cannot provide.
    if cfg.agent.recurrent:
        w = agent.actor_mean.weight.detach().numpy()
        # Movement parameters are not in the observation, so decoding them from the
        # pre-go state is the part that actually implies preparation.
        kin = metrics.segment_kinematics(dtraj, slot=0)
        extra = {"peak_v": kin["peak_v"], "peak_v__trials": kin["trials"],
                 "duration": kin["duration_s"], "duration__trials": kin["trials"]}
        prep = metrics.preparatory_analysis(dtraj, ddesign, w, extra=extra)
        results["preparatory"] = jsonable(prep)
        print(f"  prep decode R² (cross-validated): "
              f"T1={prep.get('r2_t1', float('nan')):.2f} (pass-through check) "
              f"peak_v={prep.get('r2_peak_v', float('nan')):.2f} "
              f"duration={prep.get('r2_duration', float('nan')):.2f}")
        print(f"  output-null ratio={prep['potent_ratio']:.2f} "
              f"(pct={prep['potent_percentile']:.2f}; <1 means preparation avoids "
              f"the output direction)")
        if not args.no_figures:
            plots.plot_preparatory(dtraj, ddesign, prep,
                                   out_dir / f"fig5_preparatory{sfx}.png")

    (out_dir / f"results{sfx}.json").write_text(json.dumps(jsonable(results), indent=2))
    print(f"[analyze] wrote {out_dir}/results{sfx}.json")


if __name__ == "__main__":
    main()
