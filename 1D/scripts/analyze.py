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
    # measuring the thing it is named after.  See doc/1D_results.md §2 for the
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
    # Budget guard.  The competence floor says *that* a run failed; this says *why*.
    # A max_steps that truncates the exploratory trajectories learning has to pass
    # through makes runs fail by timeout -- a setup fault, not an ablation effect, and
    # one that still produces significant-looking indices.  A run dominated by
    # timeouts should be re-run with a larger budget, not interpreted.
    failure_rate = 1.0 - outcome["success_rate"]
    budget_limited = bool(not competent and failure_rate > 0
                          and outcome["timeout_rate"] / failure_rate > 0.5)
    results["budget_limited"] = budget_limited
    if not competent:
        print(f"  !! NOT COMPETENT (success {outcome['success_rate']:.2f} < floor "
              f"{args.competence_floor:.2f}).  Indices below are recorded but must not "
              f"be interpreted as measurements of lookahead.")
        if budget_limited:
            print(f"     !! BUDGET-LIMITED: {outcome['timeout_rate'] / failure_rate:.0%} "
                  f"of failures are timeouts (task.max_steps={pcfg.task.max_steps}). "
                  f"Raise the budget and re-run -- this is a setup fault, not a result.")
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
    # Sign-agnostic companion: does T2 change the reach at all?  Reported next to CI
    # because a signed slope is only meaningful when the two reach directions agree.
    td = metrics.t2_dependence_by_preview(traj, design, tol=pcfg.task.tolerance)
    results["coarticulation"]["t2_spread_rms"] = td["preview_on"]["spread_rms"]
    results["coarticulation"]["t2_spread_peak"] = td["preview_on"]["spread_peak"]
    results["coarticulation"]["t2_spread_rms_tol"] = td["preview_on"]["spread_rms_tol"]
    results["coarticulation"]["t2_spread_control"] = td["preview_off"]["spread_rms"]
    results["coarticulation"]["t2_v_spread_rms"] = td["preview_on"]["v_spread_rms"]
    results["coarticulation"]["t2_v_spread_peak"] = td["preview_on"]["v_spread_peak"]
    results["coarticulation"]["t2_v_spread_control"] = td["preview_off"]["v_spread_rms"]
    results["coarticulation"]["by_group"] = co["preview_on"]["by_group"]
    results["coarticulation"]["ci_balanced"] = co["preview_on"]["ci_balanced"]
    results["coarticulation"]["ci_balanced_se"] = co["preview_on"]["ci_balanced_se"]
    results["coarticulation"]["n_balanced"] = co["preview_on"]["n_balanced"]
    results["coarticulation"]["dir_agree"] = co["preview_on"]["dir_agree"]
    results["ci"] = co["preview_on"]["ci"]
    results["ci_se"] = co["preview_on"]["ci_se"]
    results["ci_interpretable"] = competent
    # The probe fixes T1 and sweeps T2, so every trial should contribute.  A short count
    # means the policy could not even reach T1 on the rest, so the index is fit to a
    # biased subset -- possibly a single reach direction, or nothing at all (NaN).
    n_expected = int((design["preview"] == 2).sum())
    results["coarticulation"]["n_expected"] = n_expected
    if co["preview_on"]["n"] < n_expected:
        print(f"  !! probe attrition: T1 reached on {co['preview_on']['n']}/{n_expected} "
              f"preview-on trials -- the index is fit to a biased subset")
    print(f"  CI (T2 visible) = {co['preview_on']['ci']:+.4f} "
          f"± {co['preview_on']['ci_se']:.4f}  p={co['preview_on']['ci_p']:.2e}"
          f"{'' if competent else '   [UNINTERPRETABLE: below competence floor]'}")
    print(f"  CI (T2 hidden)  = {co['preview_off']['ci']:+.4f}   [control]")
    print(f"  T2 influence on the reach to T1: rms {td['preview_on']['spread_rms']:.4f} "
          f"({td['preview_on']['spread_rms_tol']:.2f} tol), peak "
          f"{td['preview_on']['spread_peak']:.4f}  [control {td['preview_off']['spread_rms']:.6f}]")
    print(f"  T2 influence on velocity: rms {td['preview_on']['v_spread_rms']:.4f}, "
          f"peak {td['preview_on']['v_spread_peak']:.4f}  "
          f"[control {td['preview_off']['v_spread_rms']:.6f}]")
    print(f"  per-direction CI: " + ", ".join(f"{g:+.0f}: {v:+.4f}"
          for g, v in sorted(co["preview_on"]["by_group"].items())))
    print(f"  CI (balanced window) = {co['preview_on']['ci_balanced']:+.4f} "
          f"± {co['preview_on']['ci_balanced_se']:.4f}  on {co['preview_on']['n_balanced']} trials")
    if not co["preview_on"]["dir_agree"]:
        print("  !! the two reach directions disagree on the sign of CI -- the pooled "
              "slope is not a signed effect")

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
