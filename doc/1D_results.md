# 1D sequential reaching — results

Paper-facing record of every measurement from the 1D stage. Design, task definition, and
the definition of every index live in [`1D_task_and_design.md`](1D_task_and_design.md);
this file assumes them.

**Last updated:** 2026-09-22.

## How to read this file

Each result carries its **n (seeds)**, the **competence** of the runs it is fit to, and the
checks that license reading it. Three rules are applied throughout:

1. **An index from an agent below the competence floor (0.90) is not a measurement.** Such
   numbers appear only in the "diagnostics" subsections, never in a claim.
2. **A signed CI requires `dir_agree = true`** (the two reach directions agree in sign) and
   `ci` ≈ `ci_balanced`. Otherwise only the sign-agnostic `t2_v_spread` is reportable.
3. **One seed is not a result.** §2 is the reason.

Status tags: **[established]** ≥8 seeds, checks pass. **[provisional]** 1–3 seeds.
**[pending]** trained, not yet analysed.

---

## 1. Summary

| finding | status | n | headline number |
| --- | --- | --- | --- |
| Lookahead emerges in the baseline | **established** | 8 | CI **+0.514 ± 0.139** (sd), 8/8 competent |
| It is caused by T2 information | **established** | 8 | control CI exactly `0.0000`; `t2_v_spread` control ~1e-16 |
| The time budget was the dominant confound | **established** | 3+3 | at `max_steps=200`, 1/3 seeds learn; at 400, 3/3 |
| Shaping form does not affect it | **established** | 8×3 | pbrs +0.514, sparse +0.425, greedy +0.486; all p > 0.47 |
| Greedy dose-response | **pending** | 8×3 | `coef` 6 / 20 / 60 trained; not yet written up |
| C7 (γ) abolishes *signed* co-articulation | **provisional** | 1 | CI +0.048 (p 0.57) at 1.000 success — but see §7 |
| C1 (rest at each target) reverses CI | **provisional** | 1 | CI −0.233 at 1.000 success |
| C2 (no effort cost) leaves CI intact | **provisional** | 1 of 4 | +0.409 on the one seed that learned |
| CI's sign is not always interpretable | **established** | — | §10 |

---

## 2. The time budget — the most consequential finding

A methodological result, and the reason the first campaign (at `task.max_steps = 200`) was
discarded.

**At 200 steps the baseline did not reproduce.** Three seeds of an identical configuration
reached success 1.000 / ~0.30 / 0.297. The outcome is bimodal — a seed either breaks
through or never does — and on a failing seed the CI does not merely weaken, it changes
sign (s1 +0.406, s3 −0.169). At `max_steps = 400` all three seeds reach 1.00.

**Cause.** Every failing run failed by timeout. The cap never bound for a converged policy
(~110 of 200 steps); it truncated the longer exploratory trajectories learning must pass
through. `stop_at_targets` shows it directly: mean episode length peaks at ~335 steps
mid-training (958k steps, success 0.47) and converges to ~207 (success 0.95). **The cap
must clear the mid-training distribution, not the converged one.**

**Fix.** The default is now 600 (700 for `stop_at_targets`), and `analyze.py` flags any run
whose failures are >50% timeouts as `BUDGET-LIMITED` instead of reporting an index.

**The budget is not monotone.** A single-seed sweep on `stop_at_targets` gave success
0.29 / 0.23 / 0.48 / 0.87 / 0.73 / 1.00 at caps 250 / 300 / 400 / 500 / 600 / 700, and
training the 500-step arm for 6M steps made it worse (0.77). A larger cap stops truncating
exploration but also lets doomed episodes run longer, so fewer episodes are seen per unit
of experience.

---

## 3. Lookahead in the baseline **[established, n=8]**

`max_steps=600`, 3M steps, `shaping=pbrs`, seeds 1–8. **8/8 competent** (success
0.994–1.000, multi-target 0.992–1.000).

| seed | success | multi | CI |
| --- | --- | --- | --- |
| s1 | 0.994 | 0.992 | +0.6220 |
| s2 | 1.000 | 1.000 | +0.6062 |
| s3 | 0.994 | 0.992 | +0.2378 |
| s4 | 0.996 | 0.995 | +0.6135 |
| s5 | 0.998 | 0.997 | +0.4723 |
| s6 | 1.000 | 1.000 | +0.4370 |
| s7 | 1.000 | 1.000 | +0.4719 |
| s8 | 1.000 | 1.000 | +0.6513 |

**CI = +0.514, sd 0.139, se 0.049 (n = 8).** Every seed positive.

Per-seed checks on s1: `ci_r` = +0.980, `ci_balanced` = +0.662 ± 0.024 (30 trials) against
`ci` = +0.622, per-direction slopes +0.713 / +0.526 (`dir_agree = true`). The relation is
linear, symmetric-window-stable, and direction-consistent.

Report the effect as **a distribution across seeds, not a point estimate**: the seed sd
(0.139) is ~6× the within-seed se of a typical single run (~0.02–0.07).

### 3.1 The within-agent control

Re-probing the same weights with T2 hidden (`H = 1`) gives **CI = 0.0000 exactly** in every
run, and `t2_v_spread` = ~1e-16 (floating-point zero). With T2 absent from the observation
the trajectories are bit-identical across all T2 values. This is a hard null, and it
certifies that the measured slope is carried by T2 *information* rather than by target
geometry, trial ordering, or the estimator.

### 3.2 Sign-agnostic magnitude

Baseline s1: position spread rms **0.0132** (0.26 tolerance units), velocity spread rms
**0.1027**, peak 0.2897; control ~1e-16.

Velocity spread is ~8× position spread, confirming that in 1D the effect lives in the
velocity profile (see design doc §2.1). **Report `t2_v_spread`, not the position spread.**

---

## 4. Shaping does not affect the result **[established, n=8 per condition]**

All at `max_steps=600`, 3M steps, seeds 1–8, so the comparison is budget-matched.

| condition | competent | mean CI | sd |
| --- | --- | --- | --- |
| `pbrs` (baseline) | 8/8 | **+0.514** | 0.139 |
| `none` (sparse reward) | 7/8 | **+0.425** | 0.290 |
| `greedy` (coef 2, naive) | 7/8 | **+0.486** | 0.198 |

Welch t-tests on CI, competent runs only:

```
pbrs   vs none     diff −0.089   t=+0.74   p=0.48
pbrs   vs greedy   diff −0.028   t=+0.32   p=0.76
none   vs greedy   diff +0.061   t=−0.46   p=0.66
```

Levene test on spread: **W = 0.89, p = 0.43** — no evidence that shaping stabilises the
effect either.

**Three conclusions.**

1. **The shaping term is not the cause.** Sparse-reward agents co-articulate just as
   strongly, across 7 competent seeds.
2. **The naive greedy term does not suppress it**, at `coef = 2` — a dose where the greedy
   cost at a typical mid-reach distance (|x−T| ≈ 0.5) equals the time penalty exactly, and
   totals ~8% of the episode return. The design-doc argument for refusing the naive term
   remains sound *in principle* (it is not policy-invariant) but is not supported by an
   empirical need at this magnitude.
3. Across **22 competent agents spanning three reward specifications**, 21 have positive CI
   (the exception being `no_shaping_s6` at −0.114, which has `r = −0.175`, p = 0.31 — no
   signal; see §10).

### 4.1 Greedy dose-response **[pending]**

`shaping_coef` ∈ {6, 20, 60} × 8 seeds, `max_steps=600`. Scale reference:

| coef | cost/step at \|x−T\|=0.5 | vs time cost | ≈ episode total | vs return ~27 |
| --- | --- | --- | --- | --- |
| 2 (done) | 0.020 | 1× | 2.2 | 8% |
| 6 | 0.060 | 3× | 6.6 | 24% |
| 20 | 0.200 | 10× | 22.0 | 81% |
| 60 | 0.600 | 30× | 66.0 | 240% |

At coef 60 the greedy term exceeds the entire task return, so "stay near the current
target" is the dominant objective. If competence collapses there, that is itself the
boundary: the term cannot be pushed hard enough to remove lookahead without first removing
the task.

Mid-training success at coef 6 and 20 was ~1.00, so the term is not impairing task
learning at those doses.

---

## 5. C6 — information **[established, within-agent]**

Covered by §3.1. The between-agent `no_preview` arm was removed from the grid: an agent
trained with `H = 1` is out of distribution the moment a second target appears and reached
T1 on 0 of 36 preview-on trials, so it cannot be probed in the condition that matters. The
within-agent contrast is strictly stronger — it holds weights and competence fixed.

---

## 6. C1 — separability **[provisional, n=1 competent]**

`dwell_inter = 25`, `max_steps = 700`, seed 1: **success 1.000, multi 1.000, timeout
0.000, CI = −0.233 ± 0.016** (p = 5×10⁻¹⁶), per-direction −0.226 / −0.239
(`dir_agree = true`, both fits R² 0.80–0.91).

The first competent C1 agent obtained. Forcing rest at each target does not merely reduce
co-articulation — it reverses the sign, against baseline's +0.514.

**Caveats, both important.**

- **1 of 3 seeds** reached competence at this cap (others 0.479, 0.215). 8 seeds pending.
- **The index may not measure what C1 predicts.** The separability argument constrains
  velocity at *departure* from T1, which the dwell forces to zero by construction. The
  metric measures velocity at *entry*, deliberately, so that the dwell does not manufacture
  a zero — but that is a different quantity. A theory predicting exactly 0 and a measurement
  of −0.233 are not necessarily in conflict; they may be about different variables.
- `t2_v_spread` for this agent is low relative to baseline, consistent with a genuinely
  reduced T2 dependence, but the 8-seed set is needed before quantifying it.

**[pending]** 8 seeds at `max_steps=700`.

---

## 7. C7 — discount horizon **[provisional, n=1]**

γ = 0.97, sparse reward, 6M steps, seed 1: **success 1.000, multi 1.000, timeout 0.000**,
CI = **+0.048 ± 0.085, p = 0.57** — statistically indistinguishable from zero, against
baseline's +0.514.

The first γ = 0.97 agent above the competence floor. At 3M steps this arm sat at 0.596 and
was uninterpretable; the only change needed was more training.

### 7.1 γ = 0.97 does not make the agent indifferent to T2

Its trajectories do not collapse onto the T2-hidden control. The sign-agnostic metric:

| agent | x-spread rms | v-spread rms | v-spread peak | control |
| --- | --- | --- | --- | --- |
| baseline (γ=0.995) | 0.0132 | **0.1027** | 0.2897 | 9.8e-17 |
| gamma097_sparse (γ=0.97) | 0.0113 | **0.0457** | 0.1343 | 1.2e-16 |

The γ=0.97 agent retains **45% of baseline's T2-driven velocity modulation** and 86% of its
position modulation, against a control of ~1e-16. Its trajectory to T1 *is* conditioned on
T2.

**The correct claim is therefore narrower:** shortening the discount horizon to 0.97
abolishes *signed, monotonic co-articulation* (CI ≈ 0) while leaving substantial
T2-dependent trajectory modulation intact. Whether that residual is lookahead in another
form or unstructured sensitivity is open.

This is exactly the CI/`t2_dependence` disagreement the sign-agnostic metric was built to
surface, and it is the strongest argument in this document for never reporting CI alone.

**[pending]** 8 seeds at γ = 0.97.

### 7.2 Why γ = 0.90 is not a weaker version of the same test

At γ = 0.90 the arms sit at ~0.27 success with **~54% of trials in contact with the
workspace boundary** — a degenerate strategy, not a myopic-but-functional policy. The
arithmetic (design doc §5.2) shows the agent's entire visible payoff has shrunk toward the
cost of trying. γ = 0.90 removes the task, not just the lookahead, and cannot serve as a C7
condition.

---

## 8. C2 — effort cost **[provisional, 1 of 4 seeds]**

`w_effort = 0`, `w_rate = 0`. At the original `max_steps = 200` this arm scored 0.129
success with **87% timeouts** — a budget failure, not a result.

At `max_steps = 400`, seed 1: **success 0.998, multi 0.997, CI = +0.409 ± 0.051**
(p = 3×10⁻⁹) — statistically indistinguishable from baseline's +0.514. Removing the effort
cost does **not** abolish co-articulation.

But the arm is **seed-fragile**:

| seed | success | episode length |
| --- | --- | --- |
| s1 | **0.998** | 114 |
| s2 | 0.51 | 246 |
| s3 | 0.34 | 309 |
| s4 | 0.48 | 254 |

Three of four seeds fail, all with the timeout signature. A variant with
`ent_coef = 0.0005` reached 1.000 success with **CI = −0.582** — same task, same reward, a
PPO hyperparameter apart, opposite sign. Both were competent.

Two readings remain open: **degeneracy** (with no cost on arrival velocity the optimum is a
flat manifold, so the CI is decided by optimisation incidentals) versus **plain
non-reproducibility**. The failing seeds all show timeouts rather than "different but
equally good" trajectories, which favours the second.

**[pending]** 8 seeds at `max_steps=600`.

---

## 9. Behavioural signatures **[provisional, baseline only]**

From the competent baseline. These have not been re-derived at 8 seeds and should be
treated as qualitative until they are.

- **Preplanned, not repaired** — the reach diverges by T2 within 20–40 ms of the go cue, and
  the holding posture *before* the cue is already T2-biased.
- **Preparation is largely output-null** — pre-go state variance along the policy's output
  direction is a small fraction of a typical random direction.
- **Online replanning generalises for free** — an agent never trained on perturbations
  alters its ongoing reach to T1 when only T2 moves. It fails when T1 itself jumps outward.
- **Reaction time** decreases with hold duration, the classic signature, nowhere trained
  for.

---

## 10. What the co-articulation index can and cannot support **[established]**

Two failure modes, both with consequences for any number reported.

### 10.1 The two reach directions can disagree

`by_group` (per-direction slopes) is now persisted. Applied retrospectively:

| agent | reach right | reach left | pooled CI | `ci_r` | verdict |
| --- | --- | --- | --- | --- | --- |
| baseline | +0.526 (R² 0.96) | +0.713 (R² 0.99) | +0.622 | +0.98 | consistent |
| stop_at_targets | −0.226 (R² 0.80) | −0.239 (R² 0.91) | −0.233 | −0.93 | consistent |
| no_effort lowent | −0.526 (R² 0.77) | −0.634 (R² 0.98) | −0.582 | −0.94 | consistent |
| **with_jumps** | **−0.683** | **+0.233** | −0.213 | **−0.38** | **directions disagree** |

`with_jumps` (an arm since dropped) is the example: a significant pooled CI of −0.213
(p = 0.02) whose two reach directions give opposite signs, with `r` = −0.38 (15% of
variance). It is not a signed effect.

**`no_shaping_s6` CI = −0.114 is not a result**: `r` = −0.175, p = 0.31, and both
per-direction slopes are *positive* (+0.175, +0.354). The pooled negative slope is noise.

### 10.2 Probe asymmetry

T1 is fixed at ±0.45 while T2 sweeps ±0.9, so `d_next` spans `[−1.35, +0.45]` — **27
reversing vs 9 onward trials**. Three quarters of every slope is measured on one arm of the
relation, and with only 4–5 onward points per direction the shape cannot be tested. Any
curvature is converted into a spurious *linear* slope whose sign is set by the dominant arm.

`ci_balanced` (refit on the widest window symmetric about zero) is now reported and costs
no extra rollouts. A balanced probe design would be better and is not yet implemented.

---

## 11. Run inventory

72 runs in `1D/runs/`: 8 seeds per condition, 3M steps, `max_steps = 600` (700 for
`stop_at_targets`). The directories are gitignored; every checkpoint is on W&B
(`alighavam79-western-university/seqhorizon-1d`) as a `model` artifact.

| local name | W&B run name | W&B group | analysed |
| --- | --- | --- | --- |
| `baseline_s1–8` | `baseline_ms600_s*` | `followup-20260817` (s1–3), `shaping-8seed` (s4–8) | yes |
| `no_shaping_s1–8` | `no_shaping_ms600_s*` | `followup-20260817` (s1–3), `shaping-8seed` (s4–8) | yes |
| `greedy2_s1–8` | `greedy_ms600_s*` | `shaping-8seed` | yes |
| `greedy6_s1–8` | `greedy6_ms600_s*` | `greedy-dose` | yes |
| `greedy20_s1–8` | `greedy20_ms600_s*` | `greedy-dose` (s1), `pipeline-20260817` (s2–8) | yes |
| `greedy60_s1–8` | `greedy60_ms600_s*` | `pipeline-20260817` | s2–5 only; `analyze.py` crashes on the rest |
| `no_effort_s1–8` | same | `pipeline-20260817` | no |
| `gamma097_sparse_s1–8` | same | `pipeline-20260817` | no |
| `stop_at_targets_s1–8` | same | `pipeline-20260817` | no |

The local names dropped the `_ms600` suffix on 2026-09-22, once 600 was the default; W&B
keeps the original names. Everything else — the invalid 200-step grid (`ablations-3M`,
`ablations-fixed-20260817`), the budget sweeps, and the single-seed variants behind the
provisional numbers in §6–§8 — now lives only on W&B.
