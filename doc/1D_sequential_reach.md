# Emergent lookahead in 1D sequential reaching

**Status:** implemented. Code in `[1D/](../1D)`. This document is the standing reference
for the 1D stage of the project.

> ### ⚠️ All numbers below predate the 2026-08-16 fixes and are pending a re-run
>
> The first ablation campaign was invalidated by a diagnosis given in §6.6: seven of nine
> conditions never learned the task, and most of them spent half of every episode parked
> against the workspace boundary, which the environment made a free place to rest. Two
> changes landed in response — an **inelastic, charged workspace boundary** and a
> **curriculum over sequence length** — plus a **competence gate** that stops the analysis
> presenting an index for an agent that cannot do the task.
>
> Both changes alter training dynamics for every condition, so **every run under
> `1D/runs/` is stale**, including the baseline. The qualitative baseline findings (§6.2 –
> §6.4) are expected to survive — they come from the one arm that was demonstrably healthy
> — but the specific numbers must be re-derived. The ablation table in §6.5 should be read
> only as a record of what went wrong. See `plan.md` for the change log and the command
> sequence for the re-run.

---

## 1. The question

When people reach through a sequence of targets, the movement to target 1 already
reflects where targets 2 and 3 are. The first reach curves, its peak speed changes, and
it passes through target 1 carrying momentum appropriate to what comes next. Sequence
context is expressed in the *first* movement, before the later targets are relevant.

Networks trained to produce sequential reaches usually do not show this, and when they
do it is often because something in the training signal put it there — a reference
trajectory, a via-point loss, a smoothness prior over the whole path.

The goal here is the opposite: **set up conditions under which lookahead is the optimal
thing to do, train with RL, and then find out which of those conditions it actually
depended on.** The effect has to be emergent for the ablations to mean anything.

Three phenomena, in increasing order of difficulty:

1. **Lookahead / co-articulation** — the reach to T1 depends on T2.
2. **Delay-period preparation** — targets are shown, movement must wait for an
  unpredictable go cue, and the agent is nonetheless ready when it arrives.
3. **Online replanning** — a target moves mid-movement and the plan updates.

This document covers the 1D point-mass stage. The 2D / MotorNet stage inherits the
design; §9 says what transfers.

### Findings at a glance

Rows are marked by how much weight they can bear. **Solid** results come from the one arm
that reached competence; **pending** results need the re-run to survive; **methodological**
results are about the experiment rather than the agent, and are the most durable thing
this stage produced.


|                                        | result                                                                                                                                                                                                    | status              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| **Lookahead emerges**                  | Pass-through velocity at T1 tracks T2's offset: CI = **+0.549 ± 0.059** (p = 7×10⁻¹¹). Peak velocity and reach duration shift too. Nothing in the reward mentions trajectories.                           | solid, re-run due   |
| **It is caused by T2 information**     | Hiding T2 from the same agent drops CI to **machine zero** — trajectories become bit-identical across all T2 values. A within-agent contrast, so it does not depend on training two agents equally well.  | solid               |
| **It is preplanned, not repaired**     | The reach diverges by T2 within 20–40 ms of the go cue, and the *holding posture* before the cue is already T2-biased (p = 4×10⁻⁸).                                                                       | solid, re-run due   |
| **Preparation is largely output-null** | Pre-go state variance along the policy's output direction is 1.6% of a typical direction — below all 2000 random directions tested.                                                                       | solid, re-run due   |
| **Online replanning generalises free** | An agent never trained on perturbations alters its ongoing reach to T1 when only T2 moves. (It fails, though, when T1 itself jumps outward.)                                                              | solid, re-run due   |
| **Effort cost is not necessary**       | Removing the effort penalty entirely leaves CI = +0.417 at 99.4% success. Time pressure alone suffices. The only ablation arm besides baseline that reached competence.                                   | pending re-run      |
| **Discount horizon controls it**       | γ=0.97 vs γ=0.995 gave a 4× difference in CI — but both arms sat at 22% success, so this is matched *incompetence*, not matched competence. Neither completes a two-target sequence. See §6.5.            | **withdrawn**       |
| **Separability (C1) is unresolved**    | The direct test of §2. Its CI **flipped sign** (−0.64 → +0.25) when trained twice as long, so the 3M result was an artifact of a broken agent. It may also be untestable with this metric — §6.5.         | **withdrawn**       |
| **A broken agent's index is noise**    | Seven of nine arms scored ~100% on one-target trials and ~0% on two-target trials: single-reach policies, with no second movement to plan. Their indices measure nothing. Now gated in code.              | methodological      |
| **The environment had a trap**         | The workspace boundary clipped position without stopping or charging the mass, making it a free resting place that paid shaping income. The competent arms touched it 0% of the time; the failures, ~90%. | methodological, fixed |


---



## 2. Why lookahead has to be made possible before it can be made to emerge

It is worth being precise about this, because it determines the whole design and it is
the easiest thing to get wrong.

Write the sequence as targets `T₁…T_K` acquired in order, with plant state
`s = (x, v, a)` (position, velocity, activation). Total cost is additive over time, and
the trial breaks into segments at acquisition events.

**Case A — acquisition requires coming to rest.** If taking `T_k` requires dwelling
there long enough that `v ≈ 0`, then the state entering segment `k+1` is
`(x = T_k, v = 0, a = 0)` *regardless of how segment k was executed*. The optimal cost
separates:

```
J*  =  Σ_k  J*_k(T_{k-1} → T_k)
```

Every term depends on exactly two adjacent targets. Therefore

```
∂ (segment-k trajectory) / ∂ T_{k+1}  =  0
```

**identically.** Under this task structure no reward function, network, or training
budget can produce a horizon effect. It is ruled out by the structure of the problem,
not by the agent's competence. This is, I think, the main reason sequential-reaching
models often fail to show co-articulation: the task is written so that each reach ends
at rest, which factorises it.

**Case B — acquisition is positional, velocity free.** If taking `T_k` only requires
`|x − T_k| ≤ tol`, the velocity at that moment is a free decision variable, and the
Bellman recursion at the segment boundary is

```
J*_k(entry)  =  min over segment-k trajectories [ cost_k  +  J*_{k+1}(T_k, v_exit) ]
```

The optimiser must trade the cost of achieving `v_exit` during segment k against the
cost-to-go `J*_{k+1}`, which **depends on** `T_{k+1}`. The stationarity condition

```
∂ cost_k / ∂ v_exit   +   ∂ J*_{k+1} / ∂ v_exit   =  0
```

has a solution that is a function of `T_{k+1}`. Co-articulation is not encouraged here —
it is *forced by optimality*.

So the task must be in Case B. That is necessary, not sufficient. An RL agent also needs:

- **the information** — `T_{k+1}` must be observable during segment k;
- **the temporal reach** — the discount `γ` must be long enough that `J*_{k+1}` is
actually represented in the value function at the time segment k is being chosen.

These two requirements were meant to map onto the two headline ablations (`no_preview`,
`myopic`). Everything else in the ablation grid modulates *how much* arrival velocity is
worth optimising.

> **Both headline ablations turned out to be unsound as between-agent manipulations**, for
> different reasons, and this is worth knowing before reading §6. Removing the information
> (`no_preview`) also removes the agent's ability to parse the probe, so it cannot be
> tested by training a second agent — that arm has been **removed from the grid**; it *can*
> be tested within a single agent by hiding T2 at probe time, and that is the strongest
> result in this document (§6.2). Shortening the discount (`myopic`) does not selectively
> remove the incentive to look ahead; at γ = 0.90 it shrinks the incentive to do the task
> at all by a factor of ~470 (§6.5), so the two cannot be separated. The requirements above
> are still the right theory. The difficulty is that both are hard to ablate without also
> ablating the agent — which is the general case for testing a claim about *information* by
> training a second network, and the reason within-agent manipulations are preferred here.

### 1D is not a degenerate case of 2D

In 2D, co-articulation is usually read off as path curvature. In 1D there is no path to
curve — but curvature was never the underlying variable. The underlying variable is
**the plant state handed from one segment to the next**, and in 1D that is fully
captured by the signed velocity at pass-through. If anything the 1D readout is cleaner,
since 2D curvature confounds path geometry with speed profile.

The 1D prediction is monotonic and signed:


| where T2 is               | optimal way to take T1                        |
| ------------------------- | --------------------------------------------- |
| beyond T1, same direction | pass through fast, still moving outward       |
| back toward the start     | arrive slow or already decelerating/reversing |


which gives a single scalar summary — the slope of pass-through velocity against the
next target's offset.

---



## 3. Task

Point mass on a line, start at `x = 0`, 1–3 targets acquired in order.

```
   ┌── targets appear (visible ones only) ──┐
   │      HOLD at x≈0, random delay D       │ go cue    movement     final dwell
───┴────────────────────────────────────────┴─────────┬────────────┬────────────
   t=0                                  D~U[0.25,1.0]s│ hit T1,T2… │ 0.3 s in tol
                                          (unpredictable)
```

**Plant** — semi-implicit Euler, `dt = 20 ms`, 200 steps (4 s) max:

```
a  ← a + (dt/τ)·(clip(u, ±u_max) − a)          activation lag, τ = 50 ms
a_eff = a·(1 + σ·ε),  ε ~ N(0,1)               signal-dependent noise, σ = 0.05
v  ← v + dt·(a_eff − b·v)/m                    m = 1, b = 0.2
x  ← x + dt·v
if |x| > x_limit:  x ← ±x_limit,  v ← 0        inelastic boundary, x_limit = 1.5
```

`u_max = 8`. The activation lag matters: force cannot change instantaneously, so
momentum is a resource that has to be committed in advance. An optional sensory delay
(default 0) feeds the policy a stale `(x, v)`.

**The boundary is inelastic and charged for** (`−w_wall` per step of contact; see the
reward table below). This is not cosmetic and the reason is worth stating, because getting it wrong
cost an entire ablation campaign. The earlier version clipped position and left velocity
untouched and unpenalised, so the mass could pin itself against the limit and stay there
for nothing — and better than nothing, since potential-based shaping pays
`(1 − γ)·c·|x − T|` per step to an agent that holds still, which is `0.024/step` out at
the boundary against a time cost of `0.02/step`. The result was a large *flat* region of
the return landscape 0.6 units beyond any target the task ever poses. It was not an
exploit — solving the task is worth far more — but it was a plateau with no local
gradient out of it, and most of the ablation grid ended up sitting in it. §6.6 has the
measurements.

Because targets only ever appear within `±target_range = 0.9`, a policy that does the
task never makes contact and never pays `w_wall`. Measured: the two arms that reached
competence touched the boundary on **0% of trials**. The term removes a hiding place; it
does not shape the solution.

**Acquisition** — `|x − T_k| ≤ tol` (`tol = 0.05`) held for `dwell_k` steps, with
`dwell_inter = 1` (pass-through) and `dwell_final = 15` (0.3 s). The intermediate dwell
is the Case-A/Case-B switch from §2 and is an ablation axis.

**Preview** — a rolling visibility window: slot `i` is visible iff
`cur ≤ i ≤ cur + H − 1`. `H = 1` means only the current target is visible; `H = K` means
the whole sequence is visible from trial onset. Training mixes `H` across trials
(`full_preview_prob = 0.5`), which is what makes preview-on vs preview-off a
**within-agent** contrast at test time.

**Sequence length, and the curriculum over it** — each trial draws 1–3 targets with
weights `n_target_weights = (0.2, 0.3, 0.5)`, so half of all trials are full three-target
sequences. Meeting that mixture from step zero turned out to be the second thing wrong
with the first campaign. An untrained agent cannot complete a three-target sequence by
chance, but it can stumble into a single reach; early training therefore reinforces single
reaches, exploration noise shrinks, and the policy never discovers that chaining pays
more. Seven of nine arms ended up exactly there (§6.6).

Sampling weights are now annealed linearly from `curriculum_start_weights = (1, 0, 0)` to
`n_target_weights` over the first `curriculum_steps = 1,000,000` environment steps, then
held:

```
w(t) = (1 − f)·(1, 0, 0)  +  f·(0.2, 0.3, 0.5),      f = min(1, t / curriculum_steps)
```

Three properties make this safe to leave on by default, which it is:

- **It does not change the task.** Only the order in which trials are shown. The final
  mixture, and hence everything that is evaluated, is identical. Set `curriculum_steps=0`
  to disable.
- **It applies identically to every arm**, which is what an ablation grid requires.
- **Measurement cannot inherit it.** Probes and evaluation construct a fresh environment,
  which always starts at the final mixture; only the training loop advances the schedule.

`mean_n_targets` is logged so the schedule is visible in the training curve: it should
climb from 1.00 to 2.30 and then sit flat.

**Reward** — the entire task specification:


| term                                 | value                                 |
| ------------------------------------ | ------------------------------------- |
| intermediate acquisition             | `+10`                                 |
| final acquisition                    | `+20`, ends trial                     |
| effort                               | `−0.5·(u/u_max)²·dt`                  |
| control rate                         | `−0.05·((u − u_prev)/u_max)²`         |
| time                                 | `−1.0·dt` while targets remain        |
| hold violation before go             | `−20·(x−x₀)² − 2·v²`                  |
| false start (`|x−x₀| > 0.15` pre-go) | `−5`, aborts trial                    |
| workspace boundary contact           | `−0.1` per step at `|x| = x_limit`    |
| optional shaping                     | `γΦ(s′) − Φ(s)`, `Φ = −2·|x − x_cur|` |


No term references a trajectory, a velocity profile, a via point, a reaction time, or a
preparatory state.

The boundary term is sized against the two costs it competes with: it must beat the
`0.024/step` the shaping term pays for idling out there, and at `0.1` it costs roughly 4×
the time penalty, so an episode spent parked at the limit runs about `−15` — worse than
forgoing a target (`+10`), short of catastrophic. It is the only reward term that is
inert for a competent agent by construction.

**On the shaping term.** Sparse acquisition reward alone is a hard exploration problem
over 200 steps of continuous control. The shaping used is *potential-based*
(Ng, Harada & Russell 1999) and therefore provably leaves the optimal policy unchanged;
it only reshapes learning dynamics. A naive `−c·|x − x_current|` bonus is deliberately
**not** offered as an option, because a greedy pull toward the current target is
implicit supervision against exactly the effect being studied — it would penalise
carrying momentum past T1. The `no_shaping` ablation checks the headline result under
sparse reward alone.

**Observation** — `[x̂, v̂, a_prev, go_flag]` plus, per target slot,
`[visible, pending, is_current, (T_i − x̂)·visible]`. Targets are egocentric; hidden
targets read zero. No planning features, no precomputed plan, no time-remaining signal.

**Perturbations** — with probability `p_jump`, one target is displaced mid-movement,
either the one currently being reached to (`jump_current`, the classic double-step) or a
*later* one (`jump_future`). The second is the sharper probe: if only T2 moves and the
ongoing reach to T1 changes, the agent was demonstrably planning through T2.

---



## 4. Agent and algorithm

PPO, ported from CleanRL (`ppo_continuous_action.py`, plus the sequence-batched
recurrent update from `ppo_atari_lstm.py`): GAE(λ), clipped surrogate, clipped value
loss, entropy bonus, orthogonal init with `std=0.01` on the mean head, advantage
normalisation, LR annealing, gradient clipping.

Policy: GRU(128) over a 2×128 tanh trunk, Gaussian head with state-independent log-std.
The recurrent state is also the object of the neural analysis in §6.

Defaults: 32 envs × 256 steps, `γ = 0.995`, `λ = 0.95`, lr 3e-4, 4 epochs,
4 minibatches, clip 0.2, entropy coef 0.003.

**Cost, measured.** ~3300 environment steps/s on one core of a 2015 i7-6700K with video
off, so 3M steps is ~15 min per run. Two facts about that number matter for how the grid
should be run:

- **Rendering during training costs 2.8× throughput** (3300 → 1180 steps/s at
  `--video-every 5`). Use `--no-video` for grid runs and render from the checkpoint
  afterwards with `make_video.py`. The first campaign ran at 221 steps/s largely because
  of this.
- **This workload does not want a GPU, and adding threads makes it slower.** Per
  8192-step batch the rollout is 250 ms and the PPO update is 2180 ms, so ~90% of the time
  is the update — and the update is a Python loop calling `nn.GRU` one timestep at a time
  (256 sequential calls on an 8×128 tensor per minibatch). Doubling the sequence length
  doubles the update time with no FLOP scaling, and 8× the batch width costs only 2.7×:
  it is launch-latency bound, not compute bound. On CUDA that becomes ~4096 tiny kernel
  launches per iteration on matrices too small to occupy one SM. Four threads is *slower*
  than one (2.34 s vs 2.18 s per update) because the matmuls are too small to parallelise.
  **Run many single-threaded jobs across cores instead** (`--parallel N --threads 1`).
  A GPU port would first require replacing the per-timestep loop with a single fused
  sequence call and pushing `num_envs` past 1024 — worth revisiting at MotorNet, not here.

The training log carries three fields specifically for catching the failure modes in §6.6:
`episodic_return_task` (return excluding shaping — watch this, not `episodic_return`),
`wall_rate` (fraction of episodes touching the workspace boundary; should be ~0), and
`mean_n_targets` (curriculum position).

Two deliberate departures from the reference scripts, both corrections:

- **Own batched environment** instead of `gym.vector.SyncVectorEnv`. Gymnasium 1.0
changed vector autoreset to next-step semantics, which silently breaks CleanRL's
bootstrapping; owning the vectorisation avoids the trap and is far faster on CPU. The
env uses same-step autoreset — the semantics CleanRL's GAE loop assumes — and
additionally reports the pre-reset terminal state so probes can record it.
- **Correct time-limit bootstrapping.** Truncated transitions are bootstrapped with
`γ·V(final_obs)`. The stock CleanRL scripts treat truncation as termination, which
biases the value function; here a failed trial *is* a timeout, so that case dominates
early training.

---



## 5. Measurements

All post hoc, none of them in any loss.

### 5.0 The competence gate — which comes first

Every index below describes the *shape of a movement the agent is making*. If it is not
making the movement, the index is not measuring the thing it is named after. This sounds
obvious and was nevertheless the single largest error in the first campaign, so it is now
enforced in code rather than left to the reader.

`analyze.py` measures competence before anything else and reports three things:

- **success rate**, and whether it clears a floor of 90% sequence completion
  (`--competence-floor`; 0 disables the gate);
- **success split by sequence length** — the diagnostic that matters. An agent at ~1.00 on
  one-target trials and ~0.00 on two-target trials has learned a single reach, not the
  task, and there is no second movement for a lookahead index to describe. Seven of nine
  arms in the first grid had exactly this profile while showing an overall success rate
  (0.17 – 0.22) that reads as "partially competent";
- **failure attribution** — false start vs timeout vs boundary contact, so a failure has a
  cause attached rather than just a rate.

Below the floor, indices are still computed and written to `results.json` (they are useful
diagnostics) but are printed flagged, `ci_interpretable` is set false, and
`fig6_ablations.png` draws those arms hatched and greyed.

**Probe attrition** is checked independently of the floor. The CI probe fixes T1 and
sweeps T2, so all 36 preview-on trials should contribute; a short count means the policy
could not reach T1 on the rest and the slope is fit to a biased subset. This check alone
would have caught two failures in the first grid that the success rate did not: both
`stop_at_targets` and `myopic` yielded indices fit to 18 trials that were **all in one
reach direction**, because neither policy could reach rightward at all.

### 5.1 Behavioural indices

**Co-articulation index (CI).** Fix T1, sweep T2 over a grid, and record the signed
velocity at the moment the mass **first enters** T1's tolerance band. Fold by reach
direction so both directions pool:

```
v_out   = v(at first entry into T1's band) · sign(T1)
d_next  = (T2 − T1) · sign(T1)                 > 0 means T2 continues onward
CI      = d v_out / d d_next                   (OLS slope)
```

`CI > 0` = passing through T1 with momentum toward T2. `CI ≈ 0` = no lookahead.

**Entry, not acquisition.** The two coincide under the default `dwell_inter = 1`, where a
target is acquired on the step it is first touched. They diverge sharply once a dwell is
required: acquisition is then registered only after the dwell completes, at which point
velocity is ~0 *by construction* rather than by absence of lookahead — which would
manufacture a confirmation of the separability prediction in exactly the condition meant
to test it. Timing at entry keeps the index comparable across dwell settings.

This fix is necessary but, as §6.5 explains, probably not sufficient: the separability
argument of §2 constrains velocity at *departure* from a target, which entry timing does
not measure either. Under a dwell requirement the index measures something real but not
the thing §2 speaks to.

The slope is fit **within reach direction** (a fixed-effects estimator: `d_next` and
`v_out` are demeaned within each direction before regression). This is not cosmetic. The
learned policy is not mirror-symmetric — leftward and rightward reaches leave T1 at
different mean speeds (1.15 vs 1.73 in the baseline) — and the two directions do not
span the same range of `d_next`. Regressing on the pooled cloud therefore picks up a
between-direction offset unrelated to T2, which put a spurious `−0.007` on a control
whose true value is exactly `0`. Per-direction slopes are reported alongside the
combined estimate, and they do differ substantially (see §6).

**The control.** The identical grid is re-run with T2 hidden (`H = 1`). Same weights,
same physics, same T1, same T2 values — the only difference is whether T2 is in the
observation. Because the policy is deterministic and probes are noiseless, the hidden
condition yields **bit-identical** trajectories across every T2 value (measured
across-trial SD of pass-through velocity: exactly 0), so its CI is exactly 0. That is
the point: it certifies that the measured slope is carried by T2 information and cannot
come from target geometry, trial ordering, or the estimator.

Secondary: slope of peak velocity and of segment-1 duration against `d_next`; and the
**divergence time** — the earliest moment after the go cue at which reaches to the same
T1 differ across T2 values.

**Reaction time.** Go cue to movement onset (5% of that trial's peak speed), as a
function of hold duration. A decrease with longer holds is the classic monkey/human
signature and is nowhere trained for.

**Jump response.** Correction latency after a target displacement, and — the
informative one — whether a `jump_future` perturbs the reach to T1 *before* T1 is taken.

**Preparatory geometry.** The mean action is a linear readout `u = w·h` of the GRU
state, so `w` spans a 1-D "output-potent" direction and its orthogonal complement is
output-null. Comparing across-trial variance of the pre-go state along `w` against a
typical random direction tests whether preparation specifies the movement while
avoiding driving the plant.

> A caution that applies to the decoding numbers: T1 and T2 are *present in the
> observation* throughout the hold period, so decoding them from the pre-go state is a
> pass-through check, not evidence of preparation — a low value would indicate a bug.
> The informative decodes are of upcoming *movement parameters* (peak velocity,
> segment duration), which are not in the input.

---



## 6. Results

Baseline: GRU policy, 3M steps, single seed (`1D/runs/baseline_s1`). Probes are
noiseless and the policy is evaluated deterministically.

### 6.1 Competence

97.1% of sequences completed (100% / 98% / 94% on one-, two- and three-target trials),
97.9% of targets acquired, 0% false starts, 0% boundary contact. Return climbs from −12 to
+28. This arm clears the competence floor of §5.0 comfortably, which is what licenses
reading anything off it — and it is one of only two arms in the grid that did. §6.5 and
§6.6 are largely a story about the other seven.

### 6.2 Lookahead emerges, and it is carried by target-2 information

With T2 visible, the velocity with which the mass passes through T1 depends strongly on
where T2 is:


| quantity                       | slope vs `d_next`  | p           |
| ------------------------------ | ------------------ | ----------- |
| **pass-through velocity (CI)** | **+0.549 ± 0.059** | 6.6 × 10⁻¹¹ |
| peak velocity of reach 1       | +0.152             | 6.3 × 10⁻⁷  |
| duration of reach 1            | −0.021 s           | 2.9 × 10⁻⁵  |


`r = 0.85`. When T2 lies further along the direction of travel the agent passes through
T1 faster, peaks faster, and gets there sooner. When T2 lies back toward the start it
arrives slowed. This is co-articulation, and nothing in the reward asked for it.

**The control is exact.** Re-running the identical grid with T2 hidden gives
`CI = −2 × 10⁻³²` — machine zero. With T2 removed from the observation the trajectories
are bit-identical across every T2 value (across-trial SD of pass-through velocity:
exactly 0). So the entire effect is carried by T2 information; none of it can come from
target geometry, trial ordering, or the estimator. **This within-agent contrast is the
proper form of the C6 ablation** — same weights, same physics, information removed —
and is stronger than the between-agent version attempted in §6.5.

Per-direction slopes differ substantially (+0.80 rightward, +0.31 leftward): the learned
policy is not mirror-symmetric. This is why the index is fit within direction (§5.1).

**The effect survives motor noise.** Re-running the same probes with signal-dependent
noise on (`analyze.py --noisy`, written to `results_noisy.json`) gives
CI = **+0.535 ± 0.058** (p = 8.8 × 10⁻¹¹) — statistically indistinguishable from the
noiseless +0.549. The control is no longer exactly zero (+0.007), as it must not be once
trials differ by noise, but it remains two orders of magnitude below the effect. The
output-null ratio is unchanged at 0.02, and the RT-vs-delay slope actually reaches
significance under noise (−7.3 ms/s, p = 0.033).

**The influence is present from movement onset — and before it.** Two things, which had
to be separated carefully:

*Before the go cue*, the agent already adopts a slightly T2-dependent holding posture.
The spread of hold position across T2 conditions is 0.010–0.023 (SD 3–8 mm on a ±0.9
workspace), and it regresses reliably on the next-target offset (`p = 4.6 × 10⁻³` and
`3.9 × 10⁻⁸` for the two directions). With T2 hidden this spread is **exactly zero**. So
a small amount of the plan is expressed posturally during the hold, well inside the
0.15 abort window.

*After the go cue*, the movement diverges beyond that pre-existing offset within
**20–40 ms** (1–2 simulation steps), and the velocity profiles separate visibly through
the deceleration phase (fig 2, left panel).

Measuring this correctly required referencing divergence to the pre-go spread rather
than to zero; against zero, the postural bias alone reads as "diverged at movement
onset" and the measurement says nothing. Either way the conclusion holds: the agent is
not executing a default reach and repairing it later — the first movement is *already*
the one shaped by T2.

### 6.3 Delay period

The agent holds position through an unpredictable 0.25–1.0 s delay and releases on the
go cue. Reaction time falls monotonically with hold duration — 28 → 24 → 20 → 21 →
19 ms — a slope of −8.5 ms per second of preparation (`r = −0.86`, `p = 0.060`).

Both the direction and the *shape* match the animal literature: RT falls steeply over
the first ~0.5 s of preparation and then saturates (fig 3), rather than declining
linearly.

But **this result is weak and the task is the reason.** With no sensory delay, one
simulation step is 20 ms, so the floor on RT is 20 ms and the agent is sitting on it —
the entire effect spans less than half a step, and the saturation may simply be the
floor. Measuring preparation properly needs `plant.sensory_delay > 0` to move RT off the
floor; that knob exists and is untested.

**Preparatory activity is largely output-null.** Across-trial variance of the pre-go GRU
state along the policy's output direction is 1.6% of that along a typical direction
(`0.00124` vs `0.0753`), and *lower than all 2000 random directions tested*
(percentile 0.00). The network configures itself for the upcoming movement while
keeping that configuration mostly off the axis that drives the plant — the
Churchland/Kaufman output-null preparatory geometry, emerging with no term in the reward
referring to it. The delay-period state is low-dimensional: the first two PCs carry 93%
of the variance.

"Largely", not perfectly: the residual 1.6% is exactly what shows up as the postural
bias in §6.2. Preparation here suppresses output leakage by roughly two orders of
magnitude rather than nulling it, which is a sharper claim than "output-null" and worth
keeping distinct.

> The decoding numbers (`R² ≈ 1.0` for T1, T2, upcoming peak velocity, and duration) are
> **not** evidence for this and should not be quoted as such. Targets are in the
> observation throughout the hold, the policy is deterministic, and the probes are
> noiseless — so every property of the upcoming movement is a deterministic function of
> the input, and perfect decoding is guaranteed for any network that carries its input
> forward. The geometry result above is the load-bearing one.



### 6.4 Online replanning, without ever training on perturbations

The baseline was trained with `p_jump = 0`. It has never seen a target move. Tested on
displacements anyway (δ = 0.35, applied 160 ms after the go cue):


| perturbation                      | outcome               | correction latency | deviation of reach 1 *before T1 is taken* |
| --------------------------------- | --------------------- | ------------------ | ----------------------------------------- |
| **T2 jumps (future)**, both signs | succeeds              | 100–120 ms         | **0.011 – 0.014**                         |
| T1 jumps inward                   | succeeds              | 60 ms              | not measurable                            |
| T1 jumps outward                  | **fails — times out** | 80 ms              | —                                         |


**The first row is the result.** Only T2 moved; T1 did not. The agent nonetheless altered
its ongoing reach to T1 *before arriving there*, and shifted the T1 acquisition time by
one step. Because probes are noiseless and the policy deterministic, the only difference
between these trials and their no-jump controls is T2's position — so this deviation is
a real causal effect. The plan for the current movement is being recomputed against the
future target, in a policy that never experienced a perturbation in training. A feedback
policy conditioned on the whole visible sequence gets that for free.

The other two rows are honest caveats, not results:

- When T1 jumps *inward*, it lands essentially on top of the mass and is acquired on the
same step as the jump, leaving no window in which to measure a deviation. The "0" here
is a measurement artifact, not a null result.
- When T1 jumps *outward*, the baseline **fails outright**: it overshoots, oscillates
around the displaced target without ever satisfying the dwell, and times out at 4 s
(fig 4, right panel). Online correction generalising for free has limits, and this is
where it breaks. This is what the `with_jumps` training condition exists to address.



### 6.5 Ablations — what the first grid actually showed

All conditions: 3M steps, seed 1, identical budget, pre-fix environment
(`stop_at_targets_long` is the same condition at 6M). **This table is a record of a failed
campaign, not a result table.** It is kept because the failure is the useful part.

The columns that decide how to read each row are `succ` and `multi` — overall sequence
success, and success restricted to multi-target trials. Everything with `multi ≈ 0.00`
learned a single reach.


| condition                        | CI         | ±se   | succ  | multi | 1-tgt | wall | verdict                          |
| -------------------------------- | ---------- | ----- | ----- | ----- | ----- | ---- | -------------------------------- |
| **baseline**                     | **+0.549** | 0.059 | 0.971 | 0.96  | 1.00  | 0.00 | ✅ competent — reference          |
| `no_effort` (C2)                 | **+0.417** | 0.036 | 0.994 | 0.99  | 1.00  | 0.00 | ✅ competent                      |
| `no_preview` (C6, between-agent) | NaN        | —     | 0.520 | 0.35  | 1.00  | 0.47 | ❌ unprobeable — **arm removed**  |
| `stop_at_targets_long` (C1, 6M)  | +0.253     | 0.054 | 0.324 | 0.15  | 1.00  | 0.31 | ❌ single-reach policy            |
| `no_shaping` (γ=0.995, sparse)   | +0.160     | 0.047 | 0.220 | 0.00  | 1.00  | 0.80 | ❌ single-reach policy            |
| `gamma097_sparse` (C7)           | +0.040     | 0.012 | 0.210 | 0.00  | 1.00  | 0.80 | ❌ single-reach policy            |
| `with_jumps`                     | +0.027     | 0.030 | 0.210 | 0.00  | 1.00  | 0.80 | ❌ single-reach policy            |
| `stop_at_targets` (C1, 3M)       | −0.642     | 0.040 | 0.170 | 0.00  | 0.80  | 0.26 | ❌ + fit to one reach direction   |
| `myopic_sparse` (γ=0.90)         | +0.046     | 0.016 | 0.000 | 0.00  | 0.02  | 0.93 | ❌ does not reach at all          |
| `myopic` (γ=0.90, shaped)        | −0.015     | 0.001 | 0.000 | 0.00  | 0.00  | 1.00 | ❌ does not reach at all          |
| `mlp` (memoryless)               | —          | —     | —     | —     | —     | —    | never trained                    |


**Seven of nine conditions are at 100% on one-target trials and 0% on two-target
trials.** That is not partial competence — it is a policy that makes one reach and stops,
and asking whether it co-articulates is asking whether a policy that plans no second
movement plans its second movement early. The CI column for those rows is a number
computed from movements that were never part of a sequence.

Two further defects are invisible in the success rate alone:

- **`stop_at_targets` and `myopic` produced indices fit to a single reach direction.** The
  probe sweeps T2 in both directions, 36 trials; both arms contributed 18, all leftward,
  because neither policy could reach rightward at all. This is now caught by the probe
  attrition check (§5.0).
- **Every failing arm was parked against the workspace boundary** for a large fraction of
  every episode — see §6.6, which is the actual explanation for the whole column.

Only comparisons at matched competence carry information. In this grid, exactly **one**
does: `no_effort` against `baseline`.

**C6 (information) — confirmed within-agent. The between-agent arm has been removed from
the grid** (2026-08-16). Two independent reasons, either sufficient.

*It cannot be probed.* `no_preview` was trained with only one target ever visible, so
showing it two at probe time is out of distribution. On preview-off trials — its training
condition — it reaches T1 on **36 of 36** and completes them. On preview-on trials it
reaches T1 on **0 of 36**. The policy breaks the moment an observation slot it has never
seen lights up. The NaN in the summary table was that, not a measurement of absent
lookahead.

*The one condition it can be probed in is a tautology.* At `H = 1` the observation contains
no information about T2 at all. With a deterministic policy and noiseless probes, two
trials sharing T1 but differing in T2 have bit-identical observation sequences, hence
identical actions, hence identical trajectories. Velocity at T1 is the same floating-point
number. **The slope is exactly zero for any agent whatsoever** — it verifies the pipeline,
it cannot test a hypothesis. That is what the −2 × 10⁻³² in the control column is.

And the measurement it was meant to provide already exists, better: the within-agent
control of §6.2 takes the *same weights, same physics, same competence* and removes T2
from the observation, giving CI +0.549 → machine zero. Information about the next target
is necessary, and the effect is carried entirely by it.

That within-agent contrast is the strongest result in this document, and the reason is
worth naming: **it does not require training two agents to comparable competence** — the
assumption that invalidated almost everything else here. A causal manipulation applied to
one frozen agent is immune to the confound that a between-agent ablation cannot escape.

The arm is not recoverable, either: the only way to make the probe in-distribution is to
show the agent two-target trials during training, at which point it is no longer the "no
preview" manipulation. `task.fixed_preview` remains available as a config knob; it is
simply not an ablation arm. What is lost by dropping it is *not* anything about
lookahead — only the separate question of whether an agent raised without preview builds a
structurally different controller, which is about representation learning and would need
its own probe design.

**C7 (discount horizon) — withdrawn.** This was previously reported as the cleanest
between-agent result in the grid, on the grounds that two arms of the matched sparse-reward
γ series landed at near-identical competence:


| γ     | effective horizon  | success | success on multi-target | CI         |
| ----- | ------------------ | ------- | ----------------------- | ---------- |
| 0.97  | ~33 steps (0.67 s) | 0.210   | **0.00**                | **+0.040** |
| 0.995 | ~200 steps (4 s)   | 0.220   | **0.00**                | **+0.160** |


They are matched, but they are matched at **22% success with zero completed two-target
sequences each**. That is matched *incompetence*. Both arms reach T1 and stop; the 4×
difference is between two policies that never produce a second movement. The comparison is
not necessarily empty — both do see T2, and their approach to T1 does vary with it — but it
cannot support a claim about lookahead in sequential reaching, and it should not have been
presented as the headline C7 result. It needs re-running above the competence floor.

**The γ = 0.90 arms are not a weaker version of the same test — they are a different
failure, and an instructive one.** Measuring the *discounted* return of a task-solving PD
controller against a controller that simply runs away from the target:


| γ     | solve  | run away | gap on the objective |
| ----- | ------ | -------- | -------------------- |
| 0.995 | +19.25 | −4.46    | **23.71**            |
| 0.90  | +0.03  | −0.02    | **0.05**             |


Potential-based shaping remains policy-invariant — solving wins at both discounts, so this
is not a reward inversion. What γ = 0.90 does is shrink the entire learnable signal by a
factor of ~470. The arm does not remove the incentive to look ahead; it removes almost all
incentive to do anything, so it cannot separate the two. No curriculum fixes that. It
should be reported as a demonstration of the confound rather than as ablation C7.

There is a monitoring trap attached to it. With shaping on at γ = 0.90, the *undiscounted*
return of parking far from the target (+52.8) exceeds that of solving the task (+36.8) —
so `myopic` logged **the highest episodic return of any run in the campaign, 46.7, at
0.000 success**. Its training curve looked like the best in the grid. The log now carries
`episodic_return_task`, which excludes shaping, and that is the field to watch.

**C2 (effort cost) — not necessary.** `no_effort` removes both the `u²` effort penalty
and the control-rate penalty, and reaches the *highest* competence in the grid (99.4%)
while retaining a large, highly significant CI of +0.417. Lookahead is reduced ~24%
relative to baseline but nowhere near abolished. So the effort cost modulates the effect
without being required for it — **time pressure alone is enough to make arrival velocity
worth optimising.** This was the ablation I expected to matter most, and it is the one
that clearly does not.

**C1 (separability) — withdrawn, and the most instructive failure in the grid.**
`stop_at_targets` is the direct test of the central claim in §2. Run at two budgets:


| budget   | success | CI                              | probe trials used | control |
| -------- | ------- | ------------------------------- | ----------------- | ------- |
| 3M steps | 0.170   | **−0.642** ± 0.040              | 18 / 36, one direction | 0.000 |
| 6M steps | 0.324   | **+0.253** ± 0.054 (p = 4×10⁻⁵) | 36 / 36           | 0.000   |


**The sign flipped as competence improved.** This is the single most useful number the
campaign produced, and it is a warning rather than a finding: reading the 3M value as "the
dwell abolished co-articulation" would have confirmed the theory in §2 for entirely the
wrong reason. Neither run settles C1 — the 6M agent still completes a third of sequences
against the baseline's 97%, and its two-target success is 0.15.

**The condition is feasible, but its time budget was not compensated — a third defect,
specific to this arm.** A hand-coded near-time-optimal stop-and-go controller simulated
against the real plant completes 99% of trials at `dwell_inter = 25` within the 200-step
limit, so nothing is impossible. But split that by sequence length and the margin tells a
different story:


| condition                       | median steps | p90 | slack at p90 |
| ------------------------------- | ------------ | --- | ------------ |
| baseline, 3 targets, dwell = 1   | 110          | 129 | **36%**      |
| `stop_at_targets`, 3 targets, dwell = 25 | 164 | 186 | **7%**       |


A 25-step dwell adds 24 steps of *mandatory holding* at each intermediate target — 48
steps on a three-target trial — and eats almost all of the remaining budget. A learned
policy is never time-optimal, so three-target trials become effectively impossible while
shorter ones stay easy. That is exactly the measured profile after the §6.6 fixes landed:
**0.94 success on two-target trials, 0.00 on three-target**.

So the arm was testing separability *and* a 24% cut to the movement-time budget at once.
`max_steps` is now raised to 250 for this condition — giving back precisely the 48 steps
the manipulation costs — which restores slack to 26%. Change one thing at a time.

Two further issues remain open for when this is run properly:

- This condition forced the metric change described in §5.1. With a 25-step dwell,
  "velocity at acquisition" is measured *after* the dwell and is ~0 by construction, which
  would have manufactured a spurious confirmation. Pass-through is now timed at first entry
  into the tolerance band.
- **But that change may not be sufficient, and this is a genuine open problem.** The §2
  separability argument constrains the velocity handed *out* of segment k — the state at
  **departure** — which a dwell forces to ~0. Timing at **entry** measures something the
  argument does not constrain: entry speed under a dwell is bounded by roughly
  `√(2·u_max·tol) ≈ 1.26`, not zero, and is free to vary with T2. So a positive CI-at-entry
  under a dwell requirement would not contradict §2, and a null would not confirm it.
  **`stop_at_targets` may not be testable with this metric family at all.** Deciding what
  the right readout is — departure velocity, total segment cost, something else — is a
  design decision that should be made deliberately before the condition is re-run, not
  settled by whichever number comes out.
- Separately, §2 shows a dwell severs the *dynamical* hand-off between segments. It does
  not rule out other couplings. A positive CI under a dwell that survives at competence
  would mean a second channel is carrying sequence context — a global time or effort
  budget allocated across segments is the obvious candidate — and that would be a more
  interesting result than the clean null.

**Shaping is not the source of the effect — but this cannot be concluded from
`no_shaping` yet.** That arm shows a positive, significant CI (+0.160) under sparse reward
alone, which is suggestive, but at 22% success and 0.00 two-target success it is a
single-reach policy like the rest. The claim needs the re-run. What *can* be said now is
narrower and comes from the reward analysis rather than from training: potential-based
shaping is policy-invariant at every γ tested (table above), so it cannot change which
policy is optimal — only how fast it is found.

---



### 6.6 Why seven of nine conditions failed

This section is the actual result of the first campaign. Both causes are now fixed in the
environment (§3), and both were found by looking at *why* runs failed rather than at how
often.

**Cause 1: the workspace boundary was a free place to rest.** The environment clipped
position at `|x| = 1.5` and did nothing else — velocity untouched, no cost. So the mass
could pin itself at the limit and stay. Worse, staying was very slightly *profitable*:
potential-based shaping pays `(1 − γ)·c·|x − T|` per step to an agent holding still, which
out at the boundary is `0.024/step` against a time cost of `0.02/step`. Net ≈ 0 — a flat
plateau in the return landscape, 0.6 units beyond any target the task ever poses, with no
local gradient out of it.

Measured contact rates over 512 trials per policy:


| run                     | trials touching the boundary | steps there (of 200) |
| ----------------------- | ---------------------------- | -------------------- |
| **baseline**            | **0%**                       | 0                    |
| **no_effort**           | **0%**                       | 0                    |
| `myopic`                | 100%                         | 106                  |
| `myopic_sparse`         | 93%                          | 96                   |
| `gamma097_sparse`       | 80%                          | 91                   |
| `with_jumps`            | 80%                          | 89                   |
| `no_shaping`            | 80%                          | 88                   |
| `no_preview`            | 47%                          | 36                   |
| `stop_at_targets_long`  | 31%                          | 4                    |
| `stop_at_targets`       | 26%                          | 19                   |


The split is total: the two arms that learned the task never go near the boundary; every
arm that failed lives there. The example trajectories show it directly — `with_jumps`
acquires T1 correctly, then accelerates to `x = 1.5` at `|v| = 15` and parks for the rest
of the trial; `stop_at_targets` either does the same or enters a sustained ~1.5 Hz limit
cycle around a target that it can never hold still enough to dwell in.

It is worth being precise that this was **not a reward exploit**. Solving the task is worth
far more than parking (+31.5 vs −6.1 undiscounted at γ = 0.995), and PBRS remains
policy-invariant. It was a *plateau*: a region where the gradient vanishes, which
absorbs a policy that wanders in. Fixed by making the boundary inelastic and charging
`w_wall` per step of contact (§3).

**Cause 2: full-length sequences from step zero.** Half of all trials are three-target
sequences from the first training step. An untrained agent cannot complete one by chance
but can stumble into a single reach, so early training reinforces single reaches; the
policy commits, exploration collapses, and chaining is never discovered. The evidence is
the 100%-on-one / 0%-on-two profile across seven arms, together with entropy: the healthy
baseline ends at +0.39, while `stop_at_targets_long` collapsed to **−0.10** (log-std −1.52
vs −1.03) — premature convergence, not insufficient budget. Consistent with that, doubling
`stop_at_targets`'s budget from 3M to 6M moved success only 0.17 → 0.32. Fixed by the
sequence-length curriculum (§3).

**The general lesson, which is the one that carries forward:** a fixed training budget
across ablation arms silently converts "this coupling was load-bearing" into "this
condition was harder to learn", and the two are indistinguishable from the success rate
alone. What distinguishes them is the *failure mode* — which trials fail, and what the
policy does instead. That is now measured before anything else (§5.0).

### 6.6.1 Did the fixes work? A three-arm check

Three conditions re-trained at 3M steps, seed 1, under the fixed environment. This is a
smoke test on one seed, not the campaign — but it is enough to say whether the diagnosis
was right.


| condition         | success (old → new) | boundary contact (old → new) | end entropy (old → new) | failure profile, new           |
| ----------------- | ------------------- | ---------------------------- | ----------------------- | ------------------------------ |
| `baseline`        | 0.971 → **0.973**   | 0.00 → 0.00                  | +0.39 → +0.43           | 1tgt 1.00, 2tgt 0.98, 3tgt 0.96 |
| `with_jumps`      | 0.210 → **0.539**   | 0.80 → **0.00**              | +0.40 → +0.36           | 1tgt 1.00, 2tgt **0.99**, 3tgt 0.12 |
| `stop_at_targets` | 0.170 → **0.332**   | 0.26 → **0.08**              | +0.30 → +0.27           | 1tgt 0.43, 2tgt **0.94**, 3tgt 0.00 |


What this establishes:

- **The boundary trap is closed.** Contact goes to ~0 in every arm, and no new degenerate
  resting place replaced it.
- **The single-reach plateau is broken, which was the whole point.** `with_jumps` went from
  **0.00 to 0.99** on two-target trials and `stop_at_targets` from **0.00 to 0.94**. Neither
  is a single-reach policy any more. Both now fail only on the longest sequences, which is
  an ordinary difficulty gradient rather than a qualitative failure.
- **Baseline is unharmed.** 0.973 vs 0.971 — the curriculum costs some early progress
  (it is still on easier trials at 1M steps) and pays it back by 3M.
- **`stop_at_targets`'s residual 3-target failure was a fourth, separate defect** — an
  uncompensated time budget, diagnosed above and now fixed. That fix is *not* included in
  the numbers in this table.

What it does not establish: nothing here clears the 90% competence floor except
`baseline`, so no ablation comparison is yet licensed. That needs the full grid at 6M
steps across 5 seeds.

**One number worth flagging before the re-run.** Baseline's CI moved from **+0.549 to
+0.854** under the new code at identical competence. The direction and significance hold
(p = 2 × 10⁻¹⁷, control exactly zero), so the qualitative finding is robust — but the
magnitude is clearly not stable across a change in training dynamics. Treat CI magnitudes
as seed- and schedule-dependent until the multi-seed campaign says otherwise, and compare
conditions within a campaign rather than across them.

---



## 7. Test experiments and the data they produce

`analyze.py` computes the specific indices in §6 and writes figures. It is deliberately
opinionated, which makes it the wrong tool for exploratory analysis. Alongside it there is
a general path: **define an experiment, run a frozen agent through it, get dataframes.**

An *experiment* here means what it means in the lab — a trial list with a factorial
design, run on a subject that is not learning during it. Weights are loaded from a
checkpoint and never updated.

```bash
.venv/bin/python 1D/scripts/run_experiment.py --list
```

```bash
.venv/bin/python 1D/scripts/run_experiment.py --run 1D/runs/baseline_s1 --exp preview_sweep
```

Experiments live in [`1D/seqreach/experiments.py`](../1D/seqreach/experiments.py), one
function each, in a registry. The shipped set:

| experiment | manipulates |
| ---------- | ----------- |
| `preview_sweep`   | T1 fixed, T2 swept, crossed with T2 visible/hidden and direction — the co-articulation protocol |
| `delay_sweep`     | duration of the pre-go hold — reaction time and preparatory geometry |
| `target_jump`     | a target displaced mid-movement, current or future, against matched no-jump controls |
| `random_task`     | the training distribution itself — competence and failure profile |
| `sequence_length` | T1 fixed, varying how many targets follow it |

Adding one is a function returning a trial list; any keyword argument it declares becomes
settable as `--param name=value`. `factors={...}` is free-form and becomes columns in the
output, so put the independent variables of your design there.

### The two tables

**timeseries** — one row per (trial, sample): `t_ms`, `t_from_go_ms`, `x`, `v`, `a`, `u`,
which target is currently being pursued, and the target positions *as they stand at that
sample* (so jumps are visible). The trial's condition columns are repeated down its rows,
so the table stands alone for grouping.

**trials** — one row per trial: the design, the outcome (`success`, `timeout`,
`false_start`, `wall_steps`), and per-target events — entry sample, acquisition sample,
reaction time, velocity and position at entry and at acquisition, peak velocity.

The split is deliberate: derived per-target measures belong to the trial, not the sample,
and repeating forty of them down every row triples the file for nothing. Merge on `TN`
when you want them.

Files land in `data/` as
`<experiment>_<agent>_<YYYYmmdd-HHMMSS>_<table>.csv`, with a `_meta.json` sidecar carrying
the full config, the protocol (noiseless/deterministic/seed), and the column list. Both
tables are flat, hence CSV; the GRU hidden state is `(sample, layer, trial, unit)` and is
not, so `--save-hidden` writes it as a pickle.

### Conventions that decide what the numbers mean

- `x`, `v`, `u` are signed in world coordinates — positive is rightward.
- `*_out` columns are **folded by reach direction** (multiplied by `sign(T1)`), so positive
  means "still travelling onward toward the next target". Fold before pooling across
  directions; the policy is not mirror-symmetric (§5.1).
- `d_next` is folded the same way: positive means T2 lies beyond T1.
- Trials are deterministic and noiseless by default, so two trials with identical settings
  are bit-identical and any difference between conditions is causal. `--noisy` and
  `--stochastic` relax that; `--repeats N` is only meaningful once one of them is on.
- Samples after a trial ends are dropped (the environment autoresets in place, so they
  belong to the next trial). `--include-dead` keeps them, flagged — debugging only.

The tables are tested against the raw rollout sample-for-sample, including that the
preview-off control comes back exactly zero through a CSV round-trip
(`1D/tests/test_dataset.py`).

---

## 8. Reproduction

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r 1D/requirements.txt
.venv/bin/python -m pytest 1D/tests -q
```

A single run and its analysis:

```bash
.venv/bin/python 1D/scripts/train.py --name baseline --timesteps 6000000 --no-video
```

```bash
.venv/bin/python 1D/scripts/analyze.py --run 1D/runs/baseline_s1
```

The full grid. `--parallel` should be roughly the number of **physical** cores, with
`--threads 1` — see §4 for why parallelism belongs across jobs rather than inside them:

```bash
.venv/bin/python 1D/scripts/run_ablations.py --timesteps 6000000 --parallel 12 --threads 1 --no-video --seed 1
```

Repeat for seeds 2–5. This is not optional: everything reported here is single-seed, the
outcome is bimodal (0.97 vs 0.21 success), and PPO seed variance is normally larger than
the within-condition regression error quoted alongside each CI.

Re-run the analysis on existing checkpoints without retraining:

```bash
.venv/bin/python 1D/scripts/run_ablations.py --analyze-only --seed 1
```

Render rollout GIFs from a checkpoint afterwards, rather than during training:

```bash
.venv/bin/python 1D/scripts/make_video.py --run 1D/runs/baseline_s1
```

Figures land in `1D/runs/<name>_s<seed>/analysis/`, numbers in `results.json`, and the
cross-condition summary in `1D/runs/ablation_summary.json` and `fig6_ablations.png`.

**Reading the output.** Four checks, in order, before any number is quoted:

1. `mean_n_targets` in the training log climbs 1.00 → 2.30 over the first 1M steps and
   then holds — the curriculum ran.
2. `wall_rate` ends near 0.00 — nothing is parked at the workspace boundary.
3. `episodic_return_task`, not `episodic_return`, is the progress signal.
4. `analyze.py` prints success split by sequence length. An arm near 1.00 on one target
   and 0.00 on two learned a single reach; its CI means nothing whatever the value.
   `run_ablations.py` prints competent arms above a line and the rest below it.

---



## 9. What carries over to 2D and MotorNet

The design generalises with one substitution: **the co-articulation readout changes from
signed pass-through velocity to the pass-through velocity vector**, of which path
curvature is a downstream consequence.

Carries over unchanged:

- the Case-A/Case-B argument, and `dwell_inter` as its control knob — in any dimension,
requiring rest at each target factorises the problem;
- preview as a rolling window, trained mixed, giving a within-agent causal control;
- the random-delay hold with an abort rule;
- reward = acquisition + effort + time, with no trajectory terms;
- potential-based shaping as the only shaping allowed;
- `jump_future` as the sharp test of ongoing planning;
- the output-null analysis — with muscles, `w` becomes the muscle-activation readout and
the potent space becomes multi-dimensional, which makes the test *more* informative
rather than less.

Also carried over, and load-bearing: the **sequence-length curriculum** and the
**competence gate**. In 2D with muscles, exploration is strictly harder than here, so a
condition meeting full-length sequences from step zero will fail more often, not less.

Seven methodological lessons from this stage, all of which will bite harder in 2D. The
first three are the expensive ones — each cost a campaign.

1. **Train to a competence floor, not a step count.** Fixed-budget ablation grids
   confound "removed the coupling" with "broke the agent". Seven of nine arms here were
   single-reach policies whose indices described movements that were never part of a
   sequence. Compare conditions at matched success rate, and treat matched *low* success
   as matched incompetence rather than a matched comparison.
2. **Diagnose failures, don't just count them.** The success rate alone could not
   distinguish "harder task" from "broken agent". Success split by sequence length,
   failure attribution (timeout vs false start vs boundary), and probe attrition all
   could — instantly, and none of them was expensive to compute.
3. **Audit the environment for flat regions, not just for exploits.** The workspace
   boundary was never worth more than solving the task, so no reward-hacking check would
   have flagged it. It was a *plateau*, and it absorbed most of the grid. In 2D, joint
   limits and muscle-length limits are the direct analogues, and there will be more of
   them.
4. **Prefer within-agent manipulations.** The preview-off control is the strongest result
   in this document precisely because weights and biomechanics are held fixed, and because
   it does not require two agents to be trained equally well — the assumption that broke
   everything else. The between-agent version of the same ablation was unmeasurable. In
   2D, where biomechanics can produce curvature on its own, this stops being a nicety.
5. **Check that the readout is well-defined under the ablation.** Timing pass-through at
   acquisition rather than at first entry would have produced a spurious confirmation of
   the separability prediction. Note this is only half-solved (§6.5): entry velocity is not
   what the separability argument constrains either.
6. **Log the objective, not the objective plus shaping.** A shaped return can rise
   monotonically while task success sits at zero, and here it did — the worst run in the
   grid had the best-looking return curve.
7. **Reference perturbation measures to a pre-perturbation baseline.** The agent's
   holding posture is already target-dependent, so divergence measured against zero
   reports movement-onset effects that are actually postural.

Needs rethinking in 2D:

- **Redundancy.** With a redundant arm, the null space of the kinematics is large, so
"the state handed to the next segment" includes posture, not just endpoint velocity.
Expect lookahead to show up in joint configuration at pass-through as well.
- **Effort cost.** `u²` on muscle activations is not the same as `u²` on endpoint force;
the effort landscape becomes anisotropic, and some apparent co-articulation will be
biomechanical rather than planning-driven. The preview-off control becomes essential
rather than merely reassuring, since it holds biomechanics fixed.
- **Exploration.** Sparse acquisition reward in 2D with muscles is much harder. The
sequence-length curriculum already exists (§3) and carries over directly, but a curriculum
over *reach distance* or muscle-space exploration noise will probably be needed on top of
it — the 1D version only had to get the agent past the one-reach-to-two-reaches barrier.
- **More flat regions to audit.** The workspace-boundary trap (§6.6) has direct analogues
in joint limits and muscle-length limits, and there will be more of them. Each one is a
place where the return landscape goes flat and a struggling policy can come to rest.
Check contact/limit rates as a routine diagnostic, the way `wall_rate` is checked here.

---



## 10. Limitations

Ordered by how much they should change what you conclude.

0. **Every run in `1D/runs/` predates the two environment fixes and is stale.** The
   inelastic boundary and the sequence-length curriculum both change training dynamics for
   every condition, baseline included. Nothing in §6 has been reproduced under the current
   code. The qualitative baseline findings should survive — that arm was demonstrably
   healthy, and never touched the boundary — but the numbers must be re-derived before any
   of them is quoted. `plan.md` has the command sequence.
1. **The competence confound dominated the ablation grid.** Seven of nine conditions
   scored ~100% on one-target trials and ~0% on two-target trials — single-reach policies,
   for which the CI is not a measurement of anything. The sharpest demonstration is the
   separability ablation, whose CI went from **−0.642 at 3M steps to +0.253 at 6M** as
   success rose from 0.17 to 0.32: a sign reversal driven purely by training budget. Had
   only the 3M run been done, it would have read as a clean confirmation of §2. This is now
   gated in code (§5.0), but the gate only prevents the misreading — it does not supply the
   missing comparisons, which need the re-run. Of the original grid, exactly one
   between-agent comparison survives: `no_effort` against `baseline`.
2. **Single seed per condition.** The reported standard errors are within-condition
  regression errors across the T2 grid; they say nothing about seed-to-seed
   variability, which for PPO is usually the larger term — and here the outcome
   distribution is visibly bimodal (0.97 vs ~0.21 success), which is precisely the regime
   where a single seed is uninformative. `--seed` is a one-line change and 5 seeds per
   condition should be run before any of this is treated as a finding.
3. **RT sits on the floor.** With `sensory_delay = 0`, one step is 20 ms and measured
  RTs are 19–28 ms, so the whole delay effect spans under half a simulation step. The
   direction is right and the trend is monotonic, but the measurement has almost no
   dynamic range. Setting `plant.sensory_delay > 0` should be done before anything is
   claimed about preparation timing.
4. ~~`no_preview` is not measurable as a between-agent ablation.~~ **Resolved: the arm was
   removed from the grid on 2026-08-16** (§6.5, C6). It could not be probed with T2 visible
   (out of distribution, 0/36 T1 acquisitions) and was tautological with T2 hidden (CI
   exactly zero for any agent). The within-agent control in §6.2 is the sound version and
   is strictly stronger. The stale run directory `1D/runs/no_preview_s1` can be deleted.
5. **The separability ablation may not be testable with the current readout.** §2
   constrains velocity at *departure* from a target, which a dwell forces to ~0; the metric
   times pass-through at *entry*, which a dwell leaves free (bounded by ~1.26, not 0). So
   neither a positive nor a null CI-at-entry speaks to the theory. Choosing the right
   readout is an open design decision, described in §6.5, and it should be settled
   deliberately rather than by whichever number the re-run produces.
6. **γ = 0.90 is a confound, not an ablation.** The discounted return gap between doing the
   task and doing nothing collapses from 23.71 at γ = 0.995 to 0.05 at γ = 0.90 — a ~470×
   reduction in the entire learnable signal. The arm cannot separate "removed the lookahead
   incentive" from "removed the task", and no amount of training fixes that. Report it as a
   demonstration of the confound.
7. **Deterministic, noiseless probes make decoding analyses vacuous.** Every property of
  the upcoming movement is a deterministic function of an observation that already
   contains the targets, so decoding R² ≈ 1 carries no information. Only the geometric
   analysis (output-null) is informative under this protocol. The behavioural results
   themselves do survive `--noisy` (§6.2), but making *decoding* meaningful needs a
   stochastic policy as well as a noisy plant.
8. **The plant is a point mass with an activation lag, not a limb.** Nothing here speaks
  to whether the same couplings dominate once biomechanics are realistic.

