# 1D sequential reaching — task, design, and measurement

**Status:** implemented. Code in [`1D/`](../1D). This document is the standing reference
for *how the 1D stage works*: the task, the argument that motivates its shape, the knobs,
the maths behind the reward and the discount, and the definition of every measurement.

**It deliberately contains no results.** Numbers appear here only where a design choice
cannot be justified without one. Everything measured lives in
[`1D_results.md`](1D_results.md).

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
design; §11 says what transfers.

---

## 2. Why lookahead has to be made possible before it can be made to emerge

This determines the whole design and it is the easiest thing to get wrong.

Write the sequence as targets `T₁…T_K` acquired in order, with plant state
`s = (x, v, a)` (position, velocity, activation). Total cost is additive over time, and
the trial breaks into segments at acquisition events.

**Case A — acquisition requires coming to rest.** If taking `T_k` requires dwelling there
long enough that `v ≈ 0`, the state entering segment `k+1` is `(x = T_k, v = 0, a = 0)`
*regardless of how segment k was executed*. The optimal cost separates:

```
J*  =  Σ_k  J*_k(T_{k-1} → T_k)
```

Every term depends on exactly two adjacent targets. Therefore

```
∂ (segment-k trajectory) / ∂ T_{k+1}  =  0
```

**identically.** Under this task structure no reward function, network, or training
budget can produce a horizon effect. It is ruled out by the structure of the problem, not
by the agent's competence. This is, I think, the main reason sequential-reaching models
often fail to show co-articulation: the task is written so that each reach ends at rest,
which factorises it.

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

- **the information** — `T_{k+1}` must be observable during segment k (ablation axis C6);
- **the temporal reach** — the discount `γ` must be long enough that `J*_{k+1}` is
  actually represented in the value function while segment k is being chosen (axis C7).

Both are hard to ablate without also ablating the agent, which is why within-agent
manipulations are preferred wherever they are available. §7.3 gives the arithmetic for γ.

### 1D is not a degenerate case of 2D

In 2D, lookahead has a spatial signature — the path curves. In 1D the path is a straight
line by construction, so **every trace of lookahead is in the velocity profile**: how fast
the mass is moving when it crosses T1, when it starts, when it peaks. This is not a
weakness of the 1D case; it removes the possibility of confusing a spatial artefact for a
plan. But it does mean that any measurement built on *position* will be far less sensitive
here than the corresponding one in 2D (see §8.2).

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

### 3.1 Plant

Semi-implicit Euler, `dt = 20 ms`:

```
a  ← a + (dt/τ)·(clip(u, ±u_max) − a)          activation lag, τ = 50 ms
a_eff = a·(1 + σ·ε),  ε ~ N(0,1)               signal-dependent noise, σ = 0.05
v  ← v + dt·(a_eff − b·v)/m                    m = 1, b = 0.2
x  ← x + dt·v
if |x| > x_limit:  x ← ±x_limit,  v ← 0        inelastic boundary, x_limit = 1.5
```

`u_max = 8`. The activation lag matters: force cannot change instantaneously, so momentum
is a resource that has to be committed in advance. An optional sensory delay (default 0)
feeds the policy a stale `(x, v)`.

**The boundary is inelastic and charged for** (`−w_wall` per step of contact). An earlier
version clipped position while leaving velocity untouched and unpenalised, making the
limit a free resting place — and better than free, since potential-based shaping pays
`(1 − γ)·c·|x − T|` per step to an agent that holds still, which is `0.024/step` out at
the boundary against a time cost of `0.02/step`. That created a flat region of the return
landscape with no local gradient out of it. Because targets only ever appear within
`±target_range = 0.9`, a policy that does the task never makes contact and never pays
`w_wall`: the term removes a hiding place, it does not shape the solution.

### 3.2 Acquisition

`|x − T_k| ≤ tol` (`tol = 0.05`) held for `dwell_k` consecutive steps, with
`dwell_inter = 1` (pass-through) and `dwell_final = 15` (0.3 s). The dwell counter resets
to zero the moment the mass leaves the band. The intermediate dwell is the Case-A/Case-B
switch from §2 and is ablation axis C1.

### 3.3 Preview

A rolling visibility window: slot `i` is visible iff `cur ≤ i ≤ cur + H − 1`. `H = 1`
means only the current target is visible; `H = K` means the whole sequence is visible from
trial onset. Training mixes `H` across trials (`full_preview_prob = 0.5`), which is what
makes preview-on vs preview-off a **within-agent** contrast at test time — the single most
important design decision in the measurement, because it removes the need to train two
agents to equal competence in order to compare them.

### 3.4 Sequence length and the curriculum over it

Each trial draws 1–3 targets with weights `n_target_weights = (0.2, 0.3, 0.5)`. Meeting
that mixture from step zero is a trap: an untrained agent cannot complete a three-target
sequence by chance, but it can stumble into a single reach, so early training reinforces
single reaches, exploration noise shrinks, and the policy never discovers that chaining
pays more.

Sampling weights are therefore annealed linearly from
`curriculum_start_weights = (1, 0, 0)` to `n_target_weights` over the first
`curriculum_steps = 1,000,000` environment steps, then held:

```
w(t) = (1 − f)·(1, 0, 0)  +  f·(0.2, 0.3, 0.5),      f = min(1, t / curriculum_steps)
```

Three properties make it safe to leave on by default, which it is:

- **It does not change the task** — only the order in which trials are shown. The final
  mixture, and hence everything evaluated, is identical. `curriculum_steps=0` disables it.
- **It applies identically to every arm**, which is what an ablation grid requires.
- **Measurement cannot inherit it.** Probes construct a fresh environment, which always
  starts at the final mixture; only the training loop advances the schedule.

`mean_n_targets` is logged so the schedule is visible in the training curve: it climbs
from 1.00 to 2.30 and then sits flat. A success-rate dip during that climb is the
curriculum getting harder, not the agent degrading.

### 3.5 Reward — the entire task specification

| term | value |
| --- | --- |
| intermediate acquisition | `+10` |
| final acquisition | `+20`, ends trial |
| effort | `−0.5·(u/u_max)²·dt` |
| control rate | `−0.05·((u − u_prev)/u_max)²` |
| time | `−1.0·dt` while targets remain |
| hold violation before go | `−20·(x−x₀)² − 2·v²` |
| false start (`|x−x₀| > 0.15` pre-go) | `−5`, aborts trial |
| workspace boundary contact | `−0.1` per step at `|x| = x_limit` |
| optional shaping | see §7 |

**No term references a trajectory, a velocity profile, a via point, a reaction time, or a
preparatory state.** That is the whole point: any lookahead observed has to have come from
optimality, not from supervision.

### 3.6 Time budget (`task.max_steps`)

`max_steps = 600` (12 s). This is not a cosmetic cap and the value is load-bearing.

The cap never binds for a *converged* policy — baseline finishes a full three-target
sequence in ~110 steps. It binds during **learning**, on the longer, clumsier trajectories
that exploration has to pass through before the solution is found. A truncated trajectory
never reaches the terminal reward, so the agent never learns what it was heading towards.

The previous value of 200 sat right at the edge of feasibility. The consequence was not a
mild slowdown but a **bimodal outcome**: a seed either broke through or never did, and the
same configuration produced both. Any ablation measured against a single baseline seed was
therefore partly measuring which side of that coin the baseline landed on.

The design rule that follows, and which should be applied to any new arm: **the budget
must clear the mid-training trajectory-length distribution, not the converged one.** In
practice that has meant a cap of roughly 3.5× the converged mean episode length. Arms that
lengthen episodes by construction need their own larger value — `stop_at_targets` carries
`max_steps = 700` for exactly this reason, and its entry in `run_ablations.py` records the
arithmetic. Measurements supporting all of this are in `1D_results.md` §2.

### 3.7 Observation

`[x̂, v̂, a_prev, go_flag]` plus, per target slot,
`[visible, pending, is_current, (T_i − x̂)·visible]`. Targets are egocentric; hidden
targets read zero. No planning features, no precomputed plan, no time-remaining signal.

### 3.8 Perturbations

With probability `p_jump`, one target is displaced mid-movement: either the one currently
being reached to (`jump_current`, the classic double-step) or a *later* one
(`jump_future`). The second is the sharper probe — if only T2 moves and the ongoing reach
to T1 changes, the agent was demonstrably planning through T2.

---

## 4. Agent and algorithm

PPO, ported from CleanRL (`ppo_continuous_action.py`, plus the sequence-batched recurrent
update from `ppo_atari_lstm.py`): GAE(λ), clipped surrogate, clipped value loss, entropy
bonus, orthogonal init with `std=0.01` on the mean head, advantage normalisation, LR
annealing, gradient clipping.

Policy: GRU(128) over a 2×128 tanh trunk, Gaussian head with state-independent log-std.
The recurrent state is also the object of the neural analysis.

Defaults: 32 envs × 256 steps, `γ = 0.995`, `λ = 0.95`, lr 3e-4, 4 epochs, 4 minibatches,
clip 0.2, entropy coef 0.003.

Two deliberate departures from the reference scripts, both corrections:

- **Own batched environment** instead of `gym.vector.SyncVectorEnv`. Gymnasium 1.0 changed
  vector autoreset to next-step semantics, which silently breaks CleanRL's bootstrapping;
  owning the vectorisation avoids the trap and is far faster on CPU. The env uses same-step
  autoreset — the semantics CleanRL's GAE loop assumes — and additionally reports the
  pre-reset terminal state so probes can record it.
- **Correct time-limit bootstrapping.** Truncated transitions are bootstrapped with
  `γ·V(final_obs)`. The stock CleanRL scripts treat truncation as termination, which biases
  the value function; here a failed trial *is* a timeout, so that case dominates early
  training.

### 4.1 Cost and how to run the grid

The workload is **launch-latency bound, not compute bound**: ~90% of wall time is the PPO
update, which is a Python loop calling `nn.GRU` one timestep at a time. Four threads is
*slower* than one because the matmuls are too small to parallelise, and a GPU would make it
worse without first replacing the per-timestep loop with a fused sequence call.

**Run many single-threaded jobs across cores** (`--parallel N --threads 1`). Rendering
during training costs ~2.8× throughput; use `--no-video` for grid runs and render from the
checkpoint afterwards.

One scheduling consequence discovered the hard way: **do not interleave training and
analysis in the same worker slot.** `analyze.py` on a run with any failing trials costs
minutes (§8.5), so a slot occupied by analysis starves training. Train everything first,
then analyse everything.

The training log carries three fields specifically for catching failure modes:
`episodic_return_task` (return excluding shaping — watch this, not `episodic_return`),
`wall_rate` (fraction of episodes touching the boundary; should be ~0), and
`mean_n_targets` (curriculum position).

---

## 5. What the agent maximises

The agent maximises the **discounted return**

```
G_t  =  r_t  +  γ·r_{t+1}  +  γ²·r_{t+2}  +  …  =  Σ_{k≥0} γ^k · r_{t+k}
```

γ (`ppo.gamma`, default 0.995) weights a reward arriving k steps from now by `γ^k`. It
appears in four places, and they are all the same number:

1. the return above — the objective;
2. the critic's bootstrap target, `V(s) ← r + γV(s′)`;
3. GAE, the advantage estimator;
4. the shaping term (§7) — which *borrows* it, and must, or the algebra fails.

γ is therefore **not a shaping parameter**. It is the agent's discount, and shaping is
merely written in terms of it.

### 5.1 Reading a γ value

Raw γ values are not intuitive. Two conversions:

- **effective horizon** `= 1/(1−γ)` steps — roughly the number of steps over which reward
  meaningfully accumulates (from `Σ γ^k = 1/(1−γ)`; if γ is read as the probability the
  episode survives another step, `1/(1−γ)` is the expected remaining lifetime);
- **half-life** `= ln(0.5)/ln(γ)` steps — how far ahead a reward must be to be worth half
  its face value.

At `dt = 0.02 s`, 50 steps is one second:

| γ | horizon | in seconds | half-life | in seconds |
| --- | --- | --- | --- | --- |
| 0.995 | 200 steps | 4.00 s | 138 steps | 2.77 s |
| 0.97 | 33 steps | 0.67 s | 23 steps | 0.46 s |
| 0.90 | 10 steps | 0.20 s | 7 steps | 0.13 s |

### 5.2 Why γ decides whether lookahead is possible

§2 showed co-articulation is forced by optimality only when
`∂J*_{k+1}/∂v_exit` survives in the objective. `J*_{k+1}` is the cost-to-go of the *next*
segment, which is where `T_{k+1}` enters — and it is discounted by `γ^Δ`, where Δ is the
number of steps until that segment pays off.

A worked example from a real trial (go cue at step 25, T1 acquired at step 42, T2 acquired
at step 123, total effort cost of the whole trial `0.72`). The agent chooses its approach
to T1 around step 30, which is Δ = 93 steps before T2's `+20` arrives:

| γ | γ⁹³ | value of T2's +20 at the moment of choosing | vs the trial's whole effort budget |
| --- | --- | --- | --- |
| 0.995 | 0.6274 | **12.55** | 17× |
| 0.97 | 0.0589 | **1.18** | 1.6× |
| 0.90 | 0.0000555 | **0.0011** | 1/650th |

At γ = 0.995 the future payoff dominates the cost of shaping the approach, so the
optimiser buys it. At γ = 0.90 it is rounding error and there is nothing to buy.

**γ = 0.90 is not a weaker version of the same manipulation.** Applying the same
arithmetic to T1's own `+10`, 12 steps away, gives 9.42 / 6.94 / 2.82 at γ = 0.995 / 0.97 /
0.90. At γ = 0.90 the agent's entire visible payoff has shrunk toward the cost of trying,
so the task itself degrades. γ = 0.97 is the informative value precisely because one reach
(17–50 steps) fits inside its 33-step horizon while the next one does not.

---

## 6. The ablation axes

An **arm** is one condition in the grid: a name, a set of dotted config overrides, and a
statement of what it tests. Arms are registered in `ABLATIONS` at the top of
`run_ablations.py`; anything not registered there is an exploratory variant that will not
appear in the summary table or figures.

| axis | knob | what it removes |
| --- | --- | --- |
| **C1** separability | `task.dwell_inter` | makes the problem Case A — velocity at T1 is no longer free |
| **C2** effort cost | `reward.w_effort`, `reward.w_rate` | removes the price of arrival velocity |
| **C3** time cost | `reward.w_time` | removes the pressure to chain efficiently |
| **C6** information | preview `H` | removes T2 from the observation |
| **C7** discount horizon | `ppo.gamma` | removes the temporal reach to *care* about T2 |

C6 and C7 are the two requirements identified in §2; everything else modulates *how much*
arrival velocity is worth optimising.

**C6 cannot be tested between agents.** Training a second agent with `H = 1` produces a
policy that is out of distribution the moment a second target appears, so it cannot be
probed in the condition that matters. It *can* be tested within a single agent by hiding
T2 at probe time, which holds weights and competence fixed, and that is the form used
(§8.2). There is no between-agent `no_preview` arm.

---

## 7. Reward shaping

### 7.1 The term

```
Φ(s) = −c · |x − T_current(s)| · 1[ t ≥ t_go  ∧  cur < n_targets ]        c = 2.0
F_t  = γ · Φ(s_{t+1}) − Φ(s_t)
r_t  = r_task + F_t
```

Φ is a **potential function** — one number per state, scoring how good the state looks. It
is zero at the target, more negative further away, and identically zero before the go cue
and after the last target. It depends only on `x` and the target currently being reached
for: it contains no information about T2.

`reward.shaping` selects `"pbrs"` (default), `"none"`, or `"greedy"` (§7.4).

### 7.2 Why the difference, and why γ is inside it

The name is from behaviourist shaping — reinforcing successive approximations. The classic
hazard is that a naive shaping bonus **changes the problem**: rewarding `+Φ(s)` every step
pays the agent for being near the target, which it can farm forever without proceeding, and
which makes "stay close to the current target" literally part of the objective — a
statement about trajectory shape, and therefore implicit supervision of the effect under
study.

A potential *difference* cannot do that. Summing the shaping terms over a trajectory with
the agent's own discount:

```
Σ_t γ^t · F_t  =  Σ_t γ^t [ γΦ(s_{t+1}) − Φ(s_t) ]
                =  Σ_t [ γ^{t+1}Φ(s_{t+1}) − γ^t Φ(s_t) ]
```

Writing `a_t = γ^t Φ(s_t)`, each bracket is `a_{t+1} − a_t`, so the sum **telescopes**:

```
Σ_t γ^t F_t  =  γ^T Φ(s_T) − Φ(s_0)
```

Only the endpoints survive; the path between them is irrelevant. This is the Ng, Harada &
Russell (1999) result: `V'^π(s) = V^π(s) − Φ(s)` and `Q'^π(s,a) = Q^π(s,a) − Φ(s)`, and
since Φ does not depend on the action, the greedy policy is unchanged. The form is not just
sufficient for policy invariance but necessary.

**The γ inside the formula is there to make that cancellation exact.** With `Φ(s′) − Φ(s)`
and no γ, the discount exponents do not line up, the interior terms fail to cancel, and a
path-dependent residual survives.

**Two boundary conditions in this implementation make the result stronger than generic
PBRS.** `Φ(s_0) = 0`, because the pre-go gate is always active at t = 0 (the delay is
0.25–1.00 s). And `Φ(s_T) = 0` at termination, enforced explicitly in `env.step`. So for
*every* episode that terminates:

```
Σ_t γ^t F_t  =  0 − 0  =  0     exactly
```

The total discounted shaping along any terminating trajectory is identically zero — the
same total for every possible path, so no preference over trajectory shapes can be
introduced. All the term does is redistribute reward in time to make credit assignment
easier.

### 7.3 Two subtleties worth knowing

**The acquisition impulse.** `cur` increments *before* `Φ(s′)` is computed, so the
acquisition step carries `F ≈ −γc|x − T_{k+1}|` — a negative impulse that scales with the
distance to the *next* target, and even has a small gradient favouring arrival on the side
of the tolerance ball nearest T2. It is repaid exactly over the following segment. Any
single-step "incentive" visible in a potential-based scheme is an artefact of reading one
term of a sum that cancels.

**Loitering income.** In a static state `F = (γ−1)Φ = (1−γ)c|x−T| ≥ 0` — up to `0.024`/step
against a time cost of `0.02`/step. Over a terminating episode it still sums to zero, but
it inflates *logged* returns (hence `episodic_return_task`), and at γ = 0.90 the factor
`(1−γ)` is 20× larger. **This is why the matched-γ series arms set `reward.shaping="none"`:**
a γ sweep with shaping on would confound the discount with the shaping income it controls.

**Truncation.** `Φ(s_T)` is zeroed on termination but not on truncation, so a timeout
leaves a residual `γ^T Φ(s_T) ≤ 0`. It is consistent under PPO's bootstrap of truncation
with `γV(s_T)`, since the critic learns `V′ = V − Φ`, but it is a place where the
exact-zero result does not hold.

### 7.4 The `greedy` option

`reward.shaping = "greedy"` adds the naive alternative as a *direct* per-step cost:

```
r_t  −=  c · |x − T_current| · dt        (post-go, while targets remain)
```

scaled by `dt` like the effort and time costs. This is **not** policy-invariant — it is
offered so the confound can be measured rather than asserted. It is an experimental
condition, never a default.

---

## 8. Measurements

All post hoc, none of them in any loss.

### 8.1 The gates — which come first

Every index below describes *the shape of a movement the agent is making*. If it is not
making the movement, the index is not measuring the thing it is named after. This is
enforced in code rather than left to the reader.

**Competence floor.** `analyze.py` measures competence before anything else: success rate
against a floor of 90% sequence completion (`--competence-floor`; 0 disables), success
**split by sequence length**, and failure attribution (false start / timeout / boundary).
The split is the diagnostic that matters — an agent at ~1.00 on one-target trials and
~0.00 on two-target trials has learned a single reach, and there is no second movement for
a lookahead index to describe. Below the floor, indices are still computed and written to
`results.json` as diagnostics, but printed flagged, `ci_interpretable` is set false, and
`fig6_ablations.png` draws those arms hatched.

**Budget guard.** The floor says *that* a run failed; this says *why*. If timeouts account
for more than half of all failures, the run is flagged `BUDGET-LIMITED` and should be
re-run with a larger `max_steps` rather than interpreted — it is a setup fault, not an
ablation effect. This distinction is the single most expensive thing this project has
learned (see `1D_results.md` §2).

**Probe attrition.** Checked independently. The CI probe fixes T1 and sweeps T2, so all
preview-on trials should contribute; a short count means the policy could not reach T1 on
the rest and the slope is fit to a biased subset — including, in the worst observed case,
a subset entirely within one reach direction.

### 8.2 Co-articulation index (CI), and what it does not tell you

Fix T1, sweep T2 over a grid, record the signed velocity at the moment the mass **first
enters** T1's tolerance band. Fold by reach direction so both pool:

```
v_out   = v(at first entry into T1's band) · sign(T1)
d_next  = (T2 − T1) · sign(T1)                 > 0 means T2 continues onward
CI      = d v_out / d d_next                   (OLS slope, fit within direction)
```

`CI > 0` = passing through T1 with momentum toward T2.

**Entry, not acquisition.** The two coincide under `dwell_inter = 1`. They diverge once a
dwell is required: acquisition is then registered only after the dwell completes, when
velocity is ~0 *by construction* — which would manufacture a confirmation of the
separability prediction in exactly the condition meant to test it.

**Fit within reach direction.** The learned policy is not mirror-symmetric — leftward and
rightward reaches leave T1 at different mean speeds — and the two directions do not span
the same range of `d_next`. Regressing on the pooled cloud picks up a between-direction
offset unrelated to T2. `d_next` and `v_out` are therefore demeaned within direction before
the fit (a fixed-effects estimator).

**The control.** The identical grid is re-run with T2 hidden (`H = 1`). Same weights, same
physics, same T1, same T2 values; the only difference is whether T2 is in the observation.
Because the policy is deterministic and probes are noiseless, the hidden condition yields
**bit-identical** trajectories across every T2 value, so its CI is exactly 0. That is a
hard null, not an estimated one.

**Three things CI cannot do**, all of which have bitten:

1. **A signed slope assumes monotonicity.** If `v_out` depends on `|d_next|` rather than
   `d_next`, a straight line through the cloud returns a slope whose sign is set by which
   arm of the relation dominates the sample.
2. **The probe samples `d_next` asymmetrically.** T1 sits at ±0.45 while T2 sweeps ±0.9, so
   `d_next` spans `[−1.35, +0.45]` — three quarters of every slope is measured on the
   reversing side. `ci_balanced` refits on the widest window symmetric about zero, at no
   extra rollout cost; a large gap between `ci` and `ci_balanced` means the relation is not
   linear and the signed slope should not be read.
3. **The two reach directions can disagree.** `by_group` reports the per-direction slopes
   and `dir_agree` flags the disagreement. **A CI whose directions disagree is not a signed
   effect**, regardless of its p-value.

Always report `ci_r` alongside `ci`. A slope explaining 96% of the variance and one
explaining 15% are not the same kind of result and a p-value does not distinguish them.

### 8.3 T2 dependence — the sign-agnostic companion

CI asks *which way* the reach bends. `t2_dependence` asks the weaker, unambiguous question:
**does the trajectory to T1 depend on T2 at all?**

Within one reach direction the probe trials differ only in T2, so at each timestep the
spread (SD) across them *is* T2's influence. Measured from the go cue to the earliest first
entry into T1's band, so it describes the reach to T1 and never the movement toward T2.
Reported as RMS and peak, for both position and velocity, with position also in tolerance
units.

Its virtue is the null: with T2 hidden the observation is byte-identical across trials, so
the spread is **exactly zero** (measured ~1e-16, i.e. floating-point). Any non-zero value
is real influence or a broken pipeline, with no inference in between. Being a magnitude, it
is immune to everything in §8.2 — no monotonicity assumption, no direction-consistency
assumption, no linear fit.

**Report the velocity spread, not the position spread.** Per §2.1, in 1D the path is a line
and lookahead lives in the velocity profile; position spread is several times less
sensitive.

CI and `t2_dependence` can disagree, and when they do the disagreement is the finding: a
near-zero CI with a substantial `t2_v_spread` means the agent uses T2 but not in the
co-articulation-shaped way, which is a different claim from "the agent ignores T2".

### 8.4 The rest

**Reaction time.** Go cue to movement onset (5% of that trial's peak speed), as a function
of hold duration. A decrease with longer holds is the classic monkey/human signature and is
nowhere trained for.

**Jump response.** Correction latency after a target displacement, and — the informative
one — whether a `jump_future` perturbs the reach to T1 *before* T1 is taken.

**Preparatory geometry.** The mean action is a linear readout `u = w·h` of the GRU state, so
`w` spans a 1-D output-potent direction and its orthogonal complement is output-null.
Comparing across-trial variance of the pre-go state along `w` against a typical random
direction tests whether preparation specifies the movement while avoiding driving the
plant.

> A caution on the decoding numbers: T1 and T2 are *present in the observation* throughout
> the hold, so decoding them from the pre-go state is a pass-through check, not evidence of
> preparation — a low value would indicate a bug. The informative decodes are of upcoming
> *movement parameters* (peak velocity, segment duration), which are not in the input.

### 8.5 Cost of analysis

`probes.rollout` stops early only when *every* trial in the batch has finished. The
competence battery is 512 random trials, so a single timed-out trial keeps the loop running
to `max_steps`. Raising `max_steps` from 200 to 600 therefore made analysis ~3× more
expensive for any run that is not at ~100% success — the runs that are simultaneously the
slowest to analyse and the least informative. The budget guard (§8.1) exists partly to
short-circuit them.

---

## 9. Experiments and data export

`analyze.py` computes a fixed set of indices and writes figures. It is deliberately
opinionated, which makes it the wrong tool for exploratory analysis. Alongside it there is
a general path: **define an experiment, run a frozen agent through it, get dataframes.**

An *experiment* means what it means in the lab — a trial list with a factorial design, run
on a subject that is not learning during it. Weights are loaded from a checkpoint and never
updated.

```bash
.conda/bin/python 1D/scripts/run_experiment.py --list
.conda/bin/python 1D/scripts/run_experiment.py --run 1D/runs/baseline_s1 --exp preview_sweep
```

Experiments live in [`1D/seqreach/experiments.py`](../1D/seqreach/experiments.py), one
function each, in a registry:

| experiment | manipulates |
| --- | --- |
| `preview_sweep` | T1 fixed, T2 swept, crossed with T2 visible/hidden and direction — the co-articulation protocol |
| `delay_sweep` | duration of the pre-go hold — reaction time and preparatory geometry |
| `target_jump` | a target displaced mid-movement, current or future, against matched no-jump controls |
| `random_task` | the training distribution itself — competence and failure profile |
| `sequence_length` | T1 fixed, varying how many targets follow it |

Adding one is a function returning a trial list; any keyword argument it declares becomes
settable as `--param name=value`. `factors={...}` is free-form and becomes columns in the
output, so the independent variables of a design go there.

### 9.1 The two tables

**timeseries** — one row per (trial, sample): `t_ms`, `t_from_go_ms`, `x`, `v`, `a`, `u`,
which target is currently being pursued, and the target positions *as they stand at that
sample* (so jumps are visible). Condition columns are repeated down the trial's rows, so
the table stands alone for grouping.

**trials** — one row per trial: the design, the outcome (`success`, `timeout`,
`false_start`, `wall_steps`), and per-target events — entry sample, acquisition sample,
reaction time, velocity and position at entry and at acquisition, peak velocity.

The split is deliberate: derived per-target measures belong to the trial, not the sample.
Merge on `TN` when both are needed.

Files land in `data/` as `<experiment>_<agent>_<YYYYmmdd-HHMMSS>_<table>.csv` with a
`_meta.json` sidecar carrying the full config, the protocol (noiseless/deterministic/seed),
and the column list. Both tables are flat, hence CSV; the GRU hidden state is
`(sample, layer, trial, unit)` and is not, so `--save-hidden` writes it as a pickle.

### 9.2 Conventions that decide what the numbers mean

- `x`, `v`, `u` are signed in world coordinates — positive is rightward.
- `*_out` columns are **folded by reach direction** (multiplied by `sign(T1)`), so positive
  means "still travelling onward toward the next target". Fold before pooling across
  directions; the policy is not mirror-symmetric (§8.2).
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

## 10. Reproduction

The environment is a conda prefix at the repo root, `.conda/` (gitignored). `python` below
means `.conda/bin/python`.

```bash
conda create -p .conda python=3.11 && .conda/bin/pip install -r 1D/requirements.txt

python 1D/scripts/train.py --name baseline --timesteps 3000000 --wandb
python 1D/scripts/analyze.py --run 1D/runs/baseline_s1
for s in 1 2 3 4 5 6 7 8; do
  python 1D/scripts/run_ablations.py --timesteps 3000000 --parallel 9 --threads 1 --no-video --wandb --seed $s
done
python -m pytest 1D/tests -q
```

Any config field is overridable from the CLI with dotted keys:

```bash
python 1D/scripts/train.py --name gamma097_sparse --set ppo.gamma=0.97 reward.shaping=none
```

**Seeds.** One seed is not a result. The bimodality described in §3.6 means a single run
reports which side of a coin it landed on as much as it reports an effect. Anything
intended for publication should be ≥8 seeds per condition, with the CI reported as a
distribution rather than a point estimate.

---

## 11. What carries over to 2D / MotorNet

The design generalises with one substitution: **the co-articulation readout changes from
signed pass-through velocity to the pass-through velocity vector**, of which path curvature
is a downstream consequence.

Carries over unchanged:

- the Case-A/Case-B argument (§2) and `dwell_inter` as its control knob — in any dimension,
  requiring rest at each target factorises the problem;
- preview as a rolling window, trained mixed, giving a within-agent causal control;
- the random-delay hold with an abort rule;
- reward = acquisition + effort + time, with no trajectory terms;
- potential-based shaping as the only shaping allowed by default;
- `jump_future` as the sharp test of ongoing planning;
- the output-null analysis — with muscles, `w` becomes the muscle-activation readout and the
  potent space becomes multi-dimensional, which makes the test *more* informative;
- the sequence-length curriculum, the competence gate, and the budget guard.

`t2_dependence` (§8.3) transfers directly and is the natural bridge, since it is defined on
trajectories rather than on a scalar velocity at one instant.

### 11.1 Needs rethinking in 2D

- **Redundancy.** With a redundant arm the null space of the kinematics is large, so "the
  state handed to the next segment" includes posture, not just endpoint velocity. Expect
  lookahead in joint configuration at pass-through as well.
- **Effort cost.** `u²` on muscle activations is not `u²` on endpoint force; the effort
  landscape becomes anisotropic and some apparent co-articulation will be biomechanical
  rather than planning-driven. The preview-off control becomes essential rather than merely
  reassuring, since it holds biomechanics fixed.
- **Exploration.** Sparse acquisition reward in 2D with muscles is much harder. The
  sequence-length curriculum carries over, but a curriculum over reach distance or
  exploration noise will probably be needed on top of it.
- **More flat regions to audit.** The workspace-boundary trap has direct analogues in joint
  limits and muscle-length limits, and there will be more of them. Check contact/limit rates
  routinely, the way `wall_rate` is checked here.
- **The time budget will bite harder.** 2D trajectories are longer and the exploratory
  distribution is wider, so §3.6 matters more, not less.

### 11.2 Methodological lessons

Ordered by what they cost. The first four each invalidated a campaign.

1. **Give exploration a time budget, not just the converged policy.** A cap that never binds
   at convergence can still truncate the trajectories learning must pass through, turning
   the outcome bimodal. Then a single seed reports which side of a coin it landed on, and
   every ablation measured against it inherits that. Diagnose with the failure attribution,
   not the success rate.
2. **Train to a competence floor, not a step count.** Fixed-budget grids confound "removed
   the coupling" with "broke the agent". Treat matched *low* success as matched
   incompetence, not a matched comparison.
3. **Diagnose failures, don't just count them.** Success rate alone cannot distinguish
   "harder task" from "broken agent" from "ran out of clock". Success split by sequence
   length, failure attribution, and probe attrition all can — instantly, and none is
   expensive.
4. **Audit the environment for flat regions, not just for exploits.** The workspace boundary
   was never worth more than solving the task, so no reward-hacking check would have flagged
   it. It was a *plateau*, and it absorbed most of a grid.
5. **Prefer within-agent manipulations.** The preview-off control is the strongest result
   available precisely because weights and biomechanics are held fixed, and because it does
   not require two agents to be trained equally well.
6. **Check the readout is well-defined under the ablation.** Timing pass-through at
   acquisition rather than first entry would have manufactured a confirmation of the
   separability prediction. This is only half-solved (§8.2).
7. **Never read a signed index without its shape checks.** A slope fit to a non-monotonic
   relation, or to two reach directions that disagree, has a sign and a p-value and means
   nothing. Report `ci_r`, `ci_balanced`, `dir_agree`, and the sign-agnostic magnitude.
8. **Log the objective, not the objective plus shaping.** A shaped return can rise
   monotonically while task success sits at zero.
9. **Reference perturbation measures to a pre-perturbation baseline.** The holding posture is
   already target-dependent, so divergence measured against zero reports postural effects as
   movement-onset effects.

---

## 12. Limitations

Ordered by how much they should change what you conclude.

1. **1D has no spatial signature.** Everything rests on the velocity profile, which makes
   the measurement more fragile than its 2D equivalent (§2.1). Position-based measures are
   ~8× less sensitive here.
2. **CI is a single scalar at a single instant**, with three failure modes (§8.2): it assumes
   monotonicity, it is fit to an asymmetric `d_next` sample, and its two reach directions can
   disagree. It should never be read without its control, `ci_balanced`, `dir_agree`, `ci_r`,
   and `t2_v_spread`.
3. **The probe design is unbalanced** in `d_next` (27 reversing vs 9 onward trials).
   `ci_balanced` mitigates this at no cost, but a balanced probe would be better and is not
   implemented.
4. **C1 may not be testable with a velocity-based index.** §2 constrains velocity at
   *departure* from a target, which a dwell forces to ~0; the metric times entry, which a
   dwell leaves free. Neither a positive nor a null CI-at-entry speaks directly to the
   theory. Choosing the right readout is an open design decision.
5. **γ = 0.90 is a confound, not an ablation.** It shrinks the entire learnable signal, not
   just the lookahead incentive (§5.2), and no amount of training fixes that. Report it as a
   demonstration of the confound. γ = 0.97 is the usable C7 condition.
6. **RT sits on the floor.** With `sensory_delay = 0`, one step is 20 ms and measured RTs are
   19–28 ms, so the whole delay effect spans under half a simulation step. The direction is
   right and the trend monotonic, but there is almost no dynamic range. Set
   `plant.sensory_delay > 0` before claiming anything about preparation timing.
7. **Deterministic, noiseless probes make decoding analyses vacuous.** Every property of the
   upcoming movement is a deterministic function of an observation that already contains the
   targets, so decoding R² ≈ 1 carries no information. Only the geometric (output-null)
   analysis is informative under this protocol. Behavioural results survive `--noisy`; making
   *decoding* meaningful needs a stochastic policy as well as a noisy plant.
8. **Sequence length is capped at 3**, so "lookahead" here is always one target ahead.
9. **The plant is a point mass with an activation lag, not a limb.** Nothing here speaks to
   whether the same couplings dominate once biomechanics are realistic.
