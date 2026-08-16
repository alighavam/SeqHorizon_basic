This is for temporary execution plans - after plan is executed this will be deleted.

---

# What was wrong, and what I changed (2026-08-16)

## The background in one paragraph

The first ablation grid trained 9 conditions for 3M steps each and reported a
co-articulation index (CI) for every one of them. Only two conditions — `baseline` and
`no_effort` — actually learned the task. The other seven scored **100% on single-target
trials and 0% on two-target trials**: they learned to make one reach and stop. Reading a
"lookahead" number off an agent that never plans a second movement measures nothing, and
in one case it actively misled us — `stop_at_targets` gave CI = −0.642 at 3M steps and
**+0.253 at 6M steps**, a sign reversal caused purely by training budget. The 3M number
looked like a clean confirmation of the theory. It was an artifact.

So the job was not to reinterpret the numbers. It was to make the failing conditions
able to learn, and to make the pipeline refuse to report an index it cannot justify.

---

## Fix 1 — The workspace boundary ("what I meant by *fix the workspace bound*")

### What was happening

The point mass lives on a line and is not allowed past `x = ±1.5` (`plant.x_limit`). The
old code enforced that with one line:

```python
np.clip(self.x, -p.x_limit, p.x_limit, out=self.x)
```

That clamps **position** and does nothing else. Velocity was left untouched, and there
was no cost for being there. Two consequences:

1. The mass could sit at `x = 1.5` with any velocity you like. Nothing pushed it back.
2. Sitting there was **free**, and in fact slightly profitable. The shaping term pays
   `(1 − γ)·c·|x − target|` per step to an agent that holds still — that is `0.024/step`
   out at the boundary, against a time cost of `0.02/step`. Net ≈ 0. Flat.

So there was a large, perfectly flat plateau at the edge of the workspace. A policy that
wandered there got no gradient telling it to come back. Targets only ever appear within
`±0.9`, so this region is 0.6 units beyond anything the task ever asks for — it is pure
dead space, and it was absorbing agents.

**This is not a theory.** I measured how often each trained policy touches the boundary:

| run | trials touching the boundary | steps spent there (of 200) |
|---|---|---|
| baseline | **0%** | 0 |
| no_effort | **0%** | 0 |
| myopic | 100% | 106 |
| myopic_sparse | 93% | 96 |
| gamma097_sparse | 80% | 91 |
| with_jumps | 80% | 89 |
| no_shaping | 80% | 88 |
| no_preview | 47% | 36 |
| stop_at_targets_long | 31% | 4 |
| stop_at_targets | 26% | 19 |

The two conditions that learned never go near it. Every condition that failed lives
there for a large fraction of every episode. This is what your "weird reaching
trajectory behaviour" was: `with_jumps` reaches target 1 correctly, then accelerates off
to `x = 1.5` at velocity 15 and parks for the rest of the trial.

### What I changed

The boundary is now **inelastic** and **charged for**:

- Contact clamps position *and zeroes velocity* — the mass stops, like hitting a wall,
  instead of sliding along it. (`1D/seqreach/env.py`, in `step()`.)
- Contact costs `reward.w_wall = 0.1` per step. Sized deliberately: it has to beat the
  `0.024/step` shaping income the plateau was paying, and `0.1` makes boundary contact
  cost roughly 4× the time penalty. An episode spent parked there now costs about −15,
  which is worse than missing a target (+10) but not catastrophic.

Because targets never appear beyond `±0.9`, **this term is inert for any agent that does
the task** — it only removes a place to hide. There's a test asserting exactly that
(`test_boundary_is_unreachable_for_an_on_task_policy`).

### Why zeroing velocity alone was not enough

I checked. With velocity zeroed and no penalty, the mass just freezes at the boundary
with zero commanded force — still free, still collecting the shaping payment. The
penalty is the part that actually closes the trap; zeroing velocity is what makes the
boundary behave like a physical wall instead of a frictionless rail.

---

## Fix 2 — The sequence-length curriculum ("what I meant by *curriculum on n_targets*")

### What was happening

Every trial samples how many targets it has: 1, 2, or 3, with probabilities
`task.n_target_weights = (0.2, 0.3, 0.5)`. So half of all trials are 3-target sequences,
right from the very first training step.

An untrained agent cannot complete a 3-target sequence by chance — it would have to hit
three positions in order within 4 seconds. What it *can* stumble into is a single reach.
So early training reinforces single reaches, the policy commits to that, its exploration
noise shrinks (I measured entropy collapsing to −0.10 in `stop_at_targets_long`, versus
+0.39 for the healthy baseline), and it never discovers that chaining pays more. That is
exactly the 100%-on-one / 0%-on-two signature in the table above.

`baseline` escapes this, but only just, and any condition that makes the task even
slightly harder does not.

### What I changed

Sequence length is now **annealed**. Training starts sampling only 1-target trials and
linearly mixes in longer sequences over the first `task.curriculum_steps = 1,000,000`
environment steps, reaching the specified `(0.2, 0.3, 0.5)` mixture and holding there for
the remaining 2M steps of a standard 3M run.

Concretely, the sampling weights at step *t* are

```
w(t) = (1 − f)·(1, 0, 0) + f·(0.2, 0.3, 0.5),    f = min(1, t / curriculum_steps)
```

Set `task.curriculum_steps=0` to switch it off entirely.

### Three things worth knowing about it

1. **It does not change the task.** Only the order in which trials are shown during
   training. The final mixture, and therefore everything that gets evaluated, is
   identical. All measurement code builds a fresh environment, which always starts at the
   final mixture — a probe can never accidentally inherit a curriculum stage. There's a
   test pinning that (`test_a_fresh_env_always_samples_the_final_mixture`).
2. **It is on by default, for every condition.** That is deliberate: an ablation grid is
   only interpretable if all arms get identical treatment. It does mean **every existing
   run in `1D/runs/` is now stale** and must be retrained before its numbers are quoted.
3. **You can watch it work.** The training log prints `nT`, the mean number of targets
   currently being sampled. It should climb from 1.00 to 2.30 over the first third of
   training and then sit flat.

---

## Fix 3 — Two things I added because the first two are not verifiable without them

These weren't in my original list, but the fixes above can't be checked without them.

### 3a. A competence gate on the analysis

`analyze.py` now measures *why* a run fails before reporting anything else, and refuses
to present the CI as a measurement when the agent cannot do the task:

```
success rate 0.133  hit fraction 0.292   [1tgt 0.43  2tgt 0.12  3tgt 0.01]
failures: false start 0.00  timeout 0.87  wall contact 0.00 (0 steps/episode)
!! NOT COMPETENT (success 0.13 < floor 0.90).  Indices below are recorded but must not
   be interpreted as measurements of lookahead.
   signature: solves single-target trials, fails every multi-target one -- a single-reach
   policy, so there is no second movement to plan.
!! probe attrition: T1 reached on 18/36 preview-on trials -- the index is fit to a biased subset
CI (T2 visible) = +0.0251 ± 0.0014  p=8.27e-12   [UNINTERPRETABLE: below competence floor]
```

The floor is 90% sequence success (`--competence-floor` to change it; 0 disables). The
**probe attrition** line is a second, independent check: the CI probe fixes target 1 and
sweeps target 2, so all 36 trials should contribute. Fewer means the policy could not
reach target 1 on the rest, and the slope is being fit to a biased subset. That check
would have caught two things in the old grid on its own — `stop_at_targets` and `myopic`
each produced an index fit to 18 trials that were **all in one reach direction**, because
neither policy could reach rightward at all.

`run_ablations.py` now prints competent arms above a line and everything else below it,
labelled "diagnostics only", with a `multi` column (success on multi-target trials only)
that makes the single-reach signature obvious. `fig6_ablations.png` hatches and greys the
non-competent bars and draws the floor.

### 3b. Task return logged separately from shaping return

The training log now has `episodic_return_task` (the actual objective) alongside
`episodic_return` (which includes shaping income), and prints the former.

Why this matters: the old `myopic` arm (γ=0.90) logged **the highest episodic return in
the entire campaign, 46.7, at 0.000 success**. It was collecting shaping payments for
idling. The curve looked like the best run in the grid. Watching the task return instead
makes that impossible to miss.

I also added `wall_rate` to the log — the fraction of episodes touching the boundary. It
should sit at ~0. If it is high, the policy is parked at the wall and Fix 1 is not doing
its job.

---

## Fix 4 — `stop_at_targets` had an uncompensated time budget

Found *after* fixes 1–3 landed, by looking at what still failed.

A 25-step dwell adds 24 steps of **mandatory holding** at each intermediate target — 48
extra steps on a three-target trial — out of a 200-step budget. Simulating a
near-time-optimal controller on three-target trials:

| condition | median steps needed | p90 | slack at p90 |
|---|---|---|---|
| baseline (dwell = 1) | 110 | 129 | **36%** |
| `stop_at_targets` (dwell = 25) | 164 | 186 | **7%** |

A learned policy is never time-optimal, so 7% slack means three-target trials are
effectively impossible while shorter ones stay easy. Measured after fixes 1–3: **0.94
success on two-target trials, 0.00 on three-target**.

So the arm was testing separability *and* a 24% cut to the movement-time budget at the
same time. `max_steps` is now 250 for this condition — exactly the 48 steps the
manipulation costs, giving back 26% slack. Change one thing at a time.

**This corrects something I told you earlier.** I said the time limit was not the problem,
based on a feasibility check run over the mixed distribution (99% feasible). That was true
on average and wrong where it mattered: split by sequence length, three-target trials —
half the distribution — have almost no margin.

---

## Did it work?

Three conditions retrained at 3M steps, seed 1, on this machine. One seed, so a smoke
test rather than the campaign — but enough to check the diagnosis.

| condition | success (old → new) | boundary contact | 2-target success (old → new) |
|---|---|---|---|
| `baseline` | 0.971 → **0.973** | 0.00 → 0.00 | 0.98 → 0.98 |
| `with_jumps` | 0.210 → **0.539** | 0.80 → **0.00** | 0.00 → **0.99** |
| `stop_at_targets` | 0.170 → **0.332** | 0.26 → **0.08** | 0.00 → **0.94** |

The headline is the last column. Both broken arms went from **completing zero two-target
sequences** to completing ~95% of them. They are no longer single-reach policies. Baseline
is unharmed (the curriculum costs early progress and repays it by 3M). Both still fail on
three-target sequences — for `stop_at_targets` that was Fix 4, which is not in these
numbers.

Nothing except baseline clears the 90% competence floor yet, so **no ablation comparison
is licensed from this**. It needs the full grid at 6M × 5 seeds.

One thing to know going in: baseline's CI moved from **+0.549 to +0.854** at identical
competence. Sign, significance and control are unchanged (p = 2×10⁻¹⁷, control exactly 0),
so the finding is robust — but CI *magnitudes* are clearly sensitive to training dynamics.
Compare conditions within a campaign, never across.

---

## What I did *not* change, and why

- **`myopic` (γ=0.90) is still expected to fail, and that is correct behaviour.** I
  measured the discounted return gap between solving the task and running away: 23.71 at
  γ=0.995, but **0.05** at γ=0.90. Shaping is not the problem (it stays policy-invariant
  — solving wins at both). The problem is that a 0.2 s discount horizon shrinks the entire
  learnable signal by ~470×. No curriculum fixes that; the arm cannot separate "removed
  the lookahead incentive" from "removed the task". Treat it as a demonstration of the
  confound, not as an ablation.
- **The CI-at-entry metric is left exactly as it is, by your decision (2026-08-16).** You
  will do that analysis yourself from the exported data. Nothing about the metric changed.
- **The CI-at-entry issue is documented, not fixed.** Under a dwell requirement,
  velocity at *acquisition* is ~0 by construction, so the metric times pass-through at
  first *entry* into the target band instead. But your separability argument constrains
  velocity at **departure**, not entry — so `stop_at_targets` may not test the theory at
  all with this metric family. That needs a decision from you, not a code change. It is
  written up in §6.5 and §9.6 of the doc.
- **`mlp` (memoryless policy) has never been trained.** It is defined in the ablation grid
  but there is no `runs/mlp_s1`. A full grid run will now include it.

---

## Files changed

| file | change |
|---|---|
| `1D/seqreach/config.py` | `reward.w_wall`, `task.curriculum_steps`, `task.curriculum_start_weights` |
| `1D/seqreach/env.py` | inelastic + charged boundary; curriculum hooks; shaping-excluded return; richer episode info |
| `1D/seqreach/ppo.py` | drives the curriculum; logs `episodic_return_task`, `wall_rate`, `mean_n_targets` |
| `1D/seqreach/probes.py` | records false-start/timeout/wall per trial; new `outcome_breakdown` + `COMPETENCE_FLOOR` |
| `1D/seqreach/plots.py` | `plot_ablations` greys/hatches non-competent arms, marks the floor |
| `1D/scripts/analyze.py` | competence gate, failure breakdown, probe-attrition warning |
| `1D/scripts/run_ablations.py` | splits the summary table at the competence floor; `multi` and `wall` columns; `stop_at_targets` time budget compensated |
| `1D/tests/test_env.py` | 9 new tests: boundary behaviour, curriculum schedule, return split |
| `1D/tests/test_metrics.py` | 4 new tests: the competence gate |
| `1D/seqreach/experiments.py` | **new** — registry of named test experiments; the file you edit to add one |
| `1D/seqreach/dataset.py` | **new** — rollout → tidy timeseries + trial dataframes, saved with a metadata sidecar |
| `1D/scripts/run_experiment.py` | **new** — run a frozen agent through an experiment and export it |
| `1D/tests/test_dataset.py` | **new** — 18 tests checking the tables against the raw rollout |
| `1D/requirements.txt` | added `pandas` |
| `doc/1D_sequential_reach.md` | rewritten §3, §4, §6.1, §6.5, §6.6 (new), §7 (new, data layer), §8, §9, §10 |

74 tests pass (`.venv/bin/python -m pytest 1D/tests -q`).

**All runs in `1D/runs/` are stale.** They predate both fixes. Do not quote their
numbers; retrain first.

---

# Note for the next agent: how to run the campaign

Copy everything below this line into a fresh session on the GPU workstation.

---

## Context you need

This is the 1D stage of SeqHorizon. We are testing whether lookahead (co-articulation)
emerges from RL, and which task couplings it depends on. A previous ablation grid was
invalidated because most conditions never learned the task, so their co-articulation
indices described single-reach policies. Two fixes landed (an inelastic workspace
boundary, and a curriculum over sequence length) plus a competence gate on the analysis.
See `plan.md` above for the full account. **Everything in `1D/runs/` is stale and must be
retrained.**

## Do NOT use the GPU

This was measured, not guessed. The PPO update is ~90% of wall-clock time, and it is
dominated by a Python loop that calls `nn.GRU` one timestep at a time (256 sequential
calls on an 8×128 tensor per minibatch). Doubling the sequence length exactly doubles the
update time with no FLOP scaling — it is launch-latency bound, not compute bound. On CUDA
that becomes ~4096 tiny kernel launches per iteration on matrices too small to occupy one
SM, and a 4090 would likely lose to a single Ryzen core. Extra threads don't help either
(1 thread: 2.18 s per update; 4 threads: 2.34 s).

**Run many single-threaded jobs in parallel on the Ryzen cores instead.** Measured
throughput: ~3300 env-steps/s per core with video off, so 3M steps ≈ 15 min per arm.

## The commands

Set up (once):

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r 1D/requirements.txt
```

Confirm the code is sound before spending compute:

```bash
.venv/bin/python -m pytest 1D/tests -q
```

Run the full grid — set `--parallel` to about the number of physical cores:

```bash
.venv/bin/python 1D/scripts/run_ablations.py --timesteps 6000000 --parallel 12 --threads 1 --no-video --seed 1
```

Then repeat for seeds 2–5. **This is not optional.** Everything so far is single-seed, and
the outcome is bimodal (0.97 vs 0.21 success), so a single seed per condition is not a
measurement:

```bash
for s in 2 3 4 5; do
  .venv/bin/python 1D/scripts/run_ablations.py --timesteps 6000000 --parallel 12 --threads 1 --no-video --seed $s
done
```

Re-run just the analysis on existing checkpoints (fast, no training):

```bash
.venv/bin/python 1D/scripts/run_ablations.py --analyze-only --seed 1
```

Render videos afterwards from a checkpoint, rather than during training (video costs 2.8×
throughput — it is why the original grid ran at 221 steps/s instead of ~3300):

```bash
.venv/bin/python 1D/scripts/make_video.py --run 1D/runs/baseline_s1
```

## Flags that matter

- `--timesteps 6000000` — 3M was not enough for the harder conditions even before the
  fixes. Use 6M.
- `--threads 1` — always. More threads per job is slower; parallelism belongs across jobs.
- `--no-video` — always, for grid runs.
- `--parallel N` — number of physical cores, not logical. Hyperthreads contend.
- `--wandb --wandb-group <name>` — optional, groups the whole grid as one campaign.

## What to check, in order

1. **Did the curriculum run?** The training log prints `nT`. It must climb from 1.00 to
   2.30 over the first 1M steps, then stay flat.
2. **Is anything parked at the wall?** `wall_rate` in the log should end near 0.00. If an
   arm sits high, it found a new degenerate solution and Fix 1 needs revisiting.
3. **Watch `episodic_return_task`, not `episodic_return`.** The latter includes shaping
   income and can rise while success stays at zero.
4. **Read the competence split, not the headline success rate.** `analyze.py` prints
   `[1tgt X 2tgt Y 3tgt Z]`. If an arm is near 1.00 on one target and 0.00 on two, it
   learned a single reach and its CI means nothing regardless of what the number says.
5. **Only quote CI for arms above the competence floor.** `run_ablations.py` prints those
   above a line and everything else below it. If nothing lands above the line except
   `baseline` and `no_effort`, the fixes did not work — report that rather than reading
   the numbers below the line.

## Open questions the grid will not answer by itself

- `myopic` (γ=0.90) is expected to fail and is not fixable by training: the discounted
  return gap between doing the task and doing nothing collapses from 23.71 to 0.05. It
  demonstrates the confound rather than testing lookahead.
- `no_preview` has been **removed from the grid** — it could not be probed with a second
  target visible (out of distribution) and was tautological with it hidden (CI exactly 0
  for any agent). The baseline's within-agent preview-off control is the sound version.
  The stale directory `1D/runs/no_preview_s1` can be deleted.
- Under a dwell requirement the CI metric measures velocity at *entry* to the target band,
  while the separability theory constrains velocity at *departure*. **Ali is handling this
  analysis himself from the exported dataframes — do not change the metric.**

## Exporting data for analysis

After training, run frozen agents through named test experiments and export dataframes:

```bash
.venv/bin/python 1D/scripts/run_experiment.py --list
```

```bash
.venv/bin/python 1D/scripts/run_experiment.py --run 1D/runs/*_s1 --exp all
```

Writes `data/<experiment>_<agent>_<timestamp>_{timeseries,trials}.csv` plus a
`_meta.json` sidecar. Experiments are defined in `1D/seqreach/experiments.py`, one
function each — that file is meant to be edited. See §7 of the doc.
