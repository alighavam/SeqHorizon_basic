# 1D sequential reaching — emergent lookahead testbed

A point mass on a line reaches to a sequence of targets under PPO. The question is
whether **lookahead** (shaping the reach to target 1 according to where target 2 is),
**delay-period preparation**, and **online replanning** appear *without being trained
for*. Nothing in the reward mentions trajectories, velocity profiles, reaction times, or
preparatory states — the agent is only told to hit the targets in order, cheaply and
quickly.

Two documents cover this stage:
[`doc/1D_task_and_design.md`](../doc/1D_task_and_design.md) — the separability argument,
the task, the maths behind the discount and the shaping term, and the definition of every
measurement; and [`doc/1D_results.md`](../doc/1D_results.md) — every measured result, with
seed counts and the checks that license reading it.

## Install

The environment is a conda prefix at the repo root, `.conda/` (gitignored):

```bash
conda create -p .conda python=3.11 && .conda/bin/pip install -r 1D/requirements.txt
```

Commands below write `python` for `.conda/bin/python` (or `conda activate ./.conda`).

## Use

Train the baseline:

```bash
python 1D/scripts/train.py --name baseline --timesteps 3000000
```

Run the probe battery and write figures + `results.json`:

```bash
python 1D/scripts/analyze.py --run 1D/runs/baseline_s1
```

Watch a trained agent do the task:

```bash
python 1D/scripts/make_video.py --run 1D/runs/baseline_s1
```

Train and compare the ablation grid:

```bash
python 1D/scripts/run_ablations.py --timesteps 3000000 --parallel 6 --threads 1 --no-video --seed 1
```

The arms are registered in `ABLATIONS` at the top of `scripts/run_ablations.py`. Run it
once per seed; one seed is not a result.

Any config field can be overridden from the CLI with dotted keys:

```bash
python 1D/scripts/train.py --name gamma097_sparse --set ppo.gamma=0.97 reward.shaping=none
```

Tests:

```bash
python -m pytest 1D/tests -q
```

## Layout

| path | what |
|---|---|
| `seqreach/config.py` | every knob, grouped into plant / task / reward / agent / ppo |
| `seqreach/env.py` | batched numpy env (`BatchedSeqReach1D`) + Gymnasium adapter |
| `seqreach/networks.py` | GRU / MLP actor-critic, CleanRL init |
| `seqreach/ppo.py` | PPO trainer |
| `seqreach/probes.py` | preview / delay / jump protocols, trajectory recording, competence gate |
| `seqreach/metrics.py` | co-articulation index, RT, jump latency, preparatory geometry |
| `seqreach/experiments.py` | registry of test experiments (trial lists) run on frozen agents |
| `seqreach/dataset.py` | rollout → timeseries / trial dataframes for export |
| `seqreach/plots.py` | figures |
| `seqreach/video.py` | animated GIF of a rollout (training and test time) |
| `seqreach/tracking.py` | Weights & Biases facade; inert when disabled |
| `scripts/` | `train.py`, `analyze.py`, `run_ablations.py`, `run_experiment.py`, `make_video.py`, `wandb_sync.py` |
| `notebooks/` | four lessons, in order: `01_task_and_physics`, `02_reward`, `03_agent_and_training`, `04_measuring_an_agent` (helpers in `lesson_tools.py`) |
| `runs/` | checkpoints, logs, `videos/`, `analysis/` figures (gitignored; checkpoints are on W&B) |

## Video

`seqreach/video.py` renders a rollout as a GIF. Two coupled panels, because neither
alone is enough:

- **Scene** — the workspace as a line. Targets are circles whose radius *is* the
  acquisition tolerance, numbered in sequence order and styled by what the agent knows
  right now: dashed grey = beyond the preview horizon, solid dark = visible, amber =
  the one being reached for, green = already acquired. The point mass drags a
  speed-coloured comet tail (blue slow → red fast) covering only the last
  `video.trail_steps` steps — a full-trial trace on a line turns to mush the moment the
  agent reverses. A caption flashes the events that matter for the lookahead question:
  the go cue, **a future target entering the horizon**, an acquisition, a target jump.
- **Timeline** — position against time in the same speed colormap, with the tolerance
  bands and the go cue. Time never doubles back, so here the whole trial can be shown.

A running clock, the horizon `H`, which targets are currently visible, `x`, `v`, the
cumulative return, and the full plant/task/reward/PPO parameters are on every frame.

```bash
python 1D/scripts/make_video.py --run 1D/runs/baseline_s1                # task trials
python 1D/scripts/make_video.py --run 1D/runs/baseline_s1 --probe jump   # replanning
python 1D/scripts/make_video.py --run 1D/runs/baseline_s1 --noisy --stochastic
```

The evaluation trials are pinned by `video.seed` and run noiseless with a deterministic
policy, so the *same* trials are replayed at every checkpoint and successive videos are
comparable frame for frame. The first two trials of the default battery are the same
sequence with `H = 3` and `H = 1`, which puts the visible/hidden contrast side by side
in one clip.

## Weights & Biases

Tracking is off by default and the codebase never imports `wandb` unless it is enabled,
so nothing changes if you do not use it.

```bash
wandb login                                    # once
python 1D/scripts/train.py --name baseline --timesteps 3000000 --wandb
```

Logged: every metric already in `train_log.jsonl` (returns, success rate, hit fraction,
policy/value loss, entropy, approx-KL, clipfrac, explained variance, logstd, SPS, LR)
under `charts/`, plus a rollout GIF every `video.every_iters` iterations (default 25,
which is ~11 videos over a 3M-step run) and the checkpoint as a versioned `model`
artifact at the end.

Naming, so runs stay findable once there are hundreds:

| thing | value |
|---|---|
| run name | `<exp_name>_s<seed>` — same string as the `runs/` directory |
| group | `--wandb-group`, e.g. one ablation campaign |
| artifact | `<exp_name>_s<seed>`, type `model`, aliased `latest` and `step-<N>` |

Push runs you have **already** trained — curves replayed from `train_log.jsonl` at their
original steps, analysis figures and `results.json` attached, a GIF rendered from the
checkpoint, and the checkpoint uploaded as an artifact:

```bash
python 1D/scripts/wandb_sync.py --all --group backfill
```

The whole ablation grid as one group:

```bash
python 1D/scripts/run_ablations.py --timesteps 3000000 --parallel 6 --threads 1 --no-video --wandb
```

## Notes on the implementation

**PPO** is a port of CleanRL's `ppo_continuous_action.py` with the sequence-batched
recurrent update from `ppo_atari_lstm.py`. Two deliberate departures:

- The environment is **natively batched** rather than wrapped in
  `gym.vector.SyncVectorEnv`. Gymnasium 1.0 switched vector autoreset to next-step
  semantics, which silently breaks CleanRL's bootstrapping; owning the vectorisation
  avoids that and runs much faster on CPU. The env uses same-step autoreset, the
  semantics CleanRL's GAE loop assumes.
- **Time-limit truncation is bootstrapped** with `gamma * V(final_obs)`. The stock
  CleanRL scripts treat truncation as termination, which biases the value function —
  and here a failed trial *is* a timeout, so that case is common early in training.

**Shaping.** `reward.shaping="pbrs"` adds potential-based shaping
(`F = γΦ(s') − Φ(s)`, `Φ = −c·|x − x_current|`) as a learning aid only; it is
policy-invariant. `reward.shaping="greedy"` is the naive `−c·|x − x_current|` cost; it is
*not* policy-invariant — a pull toward the current target is implicit supervision of
exactly the effect under study — so it exists only as an ablation arm, to measure that
confound. The `no_shaping` arm confirms the headline result survives with sparse reward
alone.

**Probes are noiseless by default** (`sigma_sdn=0`) so a deterministic policy yields
exactly reproducible trajectories. Re-run `analyze.py --noisy` to confirm effects
survive signal-dependent motor noise.
