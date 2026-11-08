"""Configuration for the 1D sequential-reach testbed.

Every knob that could plausibly create (or destroy) lookahead lives here, so that an
ablation is a one-line config override rather than a code change.  See ``doc/`` for the
argument about which of these are load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PlantConfig:
    """Point-mass plant.  Second order + a first-order activation lag."""

    dt: float = 0.02
    mass: float = 1.0
    damping: float = 0.2
    u_max: float = 8.0
    # First-order activation lag: force cannot change instantaneously (muscle-like).
    # tau_act = 0 collapses to a pure double integrator.  Ablation axis C5.
    tau_act: float = 0.05
    # Signal-dependent motor noise (Harris & Wolpert): sd proportional to |activation|.
    sigma_sdn: float = 0.05
    # Constant additive motor noise.
    sigma_add: float = 0.0
    # Sensory feedback delay in steps.  >0 forces the policy to predict its own state.
    # Applies to the proprioceptive channels (x, v) and hence to the egocentric target
    # and start offsets, which are computed from the delayed position.
    sensory_delay: int = 0
    # Delay, in steps, before the go cue reaches the agent.  0 means the agent sees the
    # cue on the step it fires, so it can respond with no latency and "reaction time" is
    # not a meaningful quantity.  >0 makes RT >= cue_delay * dt by construction.
    cue_delay: int = 0
    # Workspace half-width.  The boundary is *inelastic*: contact zeroes velocity and is
    # charged for (reward.w_wall).  An earlier version clipped position and left velocity
    # untouched and unpenalised, which made "drive into the wall and sit" a free resting
    # place and trapped most of the ablation grid there.  See doc/1D_task_and_design.md
    # §6.6.  Targets live within +-task.target_range = 0.9, so a competent policy never
    # reaches this and the boundary rule cannot affect it.
    x_limit: float = 1.5


@dataclass
class TaskConfig:
    """Trial structure: targets, timing, acquisition criteria, perturbations."""

    max_targets: int = 3
    # Probability weights over the number of targets in a trial (index 0 -> 1 target).
    # These are the *final* weights; see curriculum_steps for how they are approached.
    n_target_weights: tuple[float, ...] = (0.2, 0.3, 0.5)

    # Curriculum over sequence length.  Sampling weights are interpolated linearly from
    # ``curriculum_start_weights`` to ``n_target_weights`` over the first
    # ``curriculum_steps`` environment steps, then held fixed.  0 disables it.
    #
    # Why this exists: in the first ablation grid every failing arm sat at 100% success
    # on 1-target trials and 0% on 2-target trials.  The barrier is the *second* reach,
    # not reaching at all, and agents that meet the full mixture immediately commit to a
    # single-reach policy and stop exploring.  Training on 1-target trials first and
    # mixing longer sequences in gets them past that cliff.  This changes learning
    # dynamics only -- the final sampling distribution, and hence the task being
    # evaluated, is identical.
    curriculum_steps: int = 1_000_000
    curriculum_start_weights: tuple[float, ...] = (1.0, 0.0, 0.0)

    target_range: float = 0.9
    min_separation: float = 0.15
    start_x: float = 0.0
    # v2: start position drawn per trial from start_x + U[-start_range, start_range].
    # 0 (default) reproduces v1, where every trial starts at start_x.  The hold penalty
    # and the false-start check are always relative to the trial's own start.
    start_range: float = 0.0
    # v2: append the egocentric offset to the start position, (x0 - x_hat), to the
    # observation while the go cue has not yet been perceived -- the analogue of a
    # visible centre-hold target.  Without it a randomly placed start can only be
    # remembered, not seen.  Appended *last*, so the v1 layout is unchanged.
    obs_start: bool = False

    # Must clear the *mid-training* trajectory length, not the converged one.  At 200
    # the cap did not bind for a converged policy (~110 steps) but truncated the longer
    # exploratory trajectories learning has to pass through: only 1 of 3 baseline seeds
    # solved the task.  At 600 all seeds solve it.
    max_steps: int = 600  # 12.0 s at dt=0.02

    # Delay period (hold at start until an unpredictable go cue).
    use_delay: bool = True
    delay_min_s: float = 0.25
    delay_max_s: float = 1.00
    hold_abort_dist: float = 0.15  # leaving this window before go aborts the trial
    # v2 catch trials: with probability p_catch the go cue never comes.  The agent must
    # hold for catch_hold_s (drawn from [delay_min_s, catch_max_s]) and is then paid
    # reward.r_catch.  Makes drifting towards the target before the cue costly, and
    # flattens the hazard of the go cue (it is no longer certain by delay_max_s).
    p_catch: float = 0.0
    catch_max_s: float = 1.5

    # Acquisition.  dwell_inter is the separability knob (ablation axis C1):
    # a long intermediate dwell forces v=0 at each target and provably destroys lookahead.
    tolerance: float = 0.05
    dwell_inter: int = 1
    dwell_final: int = 15  # 0.3 s

    # Preview.  Rolling window: target i is visible iff cur <= i <= cur + H - 1.
    # H = 1 means "only the current target" (no lookahead possible).  Ablation axis C6.
    full_preview_prob: float = 0.5
    fixed_preview: int | None = None  # if set, forces H for every trial

    # Target jumps (online replanning probes).
    p_jump: float = 0.0
    p_jump_future: float = 0.5  # given a jump, probability it hits a *later* target
    jump_delay_min: int = 3
    jump_delay_max: int = 25
    jump_size_min: float = 0.15
    jump_size_max: float = 0.40


@dataclass
class RewardConfig:
    """The entire task specification the agent ever sees.

    Note what is absent: no trajectory target, no velocity profile, no via-point, no
    reaction-time term, no preparatory-state term.
    """

    r_hit: float = 10.0
    r_final: float = 20.0
    r_false_start: float = -5.0
    r_catch: float = 5.0  # paid for holding through a catch trial (task.p_catch)

    w_effort: float = 0.5  # on (u/u_max)^2 * dt        -- ablation axis C2
    w_rate: float = 0.05  # on ((u - u_prev)/u_max)^2
    w_time: float = 1.0  # per second while targets remain -- ablation axis C3
    w_hold: float = 20.0  # on (x - x0)^2 before the go cue
    w_hold_v: float = 2.0  # on v^2 before the go cue

    # Per step of contact with the workspace boundary.  Sized against the two costs it
    # competes with: time is 0.02/step and shaping pays at most (1-gamma)*coef*|x-T| =
    # 0.024/step out there, so 0.1 makes boundary contact decisively negative rather than
    # break-even.  A competent policy never touches the boundary (measured: baseline and
    # no_effort make contact on 0% of trials), so this term is inert once the task is
    # solved -- it exists to remove a resting place, not to shape the solution.
    w_wall: float = 0.1

    # Potential-based shaping, F = gamma*Phi(s') - Phi(s) with Phi = -c*|x - x_current|.
    # Policy-invariant (Ng et al. 1999), used only as a learning aid.  "none" disables it.
    # "greedy" is the *naive* variant: a per-step cost -coef*|x - x_current|*dt paid
    # directly, not as a potential difference.  It is NOT policy-invariant -- a pull
    # toward the current target is implicit supervision of the very effect we study --
    # so it exists only as an ablation, to measure that confound rather than assert it.
    shaping: str = "pbrs"  # "pbrs" | "none" | "greedy"
    shaping_coef: float = 2.0


@dataclass
class AgentConfig:
    recurrent: bool = True  # GRU; False -> MLP (ablation)
    hidden_size: int = 128
    rnn_size: int = 128
    n_hidden_layers: int = 2
    activation: str = "tanh"


@dataclass
class PPOConfig:
    """CleanRL PPO hyperparameters (ppo_continuous_action / ppo_atari_lstm)."""

    total_timesteps: int = 2_000_000
    num_envs: int = 32
    num_steps: int = 256
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    # gamma sets how far credit can travel.  Too myopic and the agent cannot "see" the
    # next target's cost even with full visual preview.  Headline ablation axis C7.
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    norm_adv: bool = True
    norm_reward: bool = True
    norm_obs: bool = True
    init_logstd: float = -0.5


@dataclass
class VideoConfig:
    """Rendered rollouts (GIF).  Used both during training and offline on a checkpoint.

    The evaluation trials are pinned by ``seed`` and (by default) run noiseless with a
    deterministic policy, so the *same* trials are replayed at every checkpoint and the
    videos across training are directly comparable frame for frame.
    """

    enabled: bool = True
    every_iters: int = 25  # render every N PPO iterations (0 disables periodic video)
    n_trials: int = 3  # trials concatenated into one GIF
    stride: int = 2  # render every s-th env step (2 -> 25 fps at dt=0.02)
    real_time: bool = True  # play back at 1x; False uses ``fps``
    fps: int = 25
    trail_steps: int = 30  # length of the fading speed-coloured trace, in env steps
    speed_max: float = 2.0  # |v| mapped to the top of the blue->red colormap
    width_px: int = 760
    dpi: int = 100
    seed: int = 1234
    noiseless: bool = True  # motor noise off, so replays are exactly reproducible
    deterministic: bool = True  # policy mean instead of a sample
    save_local: bool = True  # keep the GIFs under <run_dir>/videos/


@dataclass
class WandbConfig:
    """Weights & Biases tracking.  Off unless explicitly enabled."""

    enabled: bool = False
    project: str = "seqhorizon-1d"
    entity: str | None = None  # None -> your default W&B entity
    group: str | None = None  # e.g. "ablations-3M"; groups runs in the UI
    job_type: str = "train"
    tags: tuple[str, ...] = ()
    notes: str = ""
    mode: str = "online"  # "online" | "offline" | "disabled"
    log_every: int = 1  # log metrics every N PPO iterations
    save_checkpoint: bool = True  # upload checkpoint.pt as a versioned artifact
    save_code: bool = True


@dataclass
class Config:
    exp_name: str = "baseline"
    seed: int = 1
    torch_threads: int = 4
    plant: PlantConfig = field(default_factory=PlantConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def run_name(self) -> str:
        """Canonical name for this run -- the runs/ directory and the W&B run share it."""
        return f"{self.exp_name}_s{self.seed}"

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _build(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def override(self, dotted: dict[str, Any]) -> "Config":
        """Return a copy with ``{"ppo.gamma": 0.9}``-style overrides applied."""
        d = self.to_dict()
        for key, value in dotted.items():
            node = d
            *parents, leaf = key.split(".")
            for p in parents:
                if p not in node:
                    raise KeyError(f"unknown config section '{p}' in '{key}'")
                node = node[p]
            if leaf not in node:
                raise KeyError(f"unknown config key '{key}'")
            node[leaf] = value
        return Config.from_dict(d)


# Nested sections, resolved by name.  ``from __future__ import annotations`` makes
# dataclass field types plain strings, so introspecting them is not an option.
_SECTIONS = {
    "plant": PlantConfig,
    "task": TaskConfig,
    "reward": RewardConfig,
    "agent": AgentConfig,
    "ppo": PPOConfig,
    "video": VideoConfig,
    "wandb": WandbConfig,
}


def _build(cls, d: dict[str, Any]):
    """Recursively instantiate nested dataclasses, rejecting unknown keys."""
    kwargs = {}
    known = {f.name for f in fields(cls)}
    for key, value in d.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        if key in _SECTIONS and isinstance(value, dict) and cls is Config:
            kwargs[key] = _build(_SECTIONS[key], value)
        elif key in ("n_target_weights", "curriculum_start_weights",
                     "tags") and value is not None:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
