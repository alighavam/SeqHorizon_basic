"""Probe protocols: the experiments run on a trained policy.

Nothing here is used during training.  These are the behavioural assays that decide
whether lookahead emerged, borrowed directly from the human/monkey literature:

* ``preview_probe``  -- fix T1, sweep T2, with T2 either visible or hidden.  The
  visible-vs-hidden contrast is the causal test: same weights, same physics, same
  targets, the *only* difference is whether the agent can see where it goes next.
* ``delay_probe``    -- vary the duration of the hold period before the go cue.
* ``jump_probe``     -- displace the current or a future target mid-movement.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .env import JUMP_CURRENT, JUMP_FUTURE, JUMP_NONE, BatchedSeqReach1D


def probe_config(cfg: Config, noiseless: bool = True) -> Config:
    """Evaluation plant.  Motor noise off by default so a deterministic policy yields
    exactly reproducible trajectories; re-run with ``noiseless=False`` to confirm any
    effect survives signal-dependent noise."""
    over = {"task.p_jump": 0.0}
    if noiseless:
        over["plant.sigma_sdn"] = 0.0
        over["plant.sigma_add"] = 0.0
    return cfg.override(over)


@torch.no_grad()
def rollout(agent, obs_rms, cfg: Config, spec: dict, n_trials: int, seed: int = 0,
            deterministic: bool = True) -> dict:
    """Run ``n_trials`` trials defined by ``spec`` and record full trajectories.

    Returns arrays of shape (T, N) (or (T, N, K) for targets, (T, N, H) for the
    recurrent state) plus an ``alive`` mask that is False after a trial has ended --
    the env autoresets in place, so anything past ``alive`` belongs to a new trial.
    """
    env = BatchedSeqReach1D(cfg, n_trials, seed=seed)
    obs = env.reset(spec)
    T = cfg.task.max_steps
    u_max = cfg.plant.u_max

    state = agent.initial_state(n_trials, torch.device("cpu"))
    done_t = torch.zeros(n_trials)
    finished = np.zeros(n_trials, dtype=bool)

    rec = {k: [] for k in ("x", "v", "a", "u", "cur", "targets", "alive", "h", "r")}
    meta = {
        "targets0": env.targets.copy(), "n_targets": env.n_targets.copy(),
        "preview": env.preview.copy(), "go_step": env.go_step.copy(),
        "jump_step": env.jump_step.copy(), "jump_kind": env.jump_kind.copy(),
    }
    end_step = np.full(n_trials, T, dtype=np.int64)
    success = np.zeros(n_trials, dtype=bool)
    # Outcome bookkeeping for the competence gate (see ``outcome_breakdown``).
    false_start = np.zeros(n_trials, dtype=bool)
    timeout = np.zeros(n_trials, dtype=bool)
    wall_steps = np.zeros(n_trials, dtype=np.int64)
    # Step index at which each target slot was acquired (-1 = never).
    acq_step = np.full((n_trials, env.K), -1, dtype=np.int64)
    # Step index at which the mass *first entered* each target's tolerance band.  This
    # is the pass-through moment, and it is what the co-articulation metric must use:
    # when a dwell is required, acquisition happens only after the dwell completes, so
    # the velocity there is ~0 by construction rather than by absence of lookahead.
    entry_step = np.full((n_trials, env.K), -1, dtype=np.int64)

    for t in range(T):
        o = obs.astype(np.float64)
        if cfg.ppo.norm_obs:
            o = obs_rms.normalize(o)
        ot = torch.tensor(o, dtype=torch.float32)
        action, _, _, _, state = agent.get_action_and_value(
            ot, state, done_t, deterministic=deterministic
        )
        rec["h"].append(state[0].numpy().copy())

        prev_cur = env.cur.copy()
        obs, rew, term, trunc, info = env.step(action.numpy().reshape(-1) * u_max)
        snap = env.snapshot()
        done = term | trunc

        # The env autoresets in the same step, so post-step fields belong to the *next*
        # trial for anything that just finished.  Recover the pre-reset values.
        cur_after = snap["cur"].copy()
        if "episode" in info:
            di = info["done_idx"]
            cur_after[di] = info["episode"]["cur"]
            for k, val in info["final_state"].items():
                snap[k][di] = val
            fresh = di[~finished[di]]
            if fresh.size:
                pos = np.searchsorted(di, fresh)
                success[fresh] = info["episode"]["success"][pos]
                false_start[fresh] = info["episode"]["false_start"][pos]
                timeout[fresh] = info["episode"]["timeout"][pos]
                wall_steps[fresh] = info["episode"]["wall_steps"][pos]

        newly = (cur_after > prev_cur) & ~finished
        tgt_now = np.take_along_axis(
            snap["targets"], prev_cur[:, None].clip(0, env.K - 1), axis=1
        )[:, 0]
        inside = ((np.abs(snap["x"] - tgt_now) <= cfg.task.tolerance)
                  & (t >= meta["go_step"]) & ~finished)
        for slot in range(env.K):
            hit = newly & (prev_cur == slot)
            acq_step[hit, slot] = t
            first = inside & (prev_cur == slot) & (entry_step[:, slot] < 0)
            entry_step[first, slot] = t

        for k in ("x", "v", "a", "u", "targets"):
            rec[k].append(snap[k])
        rec["cur"].append(cur_after)
        rec["alive"].append(~finished)
        rec["r"].append(rew.astype(np.float64))

        just = done & ~finished
        end_step[just] = t
        finished |= done
        done_t = torch.tensor(done.astype(np.float32))
        if finished.all():
            break

    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(meta)
    out["end_step"] = end_step
    out["success"] = success
    out["false_start"] = false_start
    out["timeout"] = timeout
    out["wall_steps"] = wall_steps
    out["acq_step"] = acq_step
    out["entry_step"] = entry_step
    out["dt"] = cfg.plant.dt
    return out


# --------------------------------------------------------------- competence
# Threshold for treating a run's behavioural measurements as interpretable.  An agent
# below this is not doing the task, and a co-articulation index read off such an agent
# describes a policy that plans no second movement -- see doc/1D_task_and_design.md §8.1.
COMPETENCE_FLOOR = 0.90


def outcome_breakdown(traj: dict, floor: float = COMPETENCE_FLOOR) -> dict:
    """Why a run fails, not just how often -- the gate on every other measurement.

    The first ablation grid reported co-articulation for arms that scored 100% on
    one-target trials and 0% on two-target trials.  Splitting success by sequence length
    makes that immediately visible, so it is computed alongside the headline rate and
    checked before any index is quoted.
    """
    n_tg = traj["n_targets"]
    succ = traj["success"]
    hits = (traj["acq_step"] >= 0).sum(1) / np.maximum(n_tg, 1)
    by_len = {int(k): float(succ[n_tg == k].mean()) for k in np.unique(n_tg)}
    multi = n_tg > 1
    return {
        "success_rate": float(succ.mean()),
        "hit_fraction": float(hits.mean()),
        "success_by_n_targets": by_len,
        # The diagnostic number: an agent that only ever performs one reach scores 1.0
        # on single-target trials and 0.0 here.
        "success_multi_target": float(succ[multi].mean()) if multi.any() else float("nan"),
        "false_start_rate": float(traj["false_start"].mean()),
        "timeout_rate": float(traj["timeout"].mean()),
        "wall_contact_rate": float((traj["wall_steps"] > 0).mean()),
        "mean_wall_steps": float(traj["wall_steps"].mean()),
        "competence_floor": floor,
        "competent": bool(succ.mean() >= floor),
    }


# --------------------------------------------------------------------- probes
def preview_probe(cfg: Config, t1: float = 0.45, n_t2: int = 25, repeats: int = 1,
                  delay_steps: int = 25, both_directions: bool = True) -> tuple[dict, dict]:
    """Fix T1, sweep T2, crossed with preview on/off and (optionally) reach direction.

    Returns ``(spec, design)`` where ``design`` carries the per-trial factor levels.
    """
    t = cfg.task
    grid = np.linspace(-t.target_range, t.target_range, n_t2)
    dirs = [1.0, -1.0] if both_directions else [1.0]

    T1, T2, PREV, DIR = [], [], [], []
    for d in dirs:
        a = t1 * d
        for b in grid * 1.0:
            if abs(b - a) < t.min_separation or abs(b - t.start_x) < t.min_separation:
                continue
            for h in (2, 1):  # 2 = T2 visible from trial onset, 1 = hidden until T1 taken
                for _ in range(repeats):
                    T1.append(a), T2.append(b), PREV.append(h), DIR.append(d)

    T1, T2 = np.array(T1), np.array(T2)
    tg = np.zeros((len(T1), t.max_targets))
    tg[:, 0], tg[:, 1] = T1, T2
    spec = {
        "n_targets": np.full(len(T1), 2), "targets": tg,
        "preview": np.array(PREV), "delay_steps": np.full(len(T1), delay_steps),
        "jump_kind": np.full(len(T1), JUMP_NONE),
    }
    design = {"t1": T1, "t2": T2, "preview": np.array(PREV), "direction": np.array(DIR)}
    return spec, design


def delay_probe(cfg: Config, delays: tuple[int, ...] = (5, 12, 25, 37, 50),
                n_per: int = 40, seed: int = 0) -> tuple[dict, dict]:
    """Random 2-target trials with full preview, crossed with hold duration."""
    rng = np.random.default_rng(seed)
    t = cfg.task
    D, TG = [], []
    for d in delays:
        for _ in range(n_per):
            while True:
                a, b = rng.uniform(-t.target_range, t.target_range, 2)
                if (abs(a - t.start_x) > t.min_separation
                        and abs(b - a) > t.min_separation):
                    break
            row = np.zeros(t.max_targets)
            row[0], row[1] = a, b
            TG.append(row), D.append(d)

    D, TG = np.array(D), np.array(TG)
    spec = {
        "n_targets": np.full(len(D), 2), "targets": TG,
        "preview": np.full(len(D), 2), "delay_steps": D,
        "jump_kind": np.full(len(D), JUMP_NONE),
    }
    # t1/t2 travel with the design so the same rollout can serve the decoding analysis.
    return spec, {"delay_steps": D, "t1": TG[:, 0], "t2": TG[:, 1]}


def jump_probe(cfg: Config, t1: float = 0.45, t2: float = -0.45, delta: float = 0.35,
               jump_after_go: int = 8, delay_steps: int = 25) -> tuple[dict, dict]:
    """Matched triplets: no jump / current-target jump / future-target jump.

    The future-target condition is the sharp one -- only T2 moves, so any change in the
    *ongoing* reach to T1 is evidence the agent was planning through T2.
    """
    kinds = [JUMP_NONE, JUMP_CURRENT, JUMP_FUTURE]
    signs = [1.0, -1.0]
    K, S = [], []
    for k in kinds:
        for s in signs:
            K.append(k), S.append(s)
    n = len(K)
    tg = np.zeros((n, cfg.task.max_targets))
    tg[:, 0], tg[:, 1] = t1, t2
    spec = {
        "n_targets": np.full(n, 2), "targets": tg, "preview": np.full(n, 2),
        "delay_steps": np.full(n, delay_steps),
        "jump_kind": np.array(K),
        "jump_step": np.full(n, delay_steps + jump_after_go),
        "jump_delta": np.array(S) * delta,
    }
    return spec, {"jump_kind": np.array(K), "jump_sign": np.array(S)}
