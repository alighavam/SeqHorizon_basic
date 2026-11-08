"""SeqReach1D: a 1D point mass reaching to a sequence of targets.

The environment is written natively *batched* (N trials stepped together as numpy
arrays).  Two reasons:

1. Gymnasium 1.0 changed vector autoreset to next-step semantics, which silently breaks
   the CleanRL bootstrapping pattern.  Owning the vectorisation removes that trap; this
   env uses **same-step autoreset** (the semantics CleanRL's GAE loop assumes) and
   additionally reports ``final_obs`` so time-limit truncation can be bootstrapped
   correctly -- something the stock CleanRL script gets wrong.
2. It is ~10-50x faster than a Python-loop vector env, which decides feasibility on CPU.

``GymSeqReach1D`` wraps N=1 in the standard Gymnasium API for interop; a parity test
keeps the two paths from drifting.

Design note (the whole point of the testbed): the reward references only *target
acquisition, effort, and time*.  Nothing in it mentions trajectories, velocity profiles,
via points, reaction times, or preparatory states.  Any lookahead we measure later is a
consequence of optimisation under the plant + task structure, not of the objective.

Two features exist because the first ablation grid failed on them rather than on the
science, and both are documented at their definitions:

* the **inelastic workspace boundary** -- contact stops the mass and costs ``w_wall``.
  The previous position-only clip left a free resting place that seven of nine arms
  converged to.
* the **sequence-length curriculum** -- ``set_n_target_weights`` / ``curriculum_weights``,
  driven from the PPO loop.  Every failing arm was at 100% on one target and 0% on two,
  so the barrier is the second reach.  It changes learning dynamics only; the final
  sampling distribution is untouched.
"""

from __future__ import annotations

import numpy as np

from .config import Config

# Observation layout (per target slot): visible, pending, is_current, relative position.
_SLOT_FEATURES = 4
_BASE_FEATURES = 4  # x, v, a_prev, go_flag

# Fixed scales so the raw observation is O(1) even without running normalisation.
_X_SCALE = 1.0
_V_SCALE = 2.0

JUMP_NONE, JUMP_CURRENT, JUMP_FUTURE = 0, 1, 2


class BatchedSeqReach1D:
    """N independent 1D sequential-reach trials stepped in lockstep."""

    def __init__(self, cfg: Config, num_envs: int, seed: int = 0):
        self.cfg = cfg
        self.n = num_envs
        self.rng = np.random.default_rng(seed)

        p, t = cfg.plant, cfg.task
        self.K = t.max_targets
        self.obs_start = bool(t.obs_start)
        self.obs_dim = _BASE_FEATURES + _SLOT_FEATURES * self.K + int(self.obs_start)
        self.d_cue = int(p.cue_delay)
        self.act_dim = 1
        self.dt = p.dt
        self.d_sens = int(p.sensory_delay)

        n_ = num_envs
        f = lambda: np.zeros(n_, dtype=np.float64)
        i = lambda: np.zeros(n_, dtype=np.int64)

        # Plant state
        self.x, self.v, self.a, self.u_prev = f(), f(), f(), f()
        # Per-trial start position.  Always ``task.start_x`` in training; probes may
        # override it via ``spec["start_x"]`` to test reaches from elsewhere.
        self.x0 = np.full(n_, cfg.task.start_x, dtype=np.float64)
        # Trial state
        self.targets = np.zeros((n_, self.K), dtype=np.float64)
        self.n_targets, self.cur, self.dwell = i(), i(), i()
        self.preview, self.go_step, self.t = i(), i(), i()
        # Catch trials: the go cue never fires; the trial ends successfully at catch_end.
        self.catch = np.zeros(n_, dtype=bool)
        self.catch_end = i()
        # Perturbation state
        self.jump_kind, self.jump_step, self.jump_slot = i(), i(), i()
        self.jump_delta = f()
        self.jump_pending = np.zeros(n_, dtype=bool)
        self.jump_applied = np.zeros(n_, dtype=bool)
        # Episode bookkeeping.  ``ep_return_task`` excludes potential-based shaping: the
        # shaping term is policy-invariant but *not* return-invariant, and at low gamma it
        # pays enough for idling far from the target to dominate the logged return.  The
        # first grid's gamma=0.90 arm logged the highest episodic return of any run (46.7)
        # at 0.000 success on the strength of that.  Monitor the task return.
        self.ep_return, self.ep_return_task = f(), f()
        self.ep_len, self.ep_hits, self.ep_wall = i(), i(), i()

        self._x_hist = np.zeros((self.d_sens + 1, n_))
        self._v_hist = np.zeros((self.d_sens + 1, n_))

        # Live sampling weights over sequence length.  Starts at the *final* mixture, so
        # a freshly built env (probes, tests, evaluation) always samples the task as
        # specified; only ``set_n_target_weights`` moves it, and only training calls that.
        self._n_target_weights = self._as_weights(cfg.task.n_target_weights)

        self._pending_reset = np.ones(n_, dtype=bool)
        self._sample_trials(np.arange(n_))

    # ------------------------------------------------------------- curriculum
    def _as_weights(self, w) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        if w.size != self.K:
            raise ValueError(f"need {self.K} weights (max_targets), got {w.size}")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative and sum > 0, got {w}")
        return w / w.sum()

    def set_n_target_weights(self, w) -> None:
        """Set the sampling distribution over sequence length.

        The curriculum hook.  Affects trials sampled from now on; trials already in
        flight keep the length they were sampled with.
        """
        self._n_target_weights = self._as_weights(w)

    def curriculum_weights(self, global_step: int) -> np.ndarray:
        """Weights for ``global_step``: linear from the start mixture to the final one."""
        t = self.cfg.task
        final = self._as_weights(t.n_target_weights)
        if t.curriculum_steps <= 0:
            return final
        start = self._as_weights(t.curriculum_start_weights)
        frac = min(1.0, max(0.0, global_step / t.curriculum_steps))
        return self._as_weights((1.0 - frac) * start + frac * final)

    # ------------------------------------------------------------------ trials
    def _sample_trials(self, idx: np.ndarray, spec: dict | None = None) -> None:
        """(Re)initialise the trials at ``idx``.  ``spec`` overrides sampling (probes)."""
        if idx.size == 0:
            return
        cfg, t, p = self.cfg, self.cfg.task, self.cfg.plant
        m = idx.size
        spec = spec or {}

        if "start_x" in spec:
            x0 = np.broadcast_to(np.asarray(spec["start_x"], dtype=np.float64), (m,))
        elif t.start_range > 0:
            x0 = t.start_x + self.rng.uniform(-t.start_range, t.start_range, size=m)
        else:
            x0 = np.full(m, t.start_x)
        self.x0[idx] = x0
        self.x[idx] = x0
        self.v[idx] = 0.0
        self.a[idx] = 0.0
        self.u_prev[idx] = 0.0
        self.t[idx] = 0
        self.cur[idx] = 0
        self.dwell[idx] = 0
        self.ep_return[idx] = 0.0
        self.ep_return_task[idx] = 0.0
        self.ep_len[idx] = 0
        self.ep_hits[idx] = 0
        self.ep_wall[idx] = 0
        self.jump_applied[idx] = False
        self.jump_slot[idx] = -1

        # --- number of targets
        if "n_targets" in spec:
            n_tg = np.broadcast_to(np.asarray(spec["n_targets"]), (m,)).astype(np.int64)
        else:
            w = self._n_target_weights
            n_tg = self.rng.choice(len(w), size=m, p=w) + 1
        self.n_targets[idx] = n_tg

        # --- target positions: sequential, min-separation enforced by rejection
        if "targets" in spec:
            tg = np.array(np.broadcast_to(np.asarray(spec["targets"], dtype=np.float64),
                                          (m, self.K)))
        else:
            tg = np.zeros((m, self.K))
            prev = x0.copy()
            for k in range(self.K):
                cand = self.rng.uniform(-t.target_range, t.target_range, size=m)
                for _ in range(50):
                    bad = np.abs(cand - prev) < t.min_separation
                    if not bad.any():
                        break
                    cand[bad] = self.rng.uniform(-t.target_range, t.target_range,
                                                 size=int(bad.sum()))
                tg[:, k] = cand
                prev = cand
        self.targets[idx] = tg

        # --- preview horizon H (rolling window: slots cur .. cur+H-1 are visible)
        if "preview" in spec:
            H = np.broadcast_to(np.asarray(spec["preview"]), (m,)).astype(np.int64)
        elif t.fixed_preview is not None:
            H = np.full(m, int(t.fixed_preview), dtype=np.int64)
        else:
            full = self.rng.random(m) < t.full_preview_prob
            partial = self.rng.integers(1, np.maximum(n_tg, 2))  # 1 .. n_tg-1
            H = np.where(full, n_tg, np.minimum(partial, n_tg))
        self.preview[idx] = np.clip(H, 1, self.K)

        # --- go cue
        if "delay_steps" in spec:
            gs = np.broadcast_to(np.asarray(spec["delay_steps"]), (m,)).astype(np.int64)
        elif t.use_delay:
            lo, hi = int(t.delay_min_s / p.dt), int(t.delay_max_s / p.dt)
            gs = self.rng.integers(lo, hi + 1, size=m)
        else:
            gs = np.zeros(m, dtype=np.int64)
        # --- catch trials (go cue never comes)
        if "catch" in spec:
            ct = np.broadcast_to(np.asarray(spec["catch"], dtype=bool), (m,)).copy()
        elif t.p_catch > 0:
            ct = self.rng.random(m) < t.p_catch
        else:
            ct = np.zeros(m, dtype=bool)
        lo_c = int(t.delay_min_s / p.dt)
        hi_c = max(int(t.catch_max_s / p.dt), lo_c)
        self.catch[idx] = ct
        self.catch_end[idx] = np.where(ct, self.rng.integers(lo_c, hi_c + 1, size=m), 0)
        gs = np.where(ct, np.iinfo(np.int64).max // 2, gs)
        self.go_step[idx] = gs

        # --- jumps
        if "jump_kind" in spec:
            kind = np.broadcast_to(np.asarray(spec["jump_kind"]), (m,)).astype(np.int64)
        else:
            do = self.rng.random(m) < t.p_jump
            fut = self.rng.random(m) < t.p_jump_future
            kind = np.where(do, np.where(fut, JUMP_FUTURE, JUMP_CURRENT), JUMP_NONE)
        self.jump_kind[idx] = kind

        if "jump_step" in spec:
            js = np.broadcast_to(np.asarray(spec["jump_step"]), (m,)).astype(np.int64)
        else:
            js = gs + self.rng.integers(t.jump_delay_min, t.jump_delay_max + 1, size=m)
        self.jump_step[idx] = js

        if "jump_delta" in spec:
            jd = np.broadcast_to(np.asarray(spec["jump_delta"], dtype=np.float64), (m,))
        else:
            mag = self.rng.uniform(t.jump_size_min, t.jump_size_max, size=m)
            jd = mag * self.rng.choice([-1.0, 1.0], size=m)
        self.jump_delta[idx] = jd
        self.jump_pending[idx] = kind != JUMP_NONE

        # sensory history starts flat at the initial state
        self._x_hist[:, idx] = x0
        self._v_hist[:, idx] = 0.0

    # ------------------------------------------------------------------- masks
    def _visibility(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (visible, pending, is_current) boolean masks of shape (N, K)."""
        slots = np.arange(self.K)[None, :]
        in_seq = slots < self.n_targets[:, None]
        pending = in_seq & (slots >= self.cur[:, None])
        visible = pending & (slots <= (self.cur + self.preview - 1)[:, None])
        is_cur = slots == self.cur[:, None]
        return visible, pending, is_cur & in_seq

    def _obs(self) -> np.ndarray:
        x_hat = self._x_hist[self.d_sens]
        v_hat = self._v_hist[self.d_sens]
        visible, pending, is_cur = self._visibility()

        out = np.empty((self.n, self.obs_dim), dtype=np.float32)
        out[:, 0] = x_hat / _X_SCALE
        out[:, 1] = v_hat / _V_SCALE
        out[:, 2] = self.a / self.cfg.plant.u_max
        go_seen = self.t - self.d_cue >= self.go_step
        out[:, 3] = go_seen.astype(np.float64)
        rel = (self.targets - x_hat[:, None]) * visible / _X_SCALE
        for k in range(self.K):
            b = _BASE_FEATURES + _SLOT_FEATURES * k
            out[:, b + 0] = visible[:, k]
            out[:, b + 1] = pending[:, k]
            out[:, b + 2] = is_cur[:, k]
            out[:, b + 3] = rel[:, k]
        if self.obs_start:
            out[:, -1] = (self.x0 - x_hat) * (~go_seen) / _X_SCALE
        return out

    def _potential(self) -> np.ndarray:
        """PBRS potential Phi(s) = -c |x - x_current|, gated off before the go cue."""
        if self.cfg.reward.shaping != "pbrs":
            return np.zeros(self.n)
        cur_t = np.take_along_axis(self.targets, self.cur[:, None].clip(0, self.K - 1),
                                   axis=1)[:, 0]
        active = (self.t >= self.go_step) & (self.cur < self.n_targets)
        return -self.cfg.reward.shaping_coef * np.abs(self.x - cur_t) * active

    # -------------------------------------------------------------------- API
    def reset(self, spec: dict | None = None) -> np.ndarray:
        self._sample_trials(np.arange(self.n), spec)
        return self._obs()

    def step(self, u: np.ndarray):
        cfg, p, t, r = self.cfg, self.cfg.plant, self.cfg.task, self.cfg.reward
        u = np.clip(np.asarray(u, dtype=np.float64).reshape(self.n), -p.u_max, p.u_max)

        pre_go = self.t < self.go_step
        phi_before = self._potential()

        # --- target jumps (evaluated relative to the target being reached *right now*)
        fire = self.jump_pending & (self.t == self.jump_step)
        if fire.any():
            idx = np.flatnonzero(fire)
            slot = self.cur[idx] + (self.jump_kind[idx] == JUMP_FUTURE).astype(np.int64)
            ok = (slot < self.n_targets[idx]) & (slot <= self.cur[idx] + self.preview[idx] - 1)
            good = idx[ok]
            gslot = slot[ok]
            if good.size:
                newpos = np.clip(self.targets[good, gslot] + self.jump_delta[good],
                                 -t.target_range, t.target_range)
                self.targets[good, gslot] = newpos
                self.jump_slot[good] = gslot
                self.jump_applied[good] = True
            self.jump_pending[idx] = False

        # --- plant: activation lag -> signal-dependent noise -> point mass
        if p.tau_act > 0:
            self.a += (p.dt / p.tau_act) * (u - self.a)
        else:
            self.a = u.copy()
        noise = p.sigma_sdn * np.abs(self.a) * self.rng.standard_normal(self.n)
        if p.sigma_add > 0:
            noise += p.sigma_add * self.rng.standard_normal(self.n)
        a_eff = self.a + noise
        self.v += p.dt * (a_eff - p.damping * self.v) / p.mass
        self.x += p.dt * self.v

        # --- inelastic workspace boundary
        # Position used to be clipped here with velocity left untouched and unpenalised,
        # which made the boundary a *free* place to rest: the mass pinned there, the
        # shaping term paid (1-gamma)*coef*|x - T| per step, and nothing pushed back.
        # Seven of nine arms in the first ablation grid ended up parked there for up to
        # half the episode.  Contact now stops the mass and costs w_wall per step.
        at_wall = np.abs(self.x) > p.x_limit
        np.clip(self.x, -p.x_limit, p.x_limit, out=self.x)
        self.v = np.where(at_wall, 0.0, self.v)
        self.ep_wall += at_wall
        self.t += 1

        self._x_hist = np.roll(self._x_hist, 1, axis=0)
        self._v_hist = np.roll(self._v_hist, 1, axis=0)
        self._x_hist[0] = self.x
        self._v_hist[0] = self.v

        # --- costs
        rew = np.zeros(self.n)
        rew -= r.w_effort * (u / p.u_max) ** 2 * p.dt
        rew -= r.w_rate * ((u - self.u_prev) / p.u_max) ** 2
        self.u_prev = u

        post_go = ~pre_go
        rew -= r.w_time * p.dt * post_go
        rew -= r.w_hold * (self.x - self.x0) ** 2 * pre_go
        rew -= r.w_hold_v * self.v**2 * pre_go
        rew -= r.w_wall * at_wall

        # --- acquisition (only after the go cue)
        cur_t = np.take_along_axis(self.targets, self.cur[:, None].clip(0, self.K - 1),
                                   axis=1)[:, 0]
        inside = (np.abs(self.x - cur_t) <= t.tolerance) & post_go
        if r.shaping == "greedy":
            # Naive distance cost: a direct pull toward the target being reached for
            # right now.  Deliberately not policy-invariant -- see RewardConfig.
            rew -= (r.shaping_coef * np.abs(self.x - cur_t) * p.dt
                    * (post_go & (self.cur < self.n_targets)))
        self.dwell = np.where(inside, self.dwell + 1, 0)
        is_last = self.cur == (self.n_targets - 1)
        need = np.where(is_last, t.dwell_final, t.dwell_inter)
        acquired = inside & (self.dwell >= need)

        rew += np.where(acquired, np.where(is_last, r.r_final, r.r_hit), 0.0)
        self.cur = np.where(acquired, self.cur + 1, self.cur)
        self.dwell = np.where(acquired, 0, self.dwell)
        self.ep_hits += acquired

        completed = self.cur >= self.n_targets
        false_start = pre_go & (np.abs(self.x - self.x0) > t.hold_abort_dist)
        rew += np.where(false_start, r.r_false_start, 0.0)
        catch_done = self.catch & (self.t >= self.catch_end) & ~false_start
        rew += np.where(catch_done, r.r_catch, 0.0)

        terminated = completed | false_start | catch_done
        truncated = (self.t >= t.max_steps) & ~terminated

        # Task reward, banked before shaping is mixed in, so the training log can report
        # progress on the actual objective rather than on shaping income.
        self.ep_return_task += rew

        # --- potential-based shaping (Phi == 0 at terminal states, by convention)
        if r.shaping == "pbrs":
            phi_after = np.where(terminated, 0.0, self._potential())
            rew += cfg.ppo.gamma * phi_after - phi_before

        self.ep_return += rew
        self.ep_len += 1

        done = terminated | truncated
        obs = self._obs()
        info: dict = {}
        if done.any():
            di = np.flatnonzero(done)
            info["final_obs"] = obs[di].copy()
            info["done_idx"] = di
            # Terminal plant state, likewise unrecoverable after the same-step reset.
            info["final_state"] = {
                "x": self.x[di].copy(), "v": self.v[di].copy(),
                "a": self.a[di].copy(), "u": self.u_prev[di].copy(),
                "targets": self.targets[di].copy(),
            }
            info["episode"] = {
                "r": self.ep_return[di].copy(),
                "r_task": self.ep_return_task[di].copy(),
                "l": self.ep_len[di].copy(),
                "hits": self.ep_hits[di].copy(),
                "wall_steps": self.ep_wall[di].copy(),
                "n_targets": self.n_targets[di].copy(),
                "success": completed[di].copy(),
                "catch": self.catch[di].copy(),
                "catch_held": catch_done[di].copy(),
                "false_start": false_start[di].copy(),
                "timeout": truncated[di].copy(),
                # Pre-reset acquisition pointer.  Callers cannot recover this from the
                # returned state because autoreset happens in the same step.
                "cur": self.cur[di].copy(),
            }
            self._sample_trials(di)  # same-step autoreset (CleanRL semantics)
            obs = self._obs()

        return obs, rew.astype(np.float32), terminated, truncated, info

    # --------------------------------------------------------------- utilities
    def snapshot(self) -> dict:
        """Full observable-plus-hidden state, for trajectory recording in probes."""
        visible, pending, is_cur = self._visibility()
        return {
            "x": self.x.copy(), "v": self.v.copy(), "a": self.a.copy(),
            "u": self.u_prev.copy(), "t": self.t.copy(), "cur": self.cur.copy(),
            "targets": self.targets.copy(), "go_step": self.go_step.copy(),
            "n_targets": self.n_targets.copy(), "preview": self.preview.copy(),
            "visible": visible.copy(), "jump_applied": self.jump_applied.copy(),
            "jump_step": self.jump_step.copy(), "jump_slot": self.jump_slot.copy(),
        }


class GymSeqReach1D:
    """Single-trial Gymnasium-style adapter (interop, tests, notebooks)."""

    def __init__(self, cfg: Config, seed: int = 0):
        import gymnasium as gym

        self._b = BatchedSeqReach1D(cfg, num_envs=1, seed=seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self._b.obs_dim,),
                                                dtype=np.float32)
        self.observation_space.seed(seed)
        u = cfg.plant.u_max
        self.action_space = gym.spaces.Box(-u, u, (1,), dtype=np.float32)
        self.action_space.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._b.rng = np.random.default_rng(seed)
        return self._b.reset(options)[0], {}

    def step(self, action):
        obs, rew, term, trunc, info = self._b.step(np.asarray(action).reshape(1))
        return obs[0], float(rew[0]), bool(term[0]), bool(trunc[0]), info
