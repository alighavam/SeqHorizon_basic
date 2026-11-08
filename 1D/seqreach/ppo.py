"""PPO trainer — a faithful port of CleanRL's ``ppo_continuous_action.py`` with the
sequence-batched update from ``ppo_atari_lstm.py``.

Deliberate departures from the reference scripts, both of which are corrections rather
than tweaks:

* **Own batched env** instead of ``gym.vector.SyncVectorEnv`` (see ``env.py`` for why).
* **Correct time-limit bootstrapping.** The stock CleanRL scripts treat truncation as
  termination, which biases the value function whenever episodes hit the time limit —
  the common case early in training here, since a failed trial *is* a timeout. We
  bootstrap truncated transitions with ``gamma * V(final_obs)``.

Everything else (GAE, clipped surrogate, clipped value loss, entropy bonus, orthogonal
init, advantage normalisation, LR anneal, grad clipping, approx-KL) follows CleanRL.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import video as video_mod
from .config import Config
from .env import BatchedSeqReach1D
from .networks import Agent, RunningMeanStd
from .tracking import Tracker


def train(cfg: Config, run_dir: str | Path, log_every: int = 1, quiet: bool = False):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    tracker = Tracker(cfg, run_dir)

    p = cfg.ppo
    batch_size = p.num_envs * p.num_steps
    minibatch_size = batch_size // p.num_minibatches
    num_iterations = p.total_timesteps // batch_size
    assert p.num_envs % p.num_minibatches == 0, "recurrent update splits over envs"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    device = torch.device("cpu")

    envs = BatchedSeqReach1D(cfg, p.num_envs, seed=cfg.seed)
    obs_dim, act_dim = envs.obs_dim, envs.act_dim
    u_max = cfg.plant.u_max

    agent = Agent(obs_dim, act_dim, cfg.agent, p.init_logstd).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=p.learning_rate, eps=1e-5)

    obs_rms = RunningMeanStd((obs_dim,))
    ret_rms = RunningMeanStd(())
    ret_acc = np.zeros(p.num_envs)

    def norm_obs(o, update=True):
        if not p.norm_obs:
            return o
        if update:
            obs_rms.update(o.astype(np.float64))
        return obs_rms.normalize(o.astype(np.float64)).astype(np.float32)

    # storage
    obs_buf = torch.zeros((p.num_steps, p.num_envs, obs_dim))
    act_buf = torch.zeros((p.num_steps, p.num_envs, act_dim))
    logp_buf = torch.zeros((p.num_steps, p.num_envs))
    rew_buf = torch.zeros((p.num_steps, p.num_envs))
    done_buf = torch.zeros((p.num_steps, p.num_envs))
    val_buf = torch.zeros((p.num_steps, p.num_envs))

    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(norm_obs(envs.reset()))
    next_done = torch.zeros(p.num_envs)
    next_state = agent.initial_state(p.num_envs, device)

    ep_ret, ep_len, ep_succ, ep_hitfrac = (deque(maxlen=200) for _ in range(4))
    ep_ret_task, ep_wall = (deque(maxlen=200) for _ in range(2))
    # v2 hold diagnostics: false starts over all episodes, catch-trial holds over catch
    # episodes.  success_rate / hit_fraction count reaching trials only, so they stay
    # comparable with v1 runs (which have no catch trials).
    ep_fs, ep_catch = (deque(maxlen=200) for _ in range(2))
    log_path = run_dir / "train_log.jsonl"
    log_path.write_text("")

    vcfg = cfg.video
    video_dir = run_dir / "videos"
    video_iters = _video_iterations(vcfg, num_iterations)

    def make_video(iteration: int, step: int) -> Path | None:
        """Film the pinned evaluation trials at the current weights."""
        path = video_dir / f"rollout_it{iteration:05d}.gif"
        was_training = agent.training
        try:
            agent.eval()
            video_mod.record_video(
                agent, obs_rms, cfg, path, vcfg=vcfg,
                title=f"{cfg.run_name}  ·  it {iteration}/{num_iterations}  "
                      f"·  step {step:,}",
            )
        except Exception as exc:  # a broken renderer must never kill a training run
            print(f"[video] failed at iteration {iteration}: {exc!r}", flush=True)
            return None
        finally:
            agent.train(was_training)
        tracker.log_video(path, step=step, key="rollout",
                          caption=f"iteration {iteration} · step {step:,}")
        if not vcfg.save_local:
            path.unlink(missing_ok=True)
            return None
        return path

    for iteration in range(1, num_iterations + 1):
        initial_state = next_state.clone()
        if p.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * p.learning_rate

        # Sequence-length curriculum.  Advanced once per iteration rather than per step:
        # trials sampled mid-rollout keep whatever mixture was current when they started,
        # so a per-step schedule would buy nothing and make the rollout non-stationary
        # within a single GAE window.
        cur_weights = envs.curriculum_weights(global_step)
        envs.set_n_target_weights(cur_weights)

        for step in range(p.num_steps):
            global_step += p.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state = agent.get_action_and_value(
                    next_obs, next_state, next_done
                )
                val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logprob

            raw_obs, reward, term, trunc, info = envs.step(
                action.cpu().numpy().reshape(-1) * u_max
            )
            done = np.logical_or(term, trunc)

            # reward normalisation (Gymnasium NormalizeReward semantics)
            r = reward.astype(np.float64)
            if p.norm_reward:
                ret_acc = ret_acc * p.gamma * (1.0 - done) + r
                ret_rms.update(ret_acc)
                r = r / np.sqrt(ret_rms.var + 1e-8)

            # bootstrap value through time-limit truncation
            if trunc.any():
                ti = np.flatnonzero(trunc)
                pos = np.searchsorted(info["done_idx"], ti)
                fobs = torch.tensor(norm_obs(info["final_obs"][pos], update=False))
                with torch.no_grad():
                    fv = agent.get_value(
                        fobs, next_state[:, ti].contiguous(), torch.zeros(len(ti))
                    ).flatten()
                r[ti] += p.gamma * fv.numpy()

            rew_buf[step] = torch.tensor(r, dtype=torch.float32)
            next_obs = torch.tensor(norm_obs(raw_obs))
            next_done = torch.tensor(done, dtype=torch.float32)

            if "episode" in info:
                e = info["episode"]
                ep_ret.extend(e["r"].tolist())
                ep_ret_task.extend(e["r_task"].tolist())
                ep_len.extend(e["l"].tolist())
                reach = ~e["catch"]
                ep_succ.extend(e["success"][reach].astype(float).tolist())
                ep_hitfrac.extend((e["hits"][reach] / e["n_targets"][reach]).tolist())
                ep_fs.extend(e["false_start"].astype(float).tolist())
                ep_catch.extend(e["catch_held"][e["catch"]].astype(float).tolist())
                ep_wall.extend((e["wall_steps"] > 0).astype(float).tolist())

        # ---------------------------------------------------------------- GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(p.num_steps)):
                if t == p.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + p.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = (
                    delta + p.gamma * p.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logp_buf.reshape(-1)
        b_actions = act_buf.reshape(-1, act_dim)
        b_dones = done_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = val_buf.reshape(-1)

        # ------------------------------------------------------------- update
        clipfracs = []
        if agent.recurrent:
            envsperbatch = p.num_envs // p.num_minibatches
            envinds = np.arange(p.num_envs)
            flatinds = np.arange(batch_size).reshape(p.num_steps, p.num_envs)
        else:
            b_inds = np.arange(batch_size)

        for epoch in range(p.update_epochs):
            if agent.recurrent:
                np.random.shuffle(envinds)
                splits = [
                    (flatinds[:, envinds[s:s + envsperbatch]].ravel(),
                     envinds[s:s + envsperbatch])
                    for s in range(0, p.num_envs, envsperbatch)
                ]
            else:
                np.random.shuffle(b_inds)
                splits = [(b_inds[s:s + minibatch_size], None)
                          for s in range(0, batch_size, minibatch_size)]

            for mb_inds, mbenv in splits:
                state = (initial_state[:, mbenv].contiguous() if agent.recurrent
                         else initial_state)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], state, b_dones[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > p.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                if p.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                if p.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -p.clip_coef, p.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        v_unclipped, (v_clipped - b_returns[mb_inds]) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - p.ent_coef * entropy_loss + v_loss * p.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                optimizer.step()

            if p.target_kl is not None and approx_kl > p.target_kl:
                break

        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))

        rec = {
            "iteration": iteration,
            "global_step": global_step,
            "episodic_return": float(np.mean(ep_ret)) if ep_ret else None,
            # Return on the task objective alone, shaping excluded.  Watch this one:
            # `episodic_return` can rise on shaping income while success stays at zero.
            "episodic_return_task": float(np.mean(ep_ret_task)) if ep_ret_task else None,
            "episodic_length": float(np.mean(ep_len)) if ep_len else None,
            "success_rate": float(np.mean(ep_succ)) if ep_succ else None,
            "hit_fraction": float(np.mean(ep_hitfrac)) if ep_hitfrac else None,
            # Fraction of episodes touching the workspace boundary.  Should be ~0; a
            # persistently high value means the policy is parked at the wall.
            "wall_rate": float(np.mean(ep_wall)) if ep_wall else None,
            "false_start_rate": float(np.mean(ep_fs)) if ep_fs else None,
            "catch_hold_rate": float(np.mean(ep_catch)) if ep_catch else None,
            "mean_n_targets": float(np.dot(cur_weights, np.arange(1, len(cur_weights) + 1))),
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)),
            "explained_variance": float(explained_var),
            "logstd": float(agent.actor_logstd.mean().item()),
            "sps": sps,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if iteration % cfg.wandb.log_every == 0 or iteration == num_iterations:
            tracker.log({f"charts/{k}": v for k, v in rec.items()
                         if k not in ("iteration", "global_step")}
                        | {"iteration": iteration,
                           "charts/learning_rate": optimizer.param_groups[0]["lr"]},
                        step=global_step)
        if iteration in video_iters:
            make_video(iteration, global_step)

        if not quiet and (iteration % log_every == 0 or iteration == num_iterations):
            print(
                f"it {iteration:4d}/{num_iterations} step {global_step:>9,} "
                f"ret {rec['episodic_return_task']:7.2f} "
                f"succ {rec['success_rate']:.2f} hits {rec['hit_fraction']:.2f} "
                f"nT {rec['mean_n_targets']:.2f} wall {rec['wall_rate']:.2f} "
                f"ev {explained_var:5.2f} sps {sps}",
                flush=True,
            )

    ckpt = {
        "agent": agent.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "config": cfg.to_dict(),
    }
    torch.save(ckpt, run_dir / "checkpoint.pt")

    tracker.summary({
        "final/episodic_return": rec["episodic_return"],
        "final/success_rate": rec["success_rate"],
        "final/hit_fraction": rec["hit_fraction"],
        "final/global_step": global_step,
        "run_dir": str(run_dir),
    })
    if cfg.wandb.save_checkpoint:
        tracker.save_checkpoint(run_dir / "checkpoint.pt", step=global_step,
                                metadata={"success_rate": rec["success_rate"],
                                          "episodic_return": rec["episodic_return"]})
    tracker.finish()
    return agent, obs_rms, run_dir


def _video_iterations(vcfg, num_iterations: int) -> set[int]:
    """Which PPO iterations to film.  Always includes the first (an untrained baseline
    to compare against) and the last."""
    if not vcfg.enabled or vcfg.every_iters <= 0:
        return set()
    its = set(range(vcfg.every_iters, num_iterations + 1, vcfg.every_iters))
    return its | {1, num_iterations}


def load_agent(run_dir: str | Path):
    """Reload a trained policy plus its observation normaliser and config."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    env = BatchedSeqReach1D(cfg, 1, seed=0)
    agent = Agent(env.obs_dim, env.act_dim, cfg.agent, cfg.ppo.init_logstd)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    obs_rms = RunningMeanStd((env.obs_dim,))
    obs_rms.load_state_dict(ckpt["obs_rms"])
    return agent, obs_rms, cfg
