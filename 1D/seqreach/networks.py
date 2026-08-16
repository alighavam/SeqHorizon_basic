"""Actor-critic networks, following CleanRL's initialisation conventions.

The GRU path mirrors ``ppo_atari_lstm.py``: the recurrent state is carried across the
rollout and zeroed at episode boundaries via the ``done`` mask, so a single forward call
can span episode resets inside a sequence minibatch.

The recurrent state is also the object of the neural analysis later on -- delay-period
preparation, if it emerges, has to live here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg_agent, init_logstd: float = -0.5):
        super().__init__()
        self.recurrent = cfg_agent.recurrent
        act = _ACT[cfg_agent.activation]

        layers, d = [], obs_dim
        for _ in range(cfg_agent.n_hidden_layers):
            layers += [layer_init(nn.Linear(d, cfg_agent.hidden_size)), act()]
            d = cfg_agent.hidden_size
        self.trunk = nn.Sequential(*layers)

        if self.recurrent:
            self.rnn = nn.GRU(d, cfg_agent.rnn_size)
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
            d = cfg_agent.rnn_size
        self.feature_dim = d

        self.actor_mean = layer_init(nn.Linear(d, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d, 1), std=1.0)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(init_logstd)))

    # ------------------------------------------------------------------ core
    def initial_state(self, num_envs: int, device) -> torch.Tensor:
        if not self.recurrent:
            return torch.zeros(1, num_envs, 1, device=device)
        return torch.zeros(self.rnn.num_layers, num_envs, self.rnn.hidden_size,
                           device=device)

    def get_states(self, x, state, done):
        """Return (features, new_state).  ``x`` is (T*B, obs), ``done`` is (T*B,)."""
        hidden = self.trunk(x)
        if not self.recurrent:
            return hidden, state

        batch_size = state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.rnn.input_size))
        done = done.reshape((-1, batch_size))
        out = []
        for h, d in zip(hidden, done):
            # Zeroing the carried state at episode boundaries is what makes a single
            # forward pass over a full rollout equivalent to per-episode unrolling.
            h, state = self.rnn(h.unsqueeze(0), (1.0 - d).view(1, -1, 1) * state)
            out.append(h)
        return torch.flatten(torch.cat(out), 0, 1), state

    def get_value(self, x, state, done):
        feat, _ = self.get_states(x, state, done)
        return self.critic(feat)

    def get_action_and_value(self, x, state, done, action=None, deterministic=False):
        feat, new_state = self.get_states(x, state, done)
        mean = self.actor_mean(feat)
        if deterministic and action is None:
            return mean, None, None, self.critic(feat), new_state
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(1), dist.entropy().sum(1),
                self.critic(feat), new_state)


class RunningMeanStd:
    """Welford-style running normaliser (matches the Gymnasium wrappers' behaviour)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"])
        self.var = np.asarray(d["var"])
        self.count = float(d["count"])
