"""
dqn.py
------
Standard DQN baseline agent (DQNAgent).

This is the DQN baseline used in Tables II and III of the MASURE paper.
It implements experience replay, epsilon-greedy action selection, and soft
target-network updates (Polyak averaging).

Logging replaces all wandb calls with a simple CSV-based logger (log_metrics)
that appends per-episode metrics to a list and prints a summary every N episodes.

Classes
-------
DQNAgent  -- Standard DQN agent.

Excluded from this file
-----------------------
LossAttDQN: internal variance-attenuation variant not part of MASURE experiments.
"""

import os
import csv
import random
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from .networks import QNetwork
from .utils import ReplayBuffer


# ---------------------------------------------------------------------------
# Lightweight logging (replaces wandb)
# ---------------------------------------------------------------------------

class _CSVLogger:
    """Append-only CSV logger writing (episode, metric, ...) rows to a file."""

    def __init__(self, filepath, fieldnames):
        self._filepath = filepath
        self._fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row):
        with open(self._filepath, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writerow(row)


def log_metrics(episode, metrics_dict, logger=None, print_every=10):
    """Print metrics to stdout and optionally append to a CSV logger.

    Parameters
    ----------
    episode : int
    metrics_dict : dict
        Key-value pairs to log.
    logger : _CSVLogger or None
    print_every : int
        Print to stdout every this many episodes.
    """
    if logger is not None:
        logger.log({"episode": episode, **metrics_dict})
    if episode % print_every == 0:
        parts = ["episode=%d" % episode] + [
            "%s=%.4f" % (k, v) for k, v in metrics_dict.items()
        ]
        print(" | ".join(parts))


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Standard DQN agent with experience replay and soft target-network updates.

    Implements the DQN baseline from Table II of the MASURE paper.

    Parameters
    ----------
    env : gym.Env
        Initialized Gym environment (already wrapped with noise wrappers if needed).
    opt : argparse.Namespace
        Must contain: lr, net_seed, env_seed, tau, gamma, update_every, batch_size,
        buffer_size, test_every, eps_decay, failure_threshold, end_reward,
        num_episodes.
    device : torch.device, optional
        Default cpu.
    """

    def __init__(self, env, opt, device=None):
        self.env = env
        self.opt = opt
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.seed = random.seed(opt.env_seed)
        self.device = device or torch.device("cpu")
        self.test_scores = []
        self.loss = 0.0

        self.qnetwork_local = QNetwork(
            self.state_size, self.action_size, opt.net_seed
        ).to(self.device)
        self.qnetwork_target = QNetwork(
            self.state_size, self.action_size, opt.net_seed
        ).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        self.memory = ReplayBuffer(opt, self.action_size, 42, self.device)
        self.t_step = 0
        self.xi = 0.0

    def step(self, state, action, reward, next_state, done):
        """Store transition and trigger a learning step every update_every steps.

        Parameters
        ----------
        state, next_state : np.ndarray
        action : int
        reward : float
        done : bool

        Returns
        -------
        None  (DQNAgent does not return variance logs)
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.opt.update_every
        if self.t_step == 0 and len(self.memory) > self.opt.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.opt.gamma)
        return None

    def act(self, state, eps=0.0):
        """Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
        eps : float
            Exploration probability. eps=0 is greedy; eps=1 is fully random.

        Returns
        -------
        action : int
        mean_q : float
            Mean Q-value across actions (for logging).
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_t)
        self.qnetwork_local.train()
        action_values_np = action_values.cpu().data.numpy()
        if random.random() > eps:
            return int(np.argmax(action_values_np)), float(np.mean(action_values_np))
        return (
            int(random.choice(np.arange(self.action_size))),
            float(np.mean(action_values_np)),
        )

    def learn(self, experiences, gamma):
        """One gradient update step using a mini-batch of transitions.

        Computes MSE loss between expected Q-values and Bellman TD targets,
        then soft-updates the target network.

        Parameters
        ----------
        experiences : tuple of tensors
            (states, actions, rewards, next_states, dones)
        gamma : float
            Discount factor.
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        weights = (
            torch.ones(Q_expected.size()).to(self.device) / self.opt.batch_size
        )
        loss = self.weighted_mse(Q_expected, Q_targets, weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)

    def soft_update(self, local_model, target_model, tau):
        """Polyak-average target network weights.

        theta_target = tau * theta_local + (1 - tau) * theta_target

        Parameters
        ----------
        local_model : nn.Module
        target_model : nn.Module
        tau : float
        """
        for target_p, local_p in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_p.data.copy_(
                tau * local_p.data + (1.0 - tau) * target_p.data
            )

    def weighted_mse(self, inputs, targets, weights, mask=None):
        """Weighted mean-squared error.

        loss = sum(weights * (targets - inputs)^2)

        Parameters
        ----------
        inputs, targets : torch.Tensor of shape [B, 1]
        weights : torch.Tensor of shape [B, 1]
        mask : torch.Tensor or None

        Returns
        -------
        torch.Tensor scalar
        """
        loss = weights * ((targets - inputs) ** 2)
        if mask is not None:
            loss = loss * mask
        return loss.sum(0)

    def train(self, n_episodes=400, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, results_dir="./results"):
        """Main DQN training loop.

        Logs (episode, train_return, test_return, noise_scale) to a CSV file
        at results_dir/{model}_{env}_{env_seed}_{net_seed}.csv.

        Parameters
        ----------
        n_episodes : int
            Total training episodes. Paper default 400.
        max_t : int
            Maximum steps per episode.
        eps_start, eps_end : float
            Epsilon schedule bounds.
        eps_decay : float
            Multiplicative per-episode decay factor.
        results_dir : str
            Directory for the CSV log file.
        """
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            "DQN_%s_%d_%d.csv" % (
                getattr(self.opt, "env", "env"),
                getattr(self.opt, "env_seed", 0),
                getattr(self.opt, "net_seed", 0),
            ),
        )
        fieldnames = ["episode", "train_return", "test_return", "noise_scale", "loss"]
        logger = _CSVLogger(csv_path, fieldnames)

        scores_window = deque(maxlen=100)
        eps = eps_start

        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0.0
            ep_loss = []
            for t in range(max_t):
                action, _ = self.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    reward += getattr(self.opt, "end_reward", 0.0)
                score += reward
                ep_loss.append(self.loss)
                if done:
                    break

            scores_window.append(score)
            eps = max(eps_end, eps_decay * eps)

            if i_episode % getattr(self.opt, "test_every", 1) == 0:
                test_score = self.test(episode=i_episode)
            else:
                test_score = (
                    np.mean(self.test_scores[-1:]) if self.test_scores else 0.0
                )

            noise_scale = getattr(
                getattr(self.env, "env", self.env), "scale", 0.0
            )
            log_metrics(
                i_episode,
                {
                    "train_return": score,
                    "test_return": test_score,
                    "noise_scale": noise_scale,
                    "loss": float(np.mean(ep_loss)) if ep_loss else 0.0,
                },
                logger=logger,
                print_every=10,
            )

            if i_episode % 100 == 0:
                print(
                    "Episode %d  Average Score (last 100): %.2f"
                    % (i_episode, np.mean(scores_window))
                )

    def test(self, episode, num_trials=1, max_t=1000):
        """Greedy evaluation episode.

        Parameters
        ----------
        episode : int
            Current training episode (for logging).
        num_trials : int
            Number of evaluation rollouts.
        max_t : int
            Maximum steps per rollout.

        Returns
        -------
        float : episode return.
        """
        state = self.env.reset()
        score = 0.0
        for t in range(max_t):
            action, _ = self.act(state, eps=-1.0)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            score += reward
            if done:
                break
        self.test_scores.append(score)
        return score
