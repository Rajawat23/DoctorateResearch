"""
baselines.py
------------
The three ensemble DQN baselines compared against MASURE in Tables II and III
of the paper.

Classes
-------
BootstrapDQNAgent   -- BootstrapDQN with Randomized Prior Functions (RPF).
SunriseDQNAgent     -- SUNRISE (ICML 2021) UCB-weighted Bellman backup.
IVDQNAgent          -- IV-RL (ICLR 2022), inverse-variance weighted bootstrap.

All wandb calls from the original source have been removed and replaced with
the same CSV-based log_metrics used by DQNAgent.

Excluded from this file
-----------------------
All SAC-related classes (SACTrainer, IV_VarSAC, EnsembleSAC, etc.).
All MuJoCo-specific variants.
Internal variants: UWAC, LossAtt, VarEnsemble, MCDropout DQN.
These are not part of the published MASURE experiments.
"""

import os
import random
import numpy as np
from collections import Counter, deque

import torch
import torch.optim as optim

from .networks import QNetwork, QNet_with_prior
from .utils import (
    MaskReplayBuffer,
    ReplayBuffer,
    compute_eff_bs,
    get_iv_weights,
    get_optimal_xi,
)
from .dqn import DQNAgent, _CSVLogger, log_metrics


# ---------------------------------------------------------------------------
# Shared bootstrap / mask ensemble base
# ---------------------------------------------------------------------------

class _MaskEnsembleBase(DQNAgent):
    """Internal base for all mask-ensemble agents (not exported).

    Manages an ensemble of num_nets independent QNetworks, each with its own
    Adam optimizer.  Uses a MaskReplayBuffer so that each transition stores a
    Bernoulli mask determining which heads are updated.

    Per-episode network selection (bootstrap exploration) is implemented in
    BootstrapDQNAgent.train().
    """

    def __init__(self, env, opt, device=None):
        super().__init__(env, opt, device=device)

        self.mask = True
        self.random_state = np.random.RandomState(11)

        # Overwrite single-network memory with mask-aware buffer
        self.memory = MaskReplayBuffer(opt, self.action_size, 42, self.device)

        num_nets = getattr(opt, "num_nets", 5)
        self.qnets = []
        self.target_nets = []
        self.optims = []
        for i in range(num_nets):
            net = QNetwork(
                self.state_size, self.action_size, seed=i + opt.net_seed
            ).to(self.device)
            tgt = QNetwork(
                self.state_size, self.action_size, seed=i + opt.net_seed
            ).to(self.device)
            self.qnets.append(net)
            self.target_nets.append(tgt)
            self.optims.append(optim.Adam(net.parameters(), lr=opt.lr))

    @property
    def num_nets(self):
        return len(self.qnets)

    def step(self, state, action, reward, next_state, done):
        mask = self.random_state.binomial(
            1, getattr(self.opt, "mask_prob", 0.9), self.num_nets
        )
        self.memory.add(state, action, reward, next_state, done, mask)
        self.t_step = (self.t_step + 1) % self.opt.update_every
        if self.t_step == 0 and len(self.memory) > self.opt.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.opt.gamma)
        return None

    def _get_mse_weights(self, variance):
        """Return uniform weights (override in subclasses for SUNRISE / IV)."""
        return torch.ones(variance.size()).to(self.device) / variance.size(0)

    def learn(self, experiences, gamma):
        """Masked ensemble update shared by BootstrapDQN and SUNRISE."""
        states, actions, rewards, next_states, dones, masks = experiences
        masks = masks.unsqueeze(2)   # [B, num_nets, 1]

        Q_targets_next = torch.stack(
            [self.target_nets[i](next_states).detach() for i in range(self.num_nets)]
        )                                                                  # [N, B, A]
        Q_targets = torch.stack(
            [
                rewards + gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)
                for i in range(self.num_nets)
            ]
        )                                                                  # [N, B, 1]
        next_actions_ind = torch.stack(
            [Q_targets_next[i].max(1)[1].unsqueeze(1) for i in range(self.num_nets)]
        )                                                                  # [N, B, 1]

        Q_targets_next_mean = Q_targets_next.mean(0)                      # [B, A]
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)         # [B, A]

        for i in range(self.num_nets):
            mask_i = masks[:, i, 0]                                       # [B]
            Q_expected = self.qnets[i](states).gather(1, actions)[mask_i] # [M, 1]
            Q_target = Q_targets[i][mask_i]                               # [M, 1]
            Q_var = Q_targets_next_var.gather(1, next_actions_ind[i])[mask_i]

            weights = self._get_mse_weights(Q_var)
            loss = (weights * (Q_expected - Q_target) ** 2).sum(0)

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()

            self.loss = loss.item()

        for i in range(self.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)

    def act(self, state, eps=0.0, net_index=None):
        """Epsilon-greedy action.  Uses majority vote in eval, single net in train."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        for net in self.qnets:
            net.eval()
        with torch.no_grad():
            Q_ensemble = np.array(
                [net(state_t).cpu().data.numpy() for net in self.qnets]
            )                                                             # [N, 1, A]
        for net in self.qnets:
            net.train()

        if net_index is not None:
            action = int(np.argmax(Q_ensemble[net_index]))
        else:
            actions = [int(np.argmax(Q)) for Q in Q_ensemble]
            data = Counter(actions)
            action = data.most_common(1)[0][0]

        if random.random() > eps:
            return action, float(np.mean(Q_ensemble))
        return int(random.choice(np.arange(self.action_size))), float(np.mean(Q_ensemble))


# ---------------------------------------------------------------------------
# BootstrapDQN with Randomized Prior Functions (RPF)
# ---------------------------------------------------------------------------

class BootstrapDQNAgent(_MaskEnsembleBase):
    """BootstrapDQN with Randomized Prior Functions (RPF).

    Each ensemble member is a QNet_with_prior: a learnable QNetwork plus a fixed
    random prior scaled by prior_scale.  Bootstrap exploration selects one head per
    episode for action selection; all heads are updated using Bernoulli masks.

    Paper hyperparameters (Table II, LunarLander-v2):
        batch=64, lr=0.0005, mask_prob=0.9, prior_scale=10, ensemble_size=5.

    Parameters
    ----------
    env : gym.Env
    opt : argparse.Namespace
        Must contain: lr, net_seed, num_nets, mask_prob, prior_scale, tau, gamma,
        update_every, batch_size, buffer_size, test_every.
    device : torch.device, optional
    """

    def __init__(self, env, opt, device=None):
        super().__init__(env, opt, device=device)

        prior_scale = getattr(opt, "prior_scale", 10)

        # Replace plain QNetworks with RPF networks
        self.qnets = []
        self.target_nets = []
        self.optims = []
        for i in range(getattr(opt, "num_nets", 5)):
            net = QNet_with_prior(
                self.state_size, self.action_size,
                seed=i + opt.net_seed, prior_scale=prior_scale
            ).to(self.device)
            tgt = QNet_with_prior(
                self.state_size, self.action_size,
                seed=i + opt.net_seed, prior_scale=prior_scale
            ).to(self.device)
            self.qnets.append(net)
            self.target_nets.append(tgt)
            # Only optimize the learnable sub-network, not the frozen prior
            self.optims.append(optim.Adam(net.net.parameters(), lr=opt.lr))

    def train(self, n_episodes=400, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, results_dir="./results"):
        """Bootstrap DQN training loop with per-episode head selection."""
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            "BootstrapDQN_%s_%d_%d.csv" % (
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
            curr_net = random.choice(range(self.num_nets))

            for t in range(max_t):
                action, _ = self.act(state, eps, net_index=curr_net)
                next_state, reward, done, _ = self.env.step(action)
                clean_obs = (
                    self.env.get_clean_observation()
                    if hasattr(self.env, "get_clean_observation")
                    else state
                )
                self.step(clean_obs, action, reward, next_state, done)
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
                test_score = np.mean(self.test_scores[-1:]) if self.test_scores else 0.0

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

    def test(self, episode, max_t=1000):
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


# ---------------------------------------------------------------------------
# SUNRISE (ICML 2021)
# ---------------------------------------------------------------------------

class SunriseDQNAgent(BootstrapDQNAgent):
    """SUNRISE: UCB-based weighted Bellman backup for BootstrapDQN.

    Computes weights as sigmoid(-sqrt(variance) * temperature) + 0.5.
    High-variance bootstrap heads are down-weighted in the Bellman backup,
    making the agent more conservative under uncertainty.

    Paper hyperparameters (Table II, LunarLander-v2):
        batch=64, lr=0.0005, mask_prob=0.9, prior_scale=10, ensemble_size=5,
        sunrise_temp=20.0.

    Parameters
    ----------
    env : gym.Env
    opt : argparse.Namespace
        Must contain all BootstrapDQNAgent parameters plus:
            sunrise_temp (float): weighting temperature. Paper default 20.0.
    device : torch.device, optional
    """

    def __init__(self, env, opt, device=None):
        super().__init__(env, opt, device=device)
        self.sunrise_temp = getattr(opt, "sunrise_temp", 20.0)

    def _get_mse_weights(self, variance):
        """SUNRISE UCB weights: sigmoid(-sqrt(var) * temp) + 0.5."""
        weights = torch.sigmoid(-torch.sqrt(variance) * self.sunrise_temp) + 0.5
        return weights

    def train(self, n_episodes=400, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, results_dir="./results"):
        """SUNRISE training loop (inherits bootstrap head selection)."""
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            "SunriseDQN_%s_%d_%d.csv" % (
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
            curr_net = random.choice(range(self.num_nets))

            for t in range(max_t):
                action, _ = self.act(state, eps, net_index=curr_net)
                next_state, reward, done, _ = self.env.step(action)
                clean_obs = (
                    self.env.get_clean_observation()
                    if hasattr(self.env, "get_clean_observation")
                    else state
                )
                self.step(clean_obs, action, reward, next_state, done)
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
                test_score = np.mean(self.test_scores[-1:]) if self.test_scores else 0.0

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


# ---------------------------------------------------------------------------
# IV-RL (ICLR 2022)
# ---------------------------------------------------------------------------

class IVDQNAgent(_MaskEnsembleBase):
    """IV-RL: inverse-variance weighted BootstrapDQN (ICLR 2022 baseline).

    Uses inverse-variance weighting of bootstrap heads combined with a Gaussian
    mixture variance target.  The variance offset xi is solved dynamically via
    Nelder-Mead (get_optimal_xi) to guarantee a minimum effective batch size
    during training.

    Paper hyperparameters (Table II, LunarLander-v2):
        batch=64, lr=0.0005, gamma=0.99, tau=0.005, mask_prob=0.5,
        ensemble_size=5, min_batch=48 (dynamic xi).

    Parameters
    ----------
    env : gym.Env
    opt : argparse.Namespace
        Must contain all _MaskEnsembleBase parameters plus:
            dynamic_xi (bool)      : whether to compute xi dynamically. Default False.
            minimal_eff_bs (int)   : target minimum effective batch size.
            xi (float)             : initial xi when dynamic_xi=False. Default 1.0.
    device : torch.device, optional
    """

    def __init__(self, env, opt, device=None):
        super().__init__(env, opt, device=device)
        self.xi = getattr(opt, "xi", 1.0)

    def _get_mse_weights(self, variance):
        """Inverse-variance weights normalized to sum to 1."""
        weights = 1.0 / (variance + self.xi)
        weights = weights / weights.sum(0)
        return weights

    def _update_xi(self, variance_np):
        """Optionally update xi via Nelder-Mead."""
        if getattr(self.opt, "dynamic_xi", False):
            self.xi = get_optimal_xi(
                variance_np,
                getattr(self.opt, "minimal_eff_bs", 32),
                self.xi,
            )

    def learn(self, experiences, gamma):
        """IV-DQN masked ensemble update with dynamic xi."""
        states, actions, rewards, next_states, dones, masks = experiences
        masks = masks.unsqueeze(2)

        Q_targets_next = torch.stack(
            [self.target_nets[i](next_states).detach() for i in range(self.num_nets)]
        )
        Q_targets = torch.stack(
            [
                rewards + gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)
                for i in range(self.num_nets)
            ]
        )
        next_actions_ind = torch.stack(
            [Q_targets_next[i].max(1)[1].unsqueeze(1) for i in range(self.num_nets)]
        )

        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)

        for i in range(self.num_nets):
            mask_i = masks[:, i, 0]
            Q_expected = self.qnets[i](states).gather(1, actions)[mask_i]
            Q_target = Q_targets[i][mask_i]
            Q_var = Q_targets_next_var.gather(1, next_actions_ind[i])[mask_i]

            self._update_xi(Q_var.detach().cpu().numpy())
            weights = self._get_mse_weights(Q_var)
            loss = (weights * (Q_expected - Q_target) ** 2).sum(0)

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            self.loss = loss.item()

        for i in range(self.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)

    def train(self, n_episodes=400, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, results_dir="./results"):
        """IV-DQN training loop."""
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            "IVDQNAgent_%s_%d_%d.csv" % (
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
            curr_net = random.choice(range(self.num_nets))

            for t in range(max_t):
                action, _ = self.act(state, eps, net_index=curr_net)
                next_state, reward, done, _ = self.env.step(action)
                clean_obs = (
                    self.env.get_clean_observation()
                    if hasattr(self.env, "get_clean_observation")
                    else state
                )
                self.step(clean_obs, action, reward, next_state, done)
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
                test_score = np.mean(self.test_scores[-1:]) if self.test_scores else 0.0

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

    def test(self, episode, max_t=1000):
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
