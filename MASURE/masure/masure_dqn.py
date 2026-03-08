"""
masure_dqn.py
-------------
MASUREAgent: the central algorithmic contribution of the MASURE paper.

Reference: Singh, A., Ittoo, A., Vandomme, E., Ars, P.
"Uncertainty-Aware Reinforcement Learning Agents for Noisy Environments."

The core idea is to integrate Masksembles-based epistemic uncertainty into
Q-learning so that the gradient update magnitude is automatically dampened when
the agent is uncertain.  This stabilizes training under consecutive noisy states
without additional hyperparameters.

Algorithm summary (paper Equations 4-8)
----------------------------------------
Given a mini-batch of transitions (s, a, r, s', done) and the Maskemble network
with n mask heads:

    Step 1 - Ensemble mean Q-estimate (Equation 4):
        Q_hat(s, a) = (1/n) * sum_{i=1}^{n} Q_i(s, a)

    Step 2 - Epistemic variance (Equation 5):
        sigma^2(s, a) = (1/n) * sum_{i=1}^{n} (Q_i(s,a) - Q_hat(s,a))^2

    Step 3 - Bellman target via Double DQN (Equation 6):
        y = r + gamma * max_{a'} Q_hat_target(s', a')
        (Action selected by local net; evaluated by target net.)

    Step 4 - TD error (Equation 7):
        delta = y - Q_hat(s, a)

    Step 5 - Uncertainty-conscious weight:
        weight = 1 / (1 + sigma^2)
        The denominator guarantees bounded weight values in [0, 1] without
        introducing additional hyperparameters.

    Step 6 - Per-head loss aggregation:
        loss = (1/n) * sum_{k=1}^{n} mean((Q_k(s,a) - y) * weight)^2

    Step 7 - Gradient clipping (max_norm=10) + Adam step.
    Step 8 - Soft update of target network.

Variance normalization detail
------------------------------
Before computing the uncertainty weight the implementation applies z-score
normalization followed by softplus:
    sigma = (sigma - sigma.mean()) / (sigma.std() + 1e-8)
    sigma = softplus(sigma)
    weight = 1 / (1 + sigma)
This prevents extreme weight values during early training when the variance
estimates are noisy and can be arbitrarily large or small.

Classes
-------
MASUREAgent  -- Masksembles DQN agent (renamed from MaskembleDQN for clarity).

Excluded
--------
network_predictions() and decision_function(): Ethias-specific offline inference
wrappers that assumed a private dataset format.  Not part of the public release.
"""

import os
import random
import numpy as np
from collections import Counter, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import Maskemble
from .utils import ReplayBuffer
from .dqn import DQNAgent, _CSVLogger, log_metrics


class MASUREAgent(DQNAgent):
    """Masksembles-based uncertainty-aware DQN agent (MASURE).

    Inherits the experience replay buffer, soft update, and training loop
    skeleton from DQNAgent. Overrides __init__, act, learn, and reduce_q_values
    to implement the MASURE uncertainty-conscious update rule.

    Parameters
    ----------
    env : gym.Env
        Initialized Gym environment (wrapped with noise wrappers).
    opt : argparse.Namespace
        Standard DQN options plus:
            no_masks (int)  : number of Masksembles heads. Paper default 4.
            scale    (float): mask scale controlling diversity. Paper default 2.
            method_combine_mask (str): how to reduce [B, n, A] -> [B, A] during
                act(). One of 'avg', 'vote', 'rnd', 'best'. Default 'avg'.
    device : torch.device, optional
        Default cpu.
    no_masks : int, optional
        Override for opt.no_masks if provided. Default 4.
    scale : float, optional
        Override for opt.scale if provided. Default 2.
    method_combine_mask : str, optional
        Override for opt.method_combine_mask. Default 'avg'.
    """

    def __init__(
        self,
        env,
        opt,
        device=None,
        no_masks=4,
        scale=2,
        method_combine_mask="avg",
    ):
        # Initialize parent to set env, opt, device, memory, t_step, xi, loss.
        # We immediately overwrite the Q-networks below.
        super().__init__(env, opt, device=device)

        self.no_masks = getattr(opt, "no_masks", no_masks)
        self.scale = getattr(opt, "scale", scale)
        self.method_combine_mask = getattr(
            opt, "method_combine_mask", method_combine_mask
        )

        self.qnetwork_local = Maskemble(
            self.state_size,
            self.action_size,
            seed=opt.net_seed,
            n_masks=self.no_masks,
            scale=self.scale,
        ).to(self.device)
        self.qnetwork_target = Maskemble(
            self.state_size,
            self.action_size,
            seed=opt.net_seed,
            n_masks=self.no_masks,
            scale=self.scale,
        ).to(self.device)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=opt.lr
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state, eps=0.0):
        """Epsilon-greedy action selection using mean Q across mask heads.

        Sets network to eval mode to obtain [1, n, A] output, averages across
        the mask dimension to get [1, A], then applies epsilon-greedy.

        Parameters
        ----------
        state : np.ndarray
        eps : float

        Returns
        -------
        action : int
        mean_q : float
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_t)        # [1, n, A] or [1, A]
            reduced_q = self.reduce_q_values(q_values)     # [1, A]
        self.qnetwork_local.train()

        reduced_q_np = reduced_q.cpu().data.numpy()
        if random.random() > eps:
            return int(np.argmax(reduced_q_np)), float(np.mean(reduced_q_np))
        return (
            int(random.choice(np.arange(self.action_size))),
            float(np.mean(reduced_q_np)),
        )

    # ------------------------------------------------------------------
    # Uncertainty-conscious update (Equations 4-8)
    # ------------------------------------------------------------------

    def learn(self, experiences, gamma):
        """MASURE uncertainty-conscious Q-update (Equations 4-8).

        Parameters
        ----------
        experiences : tuple of tensors
            (states, actions, rewards, next_states, dones)
        gamma : float
            Discount factor.
        """
        states, actions, rewards, next_states, dones = experiences
        N = self.no_masks

        # ---- Compute Double DQN targets in eval mode ----
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

        with torch.no_grad():
            # Action selection from local network (Double DQN)
            q_local_next = self.qnetwork_local(next_states.float())    # [B, n, A]
            q_local_next_mean = q_local_next.mean(dim=1)               # [B, A]
            next_actions = q_local_next_mean.argmax(dim=1, keepdim=True)  # [B, 1]

            # Action evaluation from target network
            q_targets_next = self.qnetwork_target(next_states.float()) # [B, n, A]
            q_targets_mean = q_targets_next.mean(dim=1)                # [B, A]
            max_q_next = q_targets_mean.gather(1, next_actions)        # [B, 1]

            # Bellman target (Equation 6)
            Q_targets = rewards + gamma * max_q_next * (1 - dones)    # [B, 1]

            # Epistemic variance across heads (Equation 5)
            q_targets_var = q_targets_next.var(dim=1, unbiased=False)  # [B, A]
            var_targets = q_targets_var.gather(1, next_actions)        # [B, 1]

        # ---- Forward pass for current states ----
        q_values = self.qnetwork_local(states.float())                 # [B, n, A]

        # ---- Variance normalization and uncertainty weight ----
        # z-score normalization then softplus for numerical stability during
        # early training when variance estimates are poorly calibrated.
        sigma = var_targets                                            # [B, 1]
        sigma = (sigma - sigma.mean()) / (sigma.std() + 1e-8)
        sigma = F.softplus(sigma)
        weights = 1.0 / (1.0 + sigma)                                 # [B, 1]

        # ---- Per-head loss aggregation (Step 6) ----
        loss = 0.0
        for k in range(N):
            q_k = q_values[:, k, :]                                    # [B, A]
            q_expected_k = q_k.gather(1, actions)                      # [B, 1]
            td_error_k = q_expected_k - Q_targets                      # [B, 1]
            weighted_td_k = td_error_k * weights                       # [B, 1]
            loss = loss + (weighted_td_k ** 2).mean()

        loss = loss / N

        # ---- Backward pass with gradient clipping ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.qnetwork_local.parameters() if p.grad is not None],
            max_norm=10,
        )
        self.optimizer.step()
        self.loss = loss.item()

        # ---- Soft target-network update ----
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)

    # ------------------------------------------------------------------
    # Q-value reduction strategies
    # ------------------------------------------------------------------

    def reduce_q_values(self, q_values):
        """Reduce [B, n, A] ensemble Q-values to [B, A].

        Supports four reduction strategies matching the paper's ablation:
            'avg'  (default): mean across mask heads. Used in all paper results.
            'vote'          : majority vote on argmax across heads.
            'rnd'           : select a random head per sample.
            'best'          : select the head with the highest max Q-value.

        If q_values is 2-D ([B, A]) it is returned unchanged (training mode
        Masksembles1D output when the module is in training()).

        Parameters
        ----------
        q_values : torch.Tensor of shape [B, n, A] or [B, A]

        Returns
        -------
        torch.Tensor of shape [B, A]
        """
        if q_values.dim() == 2:
            return q_values

        B, n, A = q_values.shape
        method = self.method_combine_mask

        if method == "avg":
            return q_values.mean(dim=1)

        if method == "rnd":
            indices = torch.randint(0, n, (B,), device=q_values.device)
            return q_values[torch.arange(B), indices]

        if method == "vote":
            votes = q_values.argmax(dim=2)    # [B, n]
            action_vecs = []
            for i in range(B):
                vote_counts = Counter(votes[i].tolist())
                selected = max(vote_counts, key=vote_counts.get)
                vec = torch.zeros(A, device=q_values.device)
                vec[selected] = 1.0
                action_vecs.append(vec)
            return torch.stack(action_vecs)

        if method == "best":
            best_q_vals, _ = q_values.max(dim=2)   # [B, n]
            _, best_mask = best_q_vals.max(dim=1)  # [B]
            return q_values[torch.arange(B), best_mask]

        raise NotImplementedError("Unknown reduce method: %s" % method)

    # ------------------------------------------------------------------
    # Training loop (overrides DQNAgent to write MASURE CSV)
    # ------------------------------------------------------------------

    def train(self, n_episodes=400, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, results_dir="./results"):
        """MASURE training loop with CSV logging.

        Parameters
        ----------
        n_episodes : int
            Total training episodes. Paper default 400.
        max_t : int
            Maximum steps per episode.
        eps_start, eps_end : float
            Epsilon schedule bounds.
        eps_decay : float
            Multiplicative per-episode decay.
        results_dir : str
            Directory for CSV output.
        """
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            "MasksembleDQN_%s_%d_%d.csv" % (
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
