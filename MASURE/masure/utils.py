"""
utils.py
--------
Shared utilities for MASURE training: replay buffers, inverse-variance weighting,
and argparse helpers.

Classes
-------
ReplayBuffer      -- Fixed-size circular buffer for (s, a, r, s', done) transitions.
MaskReplayBuffer  -- Extends ReplayBuffer with a Bernoulli mask field per transition,
                     used by BootstrapDQNAgent to create ensemble diversity.

Functions
---------
str2bool              -- Argparse boolean parsing helper.
compute_eff_bs        -- Effective batch size: 1 / sum(w_i^2).
get_iv_weights        -- Inverse-variance weights normalized to sum to 1.
get_optimal_xi        -- Nelder-Mead solver for the variance offset xi used by IV-DQN.
"""

import random
import numpy as np
import torch
from collections import namedtuple, deque
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Argparse helper
# ---------------------------------------------------------------------------

def str2bool(v):
    """Parse a boolean value from a string argument.

    Accepts: yes/true/t/y/1  ->  True
             no/false/f/n/0  ->  False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected, got: %s" % v)


# ---------------------------------------------------------------------------
# Inverse-variance weighting utilities (used by IV-DQN)
# ---------------------------------------------------------------------------

def compute_eff_bs(weights):
    """Compute the effective batch size given normalized weights.

    Effective batch size = 1 / sum(w_i^2).

    A uniform weight distribution yields eff_bs equal to the batch size.
    Highly skewed weights yield a small eff_bs, signalling poor coverage.

    Parameters
    ----------
    weights : array-like of shape [N]
        Normalized importance weights (must sum to 1).

    Returns
    -------
    float
    """
    return 1.0 / np.sum([w ** 2 for w in weights])


def get_iv_weights(variances):
    """Compute inverse-variance weights normalized to sum to 1.

    w_i = (1 / var_i) / sum_j (1 / var_j)

    Parameters
    ----------
    variances : np.ndarray of shape [N]
        Per-sample variance estimates. Must be strictly positive.

    Returns
    -------
    np.ndarray of shape [N]
    """
    weights = 1.0 / variances
    weights = weights / np.sum(weights)
    return weights


def get_optimal_xi(variances, minimal_size, epsilon_start):
    """Find the variance offset xi such that effective batch size >= minimal_size.

    Uses Nelder-Mead optimization.  The offset xi is added to all variances before
    computing inverse-variance weights, smoothing extremely skewed distributions.

    IV-DQN applies this to guarantee a minimum effective batch size during training,
    which stabilizes gradient updates when one bootstrap head has much lower variance
    than the others.

    Parameters
    ----------
    variances : np.ndarray of shape [N]
    minimal_size : int
        Target minimum effective batch size.
    epsilon_start : float
        Starting value for xi (warm-start from the previous step).

    Returns
    -------
    float
        Optimal xi >= 0.
    """
    minimal_size = min(variances.shape[0] - 1, minimal_size)
    if compute_eff_bs(get_iv_weights(variances)) >= minimal_size:
        return 0.0

    def objective(x):
        return np.abs(
            compute_eff_bs(get_iv_weights(variances + np.abs(x))) - minimal_size
        )

    result = minimize(
        objective, 0, method="Nelder-Mead", options={"fatol": 1.0, "maxiter": 100}
    )
    xi = np.abs(result.x[0])
    return xi if xi is not None else 0.0


# ---------------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay.

    Stores (state, action, reward, next_state, done) tuples and returns random
    mini-batches as stacked PyTorch tensors on the specified device.

    Parameters
    ----------
    opt : argparse.Namespace
        Must contain: buffer_size (int), batch_size (int).
    action_size : int
    seed : int
    device : torch.device
    """

    def __init__(self, opt, action_size, seed, device):
        self.opt = opt
        self.action_size = action_size
        self.memory = deque(maxlen=opt.buffer_size)
        self.batch_size = opt.batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done, mask=None):
        """Add one transition to the buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, sample_size=None):
        """Return a random mini-batch of transitions as PyTorch tensors.

        Returns
        -------
        tuple of (states, actions, rewards, next_states, dones)
            Each tensor is on self.device.
        """
        if sample_size is None:
            sample_size = self.batch_size
        sample_size = min(sample_size, len(self.memory))
        experiences = random.sample(self.memory, k=sample_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack(
                [e.done for e in experiences if e is not None]
            ).astype(np.uint8)
        ).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class MaskReplayBuffer(ReplayBuffer):
    """Extends ReplayBuffer with a per-transition Bernoulli mask.

    The mask is a binary vector of length num_nets.  Transitions are only used
    to update the i-th bootstrap head if mask[i] == 1, which creates statistical
    diversity across ensemble members (BootstrapDQN / RPF paper).

    Parameters
    ----------
    opt : argparse.Namespace
        Must contain: buffer_size, batch_size, num_nets.
    action_size : int
    seed : int
    device : torch.device
    """

    def __init__(self, opt, action_size, seed, device):
        super().__init__(opt, action_size, seed, device)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done", "mask"],
        )

    def add(self, state, action, reward, next_state, done, mask=None):
        """Add a masked transition to the buffer."""
        e = self.experience(state, action, reward, next_state, done, mask)
        self.memory.append(e)

    def sample(self):
        """Return a random mini-batch including the mask tensors.

        Returns
        -------
        tuple of (states, actions, rewards, next_states, dones, masks)
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack(
                [e.done for e in experiences if e is not None]
            ).astype(np.uint8)
        ).float().to(self.device)
        masks = torch.from_numpy(
            np.vstack([e.mask for e in experiences if e is not None])
        ).bool().to(self.device)
        return (states, actions, rewards, next_states, dones, masks)
