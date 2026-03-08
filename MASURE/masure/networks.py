"""
networks.py
-----------
Neural network architectures for MASURE and DQN baselines.

Two architectures are provided:

    QNetwork    -- Standard 2-hidden-layer MLP used as the backbone for DQN and all
                   ensemble baselines (Table II, hidden layers = [64, 64]).

    Maskemble   -- The core MASURE network. A single MLP with two Masksembles1D layers
                   that produces an ensemble of diverse sub-models within one forward
                   pass (paper Section III-A, Table I).

Two utility networks for the BootstrapDQN baseline with Randomized Prior Functions:

    PriorNet          -- Fixed random prior network.
    QNet_with_prior   -- Learnable network plus scaled prior (paper: RPF).

Train vs eval mode distinction for Maskemble
--------------------------------------------
Masksembles1D behaves differently depending on the module mode:

    training mode  : returns [B, D]   -- standard batch processing; masks are applied
                     by splitting the batch across sub-models.
    eval mode      : returns [B, n, D] -- each sample is processed by all n sub-models
                     simultaneously, producing one output vector per mask.

Because nn.Linear operates on the last dimension, the forward pass handles the 3-D
tensor by reshaping [B, n, D] -> [B*n, D], applying the linear layer, then reshaping
back to [B, n, D]. The final output in eval mode is [B, n, action_size], which is
used downstream to compute per-head Q-values and the epistemic variance across heads
(paper Equation 5).

Default mask parameters
-----------------------
n_masks=4, scale=2 match paper Table II.  The scale parameter controls mask overlap:
lower overlap yields more diverse sub-models, matching the statistical properties of
deep ensembles at a fraction of the parameter count (paper Table I shows MASURE uses
~6.02K parameters versus 24-26K for deep ensemble baselines).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from masksembles.torch import Masksembles1D


class QNetwork(nn.Module):
    """Standard 2-hidden-layer MLP for DQN and ensemble baselines.

    Architecture (paper Table II, hidden layers = [64, 64]):
        fc1  : Linear(state_size, fc1_units)
        fc2  : Linear(fc1_units, fc2_units)
        fc3  : Linear(fc2_units, action_size)
        ReLU activations after fc1 and fc2; no activation on the output layer.

    Parameters
    ----------
    state_size : int
        Dimension of the observation vector.
    action_size : int
        Number of discrete actions.
    seed : int
        Random seed for reproducibility.
    fc1_units : int, optional
        Width of the first hidden layer. Default 64.
    fc2_units : int, optional
        Width of the second hidden layer. Default 64.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Map state -> action Q-values.

        Parameters
        ----------
        state : torch.Tensor of shape [B, state_size]

        Returns
        -------
        torch.Tensor of shape [B, action_size]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Maskemble(nn.Module):
    """MASURE network: MLP with Masksembles1D layers for ensemble uncertainty.

    Architecture (paper Section III-A and Table I):
        fc1          : Linear(state_size, fc1_units)
        masksembles1 : Masksembles1D(fc1_units, n_masks, scale)
        ReLU
        fc2          : Linear(fc1_units, fc2_units)
        masksembles2 : Masksembles1D(fc2_units, n_masks, scale)
        ReLU
        fc3          : Linear(fc2_units, action_size)

    Default parameters from paper Table II: n_masks=4, scale=2, fc1=fc2=64.

    Train vs eval mode
    ------------------
    In training mode Masksembles1D returns [B, D]; the forward pass is identical to a
    standard MLP and the output is [B, action_size].

    In eval mode Masksembles1D returns [B, n, D].  Each nn.Linear layer handles the
    3-D tensor by reshaping to [B*n, D], applying the linear transformation, then
    reshaping back to [B, n, D].  The final output is [B, n, action_size], which
    allows the caller to compute:
        - per-head Q-values  (Equation 4 of the paper)
        - epistemic variance across heads  (Equation 5 of the paper)

    The scale parameter controls mask overlap and therefore ensemble diversity.
    Lower overlap yields more statistically independent sub-models, which improves
    uncertainty estimation without the full parameter cost of a deep ensemble.

    Parameters
    ----------
    state_size : int
        Dimension of the observation vector.
    action_size : int
        Number of discrete actions.
    seed : int
        Random seed.
    fc1_units : int, optional
        Width of the first hidden layer. Default 64.
    fc2_units : int, optional
        Width of the second hidden layer. Default 64.
    n_masks : int, optional
        Number of Masksembles sub-models. Paper default 4.
    scale : float, optional
        Mask scale (controls overlap / diversity). Paper default 2.
    """

    def __init__(
        self,
        state_size,
        action_size,
        seed=5,
        fc1_units=64,
        fc2_units=64,
        n_masks=4,
        scale=2,
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.n_masks = n_masks

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.masksembles1 = Masksembles1D(fc1_units, n_masks, scale)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.masksembles2 = Masksembles1D(fc2_units, n_masks, scale)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self._layers = [
            self.fc1,
            self.masksembles1,
            self.fc2,
            self.masksembles2,
            self.fc3,
        ]

    def _apply_linear(self, layer, x):
        """Apply a linear layer to a 2-D or 3-D tensor.

        For 3-D input [B, n, D]: reshape to [B*n, D], apply layer, reshape back to
        [B, n, out_features].
        For 2-D input [B, D]: standard linear application.
        """
        if x.dim() == 3:
            B, n, D = x.shape
            x = x.reshape(B * n, D)
            x = layer(x)
            x = x.reshape(B, n, -1)
        else:
            x = layer(x)
        return x

    def forward(self, obs):
        """Forward pass.

        Parameters
        ----------
        obs : torch.Tensor of shape [B, state_size]

        Returns
        -------
        In training mode : torch.Tensor of shape [B, action_size]
        In eval mode     : torch.Tensor of shape [B, n_masks, action_size]
        """
        out = obs.to(torch.float32)

        out = self._apply_linear(self.fc1, out)
        out = self.masksembles1(out)
        out = F.relu(out)

        out = self._apply_linear(self.fc2, out)
        out = self.masksembles2(out)
        out = F.relu(out)

        out = self._apply_linear(self.fc3, out)
        return out


class PriorNet(nn.Module):
    """Fixed random prior network for Randomized Prior Functions (RPF).

    A shallow linear network whose weights are frozen after initialization.
    Used by QNet_with_prior to inject a fixed random prior into the Q-function.

    Parameters
    ----------
    state_size : int
    action_size : int
    seed : int
    fc1_units : int, optional
        Width of the intermediate layer (unused in linear projection; kept for
        interface consistency). Default 64.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64):
        super(PriorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(state_size, action_size)

    def forward(self, state):
        return self.fc2(state)


class QNet_with_prior(nn.Module):
    """Learnable Q-network plus a scaled, fixed random prior (RPF).

    Used by BootstrapDQNAgent.  The prior network is not optimized; only
    ``self.net`` participates in gradient updates.

    Output: net(state) + prior_scale * prior(state)

    Parameters
    ----------
    state_size : int
    action_size : int
    seed : int
    prior_scale : float, optional
        Scaling factor for the prior. Paper default 10.
    fc1_units : int, optional
        Default 64.
    fc2_units : int, optional
        Default 64.
    """

    def __init__(
        self, state_size, action_size, seed, prior_scale=10, fc1_units=64, fc2_units=64
    ):
        super(QNet_with_prior, self).__init__()
        self.prior = PriorNet(state_size, action_size, seed, fc1_units)
        self.net = QNetwork(state_size, action_size, seed, fc1_units, fc2_units)
        self.prior_scale = prior_scale

    def forward(self, state):
        return self.net(state) + self.prior_scale * self.prior(state)
