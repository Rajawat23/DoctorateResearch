"""
masure
------
MASURE: Masksembles for Stable and Uncertainty-aware Reinforcement Learning Environments.

Reference: Singh, A., Ittoo, A., Vandomme, E., Ars, P.
"Uncertainty-Aware Reinforcement Learning Agents for Noisy Environments."

Public API
----------
    MASUREAgent        -- Core MASURE agent (Masksembles DQN with uncertainty-conscious
                          update rule, paper Equations 4-8).
    DQNAgent           -- Standard DQN baseline (paper Table II).
    BootstrapDQNAgent  -- BootstrapDQN + RPF baseline (paper Table II).
    SunriseDQNAgent    -- SUNRISE baseline (ICML 2021, paper Table II).
    IVDQNAgent         -- IV-RL baseline (ICLR 2022, paper Table II).
    QNetwork           -- Standard MLP backbone.
    Maskemble          -- MASURE Masksembles MLP.
    QNet_with_prior    -- RPF network for BootstrapDQN.
    ReplayBuffer       -- Standard replay buffer.
    MaskReplayBuffer   -- Bernoulli-masked replay buffer.
"""

from .masure_dqn import MASUREAgent
from .dqn import DQNAgent
from .baselines import BootstrapDQNAgent, SunriseDQNAgent, IVDQNAgent
from .networks import QNetwork, Maskemble, QNet_with_prior
from .utils import ReplayBuffer, MaskReplayBuffer

__all__ = [
    "MASUREAgent",
    "DQNAgent",
    "BootstrapDQNAgent",
    "SunriseDQNAgent",
    "IVDQNAgent",
    "QNetwork",
    "Maskemble",
    "QNet_with_prior",
    "ReplayBuffer",
    "MaskReplayBuffer",
]
