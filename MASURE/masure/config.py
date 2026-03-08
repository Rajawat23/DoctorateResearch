"""
config.py
---------
Hyperparameter configuration for the three environments and five agents from the
MASURE paper.  All values match Table II of the paper exactly.

Environments: LunarLander-v2, MountainCar-v0, CartPole-v1
Agents: DQN, BootstrapDQN, SunriseDQN, IVDQN, MasksembleDQN (MASURE)

MuJoCo environments, SAC variants, and internal experiment variants are excluded.
"""

config = {
    "LunarLander-v2": {
        "DQN": {
            "eff_batch_size": 128,
            "eps_decay": 0.99,
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0005,
        },
        "BootstrapDQN": {
            "eff_batch_size": 64,
            "eps_decay": 0.99,
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0005,
            "mask_prob": 0.9,
            "prior_scale": 10,
        },
        "SunriseDQN": {
            "eff_batch_size": 64,
            "eps_decay": 0.99,
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0005,
            "mask_prob": 0.9,
            "prior_scale": 10,
            "sunrise_temp": 20.0,
        },
        "IVDQN": {
            "eff_batch_size": 64,
            "lr": 0.0005,
            "gamma": 0.99,
            "tau": 0.005,
            "mask_prob": 0.5,
            "num_nets": 5,
            "dynamic_xi": True,
            "minimal_eff_bs": 48,
        },
        "MasksembleDQN": {
            "eff_batch_size": 128,
            "eps_decay": 0.99,
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0005,
            "no_masks": 4,
            "scale": 2,
        },
    },
    "MountainCar-v0": {
        "DQN": {
            "eff_batch_size": 256,
            "lr": 0.001,
            "eps_decay": 0.98,
            "tau": 0.01,
        },
        "BootstrapDQN": {
            "eff_batch_size": 256,
            "lr": 0.001,
            "eps_decay": 0.98,
            "tau": 0.05,
            "mask_prob": 0.5,
            "prior_scale": 10,
        },
        "SunriseDQN": {
            "eff_batch_size": 256,
            "lr": 0.001,
            "eps_decay": 0.98,
            "tau": 0.05,
            "mask_prob": 0.5,
            "prior_scale": 10,
            "sunrise_temp": 50,
        },
        "IVDQN": {
            "eff_batch_size": 64,
            "lr": 0.001,
            "tau": 0.05,
            "mask_prob": 0.5,
            "num_nets": 5,
            "dynamic_xi": True,
            "minimal_eff_bs": 48,
        },
        "MasksembleDQN": {
            "eff_batch_size": 128,
            "eps_decay": 0.98,
            "tau": 0.005,
            "lr": 0.0005,
            "no_masks": 4,
            "scale": 2,
        },
    },
    "CartPole-v1": {
        "DQN": {
            "eff_batch_size": 128,
            "eps_decay": 0.99,
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0005,
        },
        "BootstrapDQN": {
            "eff_batch_size": 64,
            "mask_prob": 0.9,
            "prior_scale": 10,
        },
        "SunriseDQN": {
            "eff_batch_size": 64,
            "mask_prob": 0.9,
            "prior_scale": 10,
        },
        "IVDQN": {
            "eff_batch_size": 64,
            "mask_prob": 0.5,
            "num_nets": 5,
            "dynamic_xi": True,
            "minimal_eff_bs": 48,
        },
        "MasksembleDQN": {
            "eff_batch_size": 128,
            "no_masks": 4,
            "scale": 2,
        },
    },
}
