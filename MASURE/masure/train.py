"""
train.py
--------
Argparse-based training entry point for MASURE and all three baselines.

Supports the five agents and three environments from the paper.  Wandb has been
removed; metrics are written to a CSV file in ./results/.

Usage
-----
python -m masure.train --model MasksembleDQN --env LunarLander-v2 --num_episodes 400

Model registry
--------------
    DQN            -> DQNAgent
    BootstrapDQN   -> BootstrapDQNAgent
    SunriseDQN     -> SunriseDQNAgent
    IVDQN          -> IVDQNAgent   (IV-RL, ICLR 2022 baseline)
    MasksembleDQN  -> MASUREAgent  (core MASURE contribution)

Noise schedule (matching paper Section III-D and Table II defaults)
--------------------------------------------------------------------
    --noise_rate 0.01   : per-step probability of triggering a burst
    --scale      1.0    : Gaussian noise amplitude N(0, scale)
    --burst_length 500  : steps per burst
    --burst_cooldown 20000 : cool-down steps between bursts
    --num_episodes 400

Logging
-------
All metrics are written to ./results/{model}_{env}_{env_seed}_{net_seed}.csv.
Fields: episode, train_return, test_return, noise_scale, loss.
"""

import os
import sys
import argparse
import warnings
import gym

import torch

# Add the parent directory to the path so masure and noisyenv are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masure import DQNAgent, MASUREAgent, BootstrapDQNAgent, SunriseDQNAgent, IVDQNAgent
from masure.utils import str2bool
from masure.config import config
from noisyenv import StepBurstNoiseObservation, EpisodeAwareEnv

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

model_dict = {
    "DQN":           DQNAgent,
    "BootstrapDQN":  BootstrapDQNAgent,
    "SunriseDQN":    SunriseDQNAgent,
    "IVDQN":         IVDQNAgent,
    "MasksembleDQN": MASUREAgent,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="MASURE training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment and model
    parser.add_argument(
        "--env", type=str, default="LunarLander-v2",
        choices=["LunarLander-v2", "MountainCar-v0", "CartPole-v1"],
        help="Gym environment name.",
    )
    parser.add_argument(
        "--model", type=str, default="MasksembleDQN",
        choices=list(model_dict.keys()),
        help="Which agent to train.",
    )

    # Training
    parser.add_argument("--num_episodes", type=int, default=400)
    parser.add_argument("--max_t", type=int, default=1000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=0.99)

    # Optimization
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eff_batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=1)
    parser.add_argument("--end_reward", type=float, default=0.0)
    parser.add_argument("--failure_threshold", type=float, default=-200.0)

    # Seeds
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--net_seed", type=int, default=0)

    # Noise schedule (paper Section III-D defaults)
    parser.add_argument(
        "--noise_rate", type=float, default=0.01,
        help="Per-step probability of triggering a noise burst.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Gaussian noise amplitude N(0, scale).",
    )
    parser.add_argument(
        "--burst_length", type=int, default=500,
        help="Duration of each noise burst in environment steps.",
    )
    parser.add_argument(
        "--burst_cooldown", type=int, default=20000,
        help="Minimum steps between two noise bursts.",
    )
    parser.add_argument(
        "--start_episode", type=int, default=100,
        help="Episode after which noise can be triggered.",
    )

    # Masksembles-specific (MASURE)
    parser.add_argument("--no_masks", type=int, default=4)
    parser.add_argument("--mask_scale", type=float, default=2.0)
    parser.add_argument(
        "--method_combine_mask", type=str, default="avg",
        choices=["avg", "rnd", "best", "vote"],
    )

    # Ensemble-specific (BootstrapDQN / SUNRISE / IVDQN)
    parser.add_argument("--num_nets", type=int, default=5)
    parser.add_argument("--mask_prob", type=float, default=0.9)
    parser.add_argument("--prior_scale", type=float, default=10.0)
    parser.add_argument("--sunrise_temp", type=float, default=20.0)
    parser.add_argument(
        "--dynamic_xi", type=str2bool, nargs="?", const=True, default=False,
    )
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--minimal_eff_bs", type=float, default=32.0)
    parser.add_argument("--minimal_eff_bs_ratio", type=float, default=1.0)

    # Output
    parser.add_argument("--results_dir", type=str, default="./results")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    opt = parser.parse_args()

    # Apply config overrides from the paper's hyperparameter table
    try:
        cfg = config[opt.env][opt.model]
        for key, val in cfg.items():
            setattr(opt, key, val)
        # Re-parse CLI to allow user overrides to win over config defaults
        opt = parser.parse_args(namespace=opt)
    except KeyError:
        pass

    # Inflate batch_size to maintain effective batch size under masking
    opt.batch_size = int(getattr(opt, "eff_batch_size", opt.batch_size) / opt.mask_prob)
    opt.minimal_eff_bs = int(opt.minimal_eff_bs_ratio * getattr(opt, "eff_batch_size", opt.batch_size))

    # Alias --mask_scale to --scale (Maskemble parameter) without shadowing noise scale
    opt.no_masks = getattr(opt, "no_masks", 4)

    print("Options:", vars(opt))

    device = torch.device("cpu")

    # Build environment with burst noise wrapper (paper Section III-D)
    env = gym.make(opt.env)
    env.seed(opt.env_seed)
    env = StepBurstNoiseObservation(
        env,
        noise_rate=opt.noise_rate,
        scale=opt.scale,
        start_episode=opt.start_episode,
        burst_length=opt.burst_length,
        burst_cooldown=opt.burst_cooldown,
    )
    env = EpisodeAwareEnv(env)

    AgentClass = model_dict[opt.model]
    agent = AgentClass(env, opt, device=device)

    agent.train(
        n_episodes=opt.num_episodes,
        max_t=opt.max_t,
        eps_start=opt.eps_start,
        eps_end=opt.eps_end,
        eps_decay=opt.eps_decay,
        results_dir=opt.results_dir,
    )

    env.close()


if __name__ == "__main__":
    main()
