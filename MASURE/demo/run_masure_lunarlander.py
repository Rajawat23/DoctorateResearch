"""
run_masure_lunarlander.py
-------------------------
Self-contained demo: train MASUREAgent on LunarLander-v2 with the burst noise
schedule from the paper (Section III-D) and save results to ./results/.

No external logging dependency (no wandb).

Expected result
---------------
MASURE achieves higher and more stable returns than IV-DQN under the burst noise
schedule (paper Figure 1 and Figure 2).  Exact numbers depend on seed; over 400
episodes with the default parameters, MASURE typically reaches an average return
of ~150-200 on LunarLander-v2 under noise.

Usage
-----
    cd DoctorateResearch/MEASURE
    pip install -e masksembles-main/
    pip install -r requirements.txt
    python demo/run_masure_lunarlander.py

Output
------
    ./results/masure_lunarlander.csv
    Columns: episode, train_return, test_return, noise_scale, loss
"""

import os
import sys
import argparse
import types

import gym

import torch

# Make masure and noisyenv importable when run from demo/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masure import MASUREAgent
from noisyenv import StepBurstNoiseObservation, EpisodeAwareEnv

# ---------------------------------------------------------------------------
# Hyperparameters matching paper Table II (LunarLander-v2, MasksembleDQN row)
# ---------------------------------------------------------------------------

def _build_opt():
    opt = types.SimpleNamespace(
        # environment
        env="LunarLander-v2",
        env_seed=0,
        net_seed=0,
        # training
        num_episodes=400,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.99,
        # optimizer
        lr=0.0005,
        gamma=0.99,
        tau=0.005,
        # replay
        batch_size=128,
        eff_batch_size=128,
        buffer_size=int(1e5),
        update_every=1,
        mask_prob=1.0,        # no masking for MASURE (single network)
        # Masksembles (paper Table II)
        no_masks=4,
        scale=2,
        method_combine_mask="avg",
        # noise schedule (paper Section III-D, Table II)
        noise_rate=0.01,
        burst_length=500,
        burst_cooldown=20000,
        start_episode=100,
        noise_scale=1.0,
        # logging
        test_every=1,
        end_reward=0.0,
        failure_threshold=-200.0,
        # variance stabilization
        dynamic_xi=False,
        xi=1.0,
        minimal_eff_bs=32,
        minimal_eff_bs_ratio=1.0,
    )
    return opt


def main():
    parser = argparse.ArgumentParser(description="MASURE LunarLander-v2 demo")
    parser.add_argument("--num_episodes", type=int, default=400)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--net_seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    opt = _build_opt()
    opt.num_episodes = args.num_episodes
    opt.env_seed = args.env_seed
    opt.net_seed = args.net_seed

    print("Training MASUREAgent on LunarLander-v2")
    print("  no_masks=%d, scale=%d" % (opt.no_masks, opt.scale))
    print("  noise_rate=%.3f, burst_length=%d, burst_cooldown=%d" % (
        opt.noise_rate, opt.burst_length, opt.burst_cooldown
    ))
    print("  episodes=%d" % opt.num_episodes)

    # Build environment
    env = gym.make("LunarLander-v2")
    env.seed(opt.env_seed)
    env = StepBurstNoiseObservation(
        env,
        noise_rate=opt.noise_rate,
        scale=opt.noise_scale,
        start_episode=opt.start_episode,
        burst_length=opt.burst_length,
        burst_cooldown=opt.burst_cooldown,
    )
    env = EpisodeAwareEnv(env)

    device = torch.device("cpu")
    agent = MASUREAgent(env, opt, device=device)

    # Write results to a dedicated path so the demo CSV is clearly named
    os.makedirs(args.results_dir, exist_ok=True)
    agent.opt.env = "LunarLander-v2"

    agent.train(
        n_episodes=opt.num_episodes,
        max_t=opt.max_t,
        eps_start=opt.eps_start,
        eps_end=opt.eps_end,
        eps_decay=opt.eps_decay,
        results_dir=args.results_dir,
    )

    env.close()
    print("Done. Results saved to %s/" % args.results_dir)


if __name__ == "__main__":
    main()
