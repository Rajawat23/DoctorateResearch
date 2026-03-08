"""
wrappers.py
-----------
Gym observation wrappers implementing the noise schedules from the MASURE paper
(Section III-D).

The three wrappers used in published MASURE experiments are:

    EpisodeAwareEnv              -- Wrapper that propagates the episode count to
                                    child wrappers so they can activate noise only
                                    after a warm-up period.

    RandomNormalNoisyObservation -- Base noise wrapper. Adds Gaussian N(loc, scale)
                                    noise with probability noise_rate per step, but
                                    only after start_episode episodes have elapsed.

    StepBurstNoiseObservation    -- The specific noise schedule from the MASURE paper.
                                    Implements rare but severe burst-style noise to
                                    test catastrophic forgetting resistance:
                                      - 100 warm-up episodes (no noise).
                                      - After warm-up, each step has probability
                                        noise_rate=0.01 of triggering a burst.
                                      - A burst lasts burst_length=500 steps and
                                        crosses episode boundaries.
                                      - A cool-down of burst_cooldown=20000 steps
                                        must pass before the next burst.
                                      - Noise amplitude: N(0, 1).
                                    Unlike prior work that tests mild or short-lived
                                    noise, this schedule introduces sustained, severe
                                    perturbations to evaluate resistance to catastrophic
                                    forgetting.

Additional wrappers (RandomMixupObservation, RandomDropoutObservation, etc.) that
were used in exploratory experiments but are not part of the published MASURE
evaluation are available in noisyenv/extras.py.

Debug print statements from the original source have been replaced with
logging.debug(...) calls to avoid flooding stdout during 400-episode runs.
"""

import logging
import gym
import numpy as np

logger = logging.getLogger(__name__)


class EpisodeAwareEnv(gym.Wrapper):
    """Wrapper that tracks episode count and propagates it to child wrappers.

    At each reset(), calls set_episode(episode) on the immediate child wrapper
    (if the method exists) so that noise wrappers can activate or deactivate
    noise based on the current episode number.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.  Typically already wrapped with a noise wrapper.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode = 0

    def reset(self, **kwargs):
        if hasattr(self.env, "set_episode"):
            self.env.set_episode(self.episode)
        self.episode += 1
        return super().reset(**kwargs)


class RandomNormalNoisyObservation(gym.ObservationWrapper):
    """Adds Gaussian noise to observations with probability noise_rate per step.

    Noise is only applied after start_episode episodes have elapsed.  The episode
    counter is updated externally by EpisodeAwareEnv via set_episode().

    Parameters
    ----------
    env : gym.Env
    noise_rate : float, optional
        Probability of adding noise to the observation each step. Default 0.05.
    loc : float, optional
        Mean of the Gaussian noise distribution. Default 0.0.
    scale : float, optional
        Standard deviation of the noise. Default 0.8.
    start_episode : int, optional
        Episode number at which noise is first enabled. Default 100.
    """

    def __init__(self, env, noise_rate=0.05, loc=0.0, scale=0.8, start_episode=100):
        super().__init__(env)
        self.noise_rate = noise_rate
        self.loc = loc
        self.base_scale = scale
        self.scale = 0.0
        self.start_episode = start_episode
        self._current_episode = 0

    def set_episode(self, episode):
        """Update the internal episode counter and activate noise if past warm-up."""
        self._current_episode = episode
        self.scale = self.base_scale if episode >= self.start_episode else 0.0

    def _observation(self, observation):
        if self.scale > 0.0 and np.random.rand() <= self.noise_rate:
            observation = observation + np.random.normal(
                loc=self.loc, scale=self.scale, size=observation.shape
            )
        return observation


class StepBurstNoiseObservation(RandomNormalNoisyObservation):
    """Rare but severe burst-style noise schedule from the MASURE paper (Sec. III-D).

    Noise schedule:
        1. Warm-up: the first start_episode episodes (default 100) are noise-free.
        2. After warm-up: each environment step has probability noise_rate of
           triggering a burst event.
        3. A burst lasts burst_length consecutive steps, crossing episode boundaries.
        4. After a burst ends, burst_cooldown steps must elapse before a new burst
           can begin.
        5. Burst noise amplitude: N(loc, scale).

    This schedule is designed to test catastrophic forgetting: unlike mild or
    step-wise noise, a sustained burst of 500 steps under Gaussian noise forces
    the agent to maintain stable Q-estimates across an extended perturbation window.
    The cool-down period (20000 steps) ensures bursts are rare but impactful.

    Default paper parameters (Table II):
        noise_rate=0.01, scale=1.0, burst_length=500, burst_cooldown=20000

    Parameters
    ----------
    env : gym.Env
    noise_rate : float, optional
        Per-step probability of triggering a burst. Default 0.05.
    loc : float, optional
        Mean of the noise distribution. Default 0.0.
    scale : float, optional
        Standard deviation of the noise. Default 0.8.
    start_episode : int, optional
        Warm-up duration in episodes. Default 100.
    burst_length : int, optional
        Duration of each noise burst in environment steps. Default 250.
    burst_cooldown : int, optional
        Minimum steps between two bursts. Default 7000.
    """

    def __init__(
        self,
        env,
        noise_rate=0.05,
        loc=0.0,
        scale=0.8,
        start_episode=100,
        burst_length=250,
        burst_cooldown=7000,
    ):
        super().__init__(
            env, noise_rate=noise_rate, loc=loc, scale=scale,
            start_episode=start_episode
        )
        self.burst_length = burst_length
        self.burst_cooldown = burst_cooldown
        self._step_counter = 0
        self._in_burst = False
        self._burst_steps_remaining = 0
        self._steps_since_last_burst = 0
        self._last_clean_obs = None

    def get_clean_observation(self):
        """Return the last observation before noise was applied."""
        return self._last_clean_obs

    def _observation(self, observation):
        self._last_clean_obs = observation.copy()
        self._step_counter += 1

        if (
            not self._in_burst
            and self._current_episode >= self.start_episode
            and np.random.rand() <= self.noise_rate
            and self._steps_since_last_burst >= self.burst_cooldown
        ):
            self._in_burst = True
            self._burst_steps_remaining = self.burst_length
            self._steps_since_last_burst = 0
            logger.debug(
                "BurstNoise: burst started at step %d for %d steps",
                self._step_counter,
                self.burst_length,
            )

        if self._in_burst:
            logger.debug(
                "BurstNoise: burst active, scale=%.3f, steps_remaining=%d",
                self.scale,
                self._burst_steps_remaining,
            )
            observation = observation + np.random.normal(
                loc=self.loc, scale=self.scale, size=observation.shape
            )
            self._burst_steps_remaining -= 1
            if self._burst_steps_remaining <= 0:
                self._in_burst = False
                logger.debug(
                    "BurstNoise: burst ended at step %d", self._step_counter
                )
        else:
            self._steps_since_last_burst += 1

        return observation
