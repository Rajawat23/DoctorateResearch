"""
noisyenv
--------
Gym observation wrappers for the MASURE noise evaluation protocol.

Public API (wrappers used in the paper):
    StepBurstNoiseObservation    -- Rare burst noise schedule (MASURE Sec. III-D).
    RandomNormalNoisyObservation -- Base Gaussian noise wrapper.
    EpisodeAwareEnv              -- Episode-count propagation wrapper.

Additional experimental wrappers (not part of published MASURE evaluation):
    See noisyenv/extras.py.
"""

from .wrappers import (
    StepBurstNoiseObservation,
    RandomNormalNoisyObservation,
    EpisodeAwareEnv,
)

__all__ = [
    "StepBurstNoiseObservation",
    "RandomNormalNoisyObservation",
    "EpisodeAwareEnv",
]
