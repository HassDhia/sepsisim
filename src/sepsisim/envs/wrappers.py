"""Environment wrappers for SB3 compatibility and observation normalization."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [-1, 1] range using observation space bounds."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.obs_range = self.obs_high - self.obs_low
        self.obs_range[self.obs_range == 0] = 1.0

        self.observation_space = spaces.Box(
            low=-np.ones_like(self.obs_low),
            high=np.ones_like(self.obs_high),
            dtype=np.float32,
        )

    def observation(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        normalized = 2.0 * (obs - self.obs_low) / self.obs_range - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


class ClipAction(gym.ActionWrapper):
    """Clip actions to the environment's action space bounds."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
