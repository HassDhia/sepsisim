"""Random baseline agent for SepsiSim environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class RandomAgent:
    """Agent that takes uniformly random actions within the action space."""

    def __init__(self, env: gym.Env, seed: int = 42) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = False
    ) -> tuple[NDArray[np.float32], None]:
        action = self.env.action_space.sample()
        return action, None

    def evaluate(self, n_episodes: int = 50) -> dict:
        """Run evaluation episodes and return statistics."""
        rewards = []
        lengths = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action, _ = self.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            rewards.append(total_reward)
            lengths.append(steps)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }
