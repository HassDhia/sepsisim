"""Evaluation utilities for trained agents."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def evaluate_agent(
    agent,
    env: gym.Env,
    n_episodes: int = 50,
    seed_offset: int = 1000,
) -> dict:
    """Evaluate an agent on an environment.

    Args:
        agent: Agent with predict(obs, deterministic) method.
        env: Gymnasium environment.
        n_episodes: Number of evaluation episodes.
        seed_offset: Seed offset for reproducibility.

    Returns:
        Evaluation results dictionary.
    """
    rewards = []
    lengths = []
    final_maps = []
    final_lactates = []
    survivals = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + seed_offset)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)
        final_maps.append(info.get("map_mmhg", 0.0))
        final_lactates.append(info.get("lactate", 0.0))
        survivals.append(0.0 if terminated else 1.0)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_final_map": float(np.mean(final_maps)),
        "mean_final_lactate": float(np.mean(final_lactates)),
        "survival_rate": float(np.mean(survivals)),
        "n_episodes": n_episodes,
    }
