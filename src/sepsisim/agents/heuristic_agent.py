"""Heuristic baseline agent implementing clinical sepsis guidelines.

Implements a simplified version of the Surviving Sepsis Campaign
hour-1 bundle: early fluids, vasopressors for refractory hypotension,
and antibiotic administration.

Reference:
    Rhodes, A., Evans, L.E., Alhazzani, W., et al. (2017).
    Surviving Sepsis Campaign: International Guidelines.
    Intensive Care Medicine, 43(3), 304-377.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class HeuristicAgent:
    """Clinical guideline-based heuristic agent.

    For FluidResuscitation-v0 (1D action: fluid mL):
        - Give 500mL bolus if MAP < 65
        - Give 250mL if MAP 65-70
        - Give 0mL if MAP > 70

    For VasopressorTitration-v0 (1D action: dose change):
        - Increase by 0.05 if MAP < 65
        - Decrease by 0.02 if MAP > 85
        - Hold if MAP 65-85

    For SepsisManagement-v0 (3D action: fluid, vaso change, abx):
        - Fluid: 500mL if MAP < 65, 250mL if 65-70, 0 otherwise
        - Vaso: increase if MAP < 65 despite fluids, decrease if MAP > 85
        - Abx: give immediately on first step
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self._step_count = 0

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[NDArray[np.float32], None]:
        self._step_count += 1
        map_mmhg = float(obs[0])

        if self.action_dim == 1 and self.env.action_space.high[0] > 10:
            action = self._fluid_policy(obs)
        elif self.action_dim == 1:
            action = self._vaso_policy(map_mmhg)
        else:
            action = self._combined_policy(obs)

        return action, None

    def _fluid_policy(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        map_mmhg = float(obs[0])
        volume = float(obs[4])
        if volume > 3.5:
            return np.array([0.0], dtype=np.float32)
        if map_mmhg < 65:
            return np.array([750.0], dtype=np.float32)
        elif map_mmhg < 70:
            return np.array([500.0], dtype=np.float32)
        elif map_mmhg < 80:
            return np.array([250.0], dtype=np.float32)
        return np.array([125.0], dtype=np.float32)

    def _vaso_policy(self, map_mmhg: float) -> NDArray[np.float32]:
        if map_mmhg < 65:
            return np.array([0.05], dtype=np.float32)
        elif map_mmhg > 85:
            return np.array([-0.02], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def _combined_policy(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        map_mmhg = float(obs[0])
        volume = float(obs[4])
        vaso_dose = float(obs[5])
        abx_active = float(obs[9]) > 0.5

        if map_mmhg < 65 and volume < 4.0:
            fluid = 750.0
        elif map_mmhg < 70:
            fluid = 500.0
        elif map_mmhg < 80:
            fluid = 250.0
        else:
            fluid = 125.0

        if map_mmhg < 65 and volume >= 2.0:
            vaso_change = 0.05
        elif map_mmhg > 85 and vaso_dose > 0:
            vaso_change = -0.02
        else:
            vaso_change = 0.0

        give_abx = 1.0 if not abx_active else 0.0

        return np.array([fluid, vaso_change, give_abx], dtype=np.float32)

    def evaluate(self, n_episodes: int = 50) -> dict:
        rewards = []
        lengths = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)
            total_reward = 0.0
            steps = 0
            self._step_count = 0
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
