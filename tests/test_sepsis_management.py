"""Tests for the SepsisManagement environment."""

import gymnasium as gym
import numpy as np
import pytest

import sepsisim  # noqa: F401


class TestManagementBasics:
    def test_env_creation(self, management_env):
        assert management_env is not None

    def test_observation_space(self, management_env):
        assert management_env.observation_space.shape == (10,)

    def test_action_space(self, management_env):
        assert management_env.action_space.shape == (3,)

    def test_reset(self, management_env):
        obs, info = management_env.reset(seed=42)
        assert obs.shape == (10,)

    def test_step(self, management_env):
        management_env.reset(seed=42)
        action = np.array([250.0, 0.02, 1.0], dtype=np.float32)
        obs, reward, term, trunc, info = management_env.step(action)
        assert obs.shape == (10,)


class TestManagementDynamics:
    def test_antibiotic_activates(self, management_env):
        management_env.reset(seed=42)
        action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        obs, _, _, _, info = management_env.step(action)
        assert info["antibiotic_active"] is True
        assert obs[9] == 1.0

    def test_antibiotic_irreversible(self, management_env):
        management_env.reset(seed=42)
        management_env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        _, _, _, _, info = management_env.step(
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        assert info["antibiotic_active"] is True

    def test_combined_fluid_and_vaso(self, management_env):
        management_env.reset(seed=42)
        action = np.array([500.0, 0.05, 1.0], dtype=np.float32)
        obs, _, _, _, info = management_env.step(action)
        assert info["total_fluid_ml"] == 500.0
        assert info["vasopressor_dose"] > 0

    def test_untreated_severe_terminates(self):
        env = gym.make("sepsisim/SepsisManagement-v0", severity="hard")
        env.reset(seed=42)
        terminated = False
        for _ in range(72):
            _, _, term, _, _ = env.step(
                np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            if term:
                terminated = True
                break
        env.close()
        assert terminated

    def test_stabilization_bonus(self, management_env):
        management_env.reset(seed=42)
        total_reward = 0.0
        for i in range(20):
            action = np.array([500.0, 0.05, 1.0 if i == 0 else 0.0], dtype=np.float32)
            _, reward, term, trunc, _ = management_env.step(action)
            total_reward += reward
            if term or trunc:
                break
        assert total_reward != 0.0


class TestManagementSeverities:
    def test_easy(self):
        env = gym.make("sepsisim/SepsisManagement-v0", severity="easy")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (10,)
        env.close()

    def test_hard(self):
        env = gym.make("sepsisim/SepsisManagement-v0", severity="hard")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (10,)
        env.close()


class TestManagementReproducibility:
    def test_deterministic_reset(self, management_env):
        obs1, _ = management_env.reset(seed=42)
        obs2, _ = management_env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
