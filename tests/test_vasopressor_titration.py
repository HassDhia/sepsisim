"""Tests for the VasopressorTitration environment."""

import gymnasium as gym
import numpy as np
import pytest

import sepsisim  # noqa: F401


class TestVasopressorBasics:
    def test_env_creation(self, vaso_env):
        assert vaso_env is not None

    def test_observation_space(self, vaso_env):
        assert vaso_env.observation_space.shape == (8,)

    def test_action_space(self, vaso_env):
        assert vaso_env.action_space.shape == (1,)
        assert vaso_env.action_space.low[0] == pytest.approx(-0.1)
        assert vaso_env.action_space.high[0] == pytest.approx(0.1)

    def test_reset(self, vaso_env):
        obs, info = vaso_env.reset(seed=42)
        assert obs.shape == (8,)
        assert "vasopressor_dose" in info

    def test_step(self, vaso_env):
        vaso_env.reset(seed=42)
        action = np.array([0.05], dtype=np.float32)
        obs, reward, term, trunc, info = vaso_env.step(action)
        assert obs.shape == (8,)
        assert isinstance(reward, float)


class TestVasopressorDynamics:
    def test_increase_dose_raises_map(self, vaso_env):
        obs, _ = vaso_env.reset(seed=42)
        initial_map = obs[0]
        for _ in range(5):
            obs, _, _, _, _ = vaso_env.step(np.array([0.05], dtype=np.float32))
        assert obs[4] > 0.05

    def test_dose_bounded(self, vaso_env):
        vaso_env.reset(seed=42)
        for _ in range(50):
            obs, _, _, _, _ = vaso_env.step(np.array([0.1], dtype=np.float32))
        assert obs[4] <= 2.0

    def test_dose_non_negative(self, vaso_env):
        vaso_env.reset(seed=42)
        for _ in range(50):
            obs, _, _, _, _ = vaso_env.step(np.array([-0.1], dtype=np.float32))
        assert obs[4] >= 0.0

    def test_hours_on_vaso_tracks(self, vaso_env):
        vaso_env.reset(seed=42)
        for _ in range(10):
            obs, _, _, _, _ = vaso_env.step(np.array([0.01], dtype=np.float32))
        assert obs[7] >= 9.0

    def test_info_keys(self, vaso_env):
        vaso_env.reset(seed=42)
        _, _, _, _, info = vaso_env.step(np.array([0.02], dtype=np.float32))
        assert "hours_on_vasopressors" in info
        assert "bacteria" in info


class TestVasopressorReproducibility:
    def test_deterministic(self, vaso_env):
        obs1, _ = vaso_env.reset(seed=42)
        r1 = []
        for _ in range(5):
            _, rew, _, _, _ = vaso_env.step(np.array([0.03], dtype=np.float32))
            r1.append(rew)

        obs2, _ = vaso_env.reset(seed=42)
        r2 = []
        for _ in range(5):
            _, rew, _, _, _ = vaso_env.step(np.array([0.03], dtype=np.float32))
            r2.append(rew)

        np.testing.assert_array_almost_equal(r1, r2)
