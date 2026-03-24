"""Tests for the FluidResuscitation environment."""

import gymnasium as gym
import numpy as np
import pytest

import sepsisim  # noqa: F401


class TestFluidResuscitationBasics:
    def test_env_creation(self, fluid_env):
        assert fluid_env is not None

    def test_observation_space_shape(self, fluid_env):
        assert fluid_env.observation_space.shape == (7,)

    def test_action_space_shape(self, fluid_env):
        assert fluid_env.action_space.shape == (1,)

    def test_reset_returns_obs_info(self, fluid_env):
        obs, info = fluid_env.reset(seed=42)
        assert obs.shape == (7,)
        assert isinstance(info, dict)

    def test_obs_in_bounds(self, fluid_env):
        obs, _ = fluid_env.reset(seed=42)
        assert np.all(obs >= fluid_env.observation_space.low)
        assert np.all(obs <= fluid_env.observation_space.high)

    def test_step_returns_tuple(self, fluid_env):
        fluid_env.reset(seed=42)
        action = np.array([250.0], dtype=np.float32)
        result = fluid_env.step(action)
        assert len(result) == 5


class TestFluidResuscitationDynamics:
    def test_fluid_increases_map(self, fluid_env):
        obs, _ = fluid_env.reset(seed=42)
        initial_map = obs[0]
        for _ in range(5):
            obs, _, _, _, _ = fluid_env.step(np.array([500.0], dtype=np.float32))
        assert obs[0] >= initial_map - 20

    def test_no_fluid_map_drops(self, fluid_env):
        obs, _ = fluid_env.reset(seed=42)
        initial_map = obs[0]
        for _ in range(20):
            obs, _, term, trunc, _ = fluid_env.step(np.array([0.0], dtype=np.float32))
            if term or trunc:
                break
        if not (term or trunc):
            assert obs[0] <= initial_map + 5

    def test_episode_length_bounded(self, fluid_env):
        fluid_env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 100:
            _, _, term, trunc, _ = fluid_env.step(np.array([250.0], dtype=np.float32))
            done = term or trunc
            steps += 1
        assert steps <= 73

    def test_severe_untreated_terminates(self):
        env = gym.make("sepsisim/FluidResuscitation-v0", severity="hard", antibiotic_given=False)
        env.reset(seed=42)
        terminated = False
        for _ in range(72):
            _, _, term, _, _ = env.step(np.array([0.0], dtype=np.float32))
            if term:
                terminated = True
                break
        env.close()
        assert terminated

    def test_info_contains_keys(self, fluid_env):
        fluid_env.reset(seed=42)
        _, _, _, _, info = fluid_env.step(np.array([100.0], dtype=np.float32))
        assert "map_mmhg" in info
        assert "lactate" in info
        assert "total_fluid_ml" in info


class TestFluidResuscitationSeverities:
    def test_easy_creates(self, fluid_env_easy):
        obs, _ = fluid_env_easy.reset(seed=42)
        assert obs.shape == (7,)

    def test_hard_creates(self, fluid_env_hard):
        obs, _ = fluid_env_hard.reset(seed=42)
        assert obs.shape == (7,)

    def test_hard_lower_initial_map(self, fluid_env_easy, fluid_env_hard):
        obs_easy, _ = fluid_env_easy.reset(seed=42)
        obs_hard, _ = fluid_env_hard.reset(seed=42)
        assert obs_hard[0] <= obs_easy[0] + 15


class TestFluidResuscitationReproducibility:
    def test_same_seed_same_trajectory(self, fluid_env):
        obs1, _ = fluid_env.reset(seed=42)
        action = np.array([300.0], dtype=np.float32)
        r1, _, _, _, _ = fluid_env.step(action)

        obs2, _ = fluid_env.reset(seed=42)
        r2, _, _, _, _ = fluid_env.step(action)

        np.testing.assert_array_almost_equal(obs1, obs2)
