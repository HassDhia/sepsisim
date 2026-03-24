"""Tests for environment wrappers."""

import gymnasium as gym
import numpy as np

import sepsisim  # noqa: F401
from sepsisim.envs.wrappers import NormalizeObservation, ClipAction


class TestNormalizeObservation:
    def test_wrapping(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        wrapped = NormalizeObservation(env)
        obs, _ = wrapped.reset(seed=42)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)
        wrapped.close()

    def test_step_normalized(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        wrapped = NormalizeObservation(env)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step(np.array([250.0], dtype=np.float32))
        assert np.all(obs >= -1.1)
        assert np.all(obs <= 1.1)
        wrapped.close()

    def test_space_shape_preserved(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        wrapped = NormalizeObservation(env)
        assert wrapped.observation_space.shape == env.observation_space.shape
        wrapped.close()


class TestClipAction:
    def test_clipping(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        wrapped = ClipAction(env)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step(np.array([5000.0], dtype=np.float32))
        assert obs.shape == (7,)
        wrapped.close()

    def test_negative_action_clipped(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        wrapped = ClipAction(env)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step(np.array([-100.0], dtype=np.float32))
        assert obs.shape == (7,)
        wrapped.close()
