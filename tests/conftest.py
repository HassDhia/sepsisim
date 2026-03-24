"""Shared test fixtures for SepsiSim."""

import gymnasium as gym
import numpy as np
import pytest

import sepsisim  # noqa: F401


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fluid_env():
    env = gym.make("sepsisim/FluidResuscitation-v0")
    yield env
    env.close()


@pytest.fixture
def vaso_env():
    env = gym.make("sepsisim/VasopressorTitration-v0")
    yield env
    env.close()


@pytest.fixture
def management_env():
    env = gym.make("sepsisim/SepsisManagement-v0")
    yield env
    env.close()


@pytest.fixture
def fluid_env_easy():
    env = gym.make("sepsisim/FluidResuscitation-v0", severity="easy")
    yield env
    env.close()


@pytest.fixture
def fluid_env_hard():
    env = gym.make("sepsisim/FluidResuscitation-v0", severity="hard")
    yield env
    env.close()
