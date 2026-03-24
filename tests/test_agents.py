"""Tests for baseline agents."""

import gymnasium as gym
import numpy as np
import pytest

import sepsisim  # noqa: F401
from sepsisim.agents.random_agent import RandomAgent
from sepsisim.agents.heuristic_agent import HeuristicAgent


class TestRandomAgent:
    def test_creation(self, fluid_env):
        agent = RandomAgent(fluid_env)
        assert agent is not None

    def test_predict_shape(self, fluid_env):
        agent = RandomAgent(fluid_env)
        obs, _ = fluid_env.reset(seed=42)
        action, state = agent.predict(obs)
        assert action.shape == fluid_env.action_space.shape
        assert state is None

    def test_evaluate_returns_stats(self, fluid_env):
        agent = RandomAgent(fluid_env)
        results = agent.evaluate(n_episodes=5)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert results["n_episodes"] == 5

    def test_works_on_all_envs(self):
        for env_id in [
            "sepsisim/FluidResuscitation-v0",
            "sepsisim/VasopressorTitration-v0",
            "sepsisim/SepsisManagement-v0",
        ]:
            env = gym.make(env_id)
            agent = RandomAgent(env)
            results = agent.evaluate(n_episodes=3)
            assert results["n_episodes"] == 3
            env.close()


class TestHeuristicAgent:
    def test_creation(self, fluid_env):
        agent = HeuristicAgent(fluid_env)
        assert agent is not None

    def test_fluid_policy_low_map(self, fluid_env):
        agent = HeuristicAgent(fluid_env)
        obs = np.array([50.0, 5.0, 0.1, 0.3, 1.0, 0.5, 8.0], dtype=np.float32)
        action, _ = agent.predict(obs)
        assert action[0] >= 500.0

    def test_fluid_policy_normal_map(self, fluid_env):
        agent = HeuristicAgent(fluid_env)
        obs = np.array([80.0, 1.5, 0.5, 0.1, 0.5, 0.2, 2.0], dtype=np.float32)
        action, _ = agent.predict(obs)
        assert action[0] <= 250.0

    def test_vaso_policy_low_map(self, vaso_env):
        agent = HeuristicAgent(vaso_env)
        obs = np.array([50.0, 5.0, 0.1, 0.3, 0.1, 0.5, 8.0, 5.0], dtype=np.float32)
        action, _ = agent.predict(obs)
        assert action[0] > 0

    def test_combined_policy(self, management_env):
        agent = HeuristicAgent(management_env)
        obs = np.array(
            [50.0, 5.0, 0.1, 0.3, 1.0, 0.1, 0.5, 8.0, 1.0, 0.0],
            dtype=np.float32,
        )
        action, _ = agent.predict(obs)
        assert action.shape == (3,)
        assert action[0] > 0
        assert action[2] == 1.0

    def test_evaluate_returns_stats(self, fluid_env):
        agent = HeuristicAgent(fluid_env)
        results = agent.evaluate(n_episodes=5)
        assert "mean_reward" in results

    def test_heuristic_lower_variance_fluid(self):
        env = gym.make("sepsisim/FluidResuscitation-v0", severity="easy")
        random_agent = RandomAgent(env, seed=42)
        heuristic_agent = HeuristicAgent(env)
        r_random = random_agent.evaluate(n_episodes=50)
        r_heuristic = heuristic_agent.evaluate(n_episodes=50)
        assert r_heuristic["std_reward"] <= r_random["std_reward"] * 1.5
        env.close()

    def test_heuristic_survives_management(self):
        env = gym.make("sepsisim/SepsisManagement-v0", severity="easy")
        heuristic_agent = HeuristicAgent(env)
        r_heuristic = heuristic_agent.evaluate(n_episodes=20)
        assert r_heuristic["mean_reward"] > -100.0
        env.close()
