"""Tests for training configuration and evaluation utilities."""

import gymnasium as gym
import numpy as np

import sepsisim  # noqa: F401
from sepsisim.training.configs import ENV_CONFIGS, SEED, EVAL_EPISODES
from sepsisim.training.evaluate import evaluate_agent
from sepsisim.agents.random_agent import RandomAgent


class TestConfigs:
    def test_all_envs_have_configs(self):
        assert "sepsisim/FluidResuscitation-v0" in ENV_CONFIGS
        assert "sepsisim/VasopressorTitration-v0" in ENV_CONFIGS
        assert "sepsisim/SepsisManagement-v0" in ENV_CONFIGS

    def test_configs_have_required_keys(self):
        required = ["total_timesteps", "learning_rate", "n_steps", "batch_size"]
        for env_id, cfg in ENV_CONFIGS.items():
            for key in required:
                assert key in cfg, f"Missing {key} in {env_id}"

    def test_timesteps_minimum(self):
        for env_id, cfg in ENV_CONFIGS.items():
            assert cfg["total_timesteps"] >= 200_000, (
                f"{env_id} has {cfg['total_timesteps']} < 200K min"
            )

    def test_seed_defined(self):
        assert SEED == 42

    def test_eval_episodes_reasonable(self):
        assert EVAL_EPISODES >= 10


class TestEvaluate:
    def test_evaluate_random_agent(self):
        env = gym.make("sepsisim/FluidResuscitation-v0")
        agent = RandomAgent(env, seed=42)
        results = evaluate_agent(agent, env, n_episodes=5)
        assert "mean_reward" in results
        assert "survival_rate" in results
        assert "mean_final_map" in results
        assert "mean_final_lactate" in results
        env.close()

    def test_evaluate_returns_correct_count(self):
        env = gym.make("sepsisim/VasopressorTitration-v0")
        agent = RandomAgent(env, seed=42)
        results = evaluate_agent(agent, env, n_episodes=10)
        assert results["n_episodes"] == 10
        env.close()

    def test_evaluate_survival_bounded(self):
        env = gym.make("sepsisim/SepsisManagement-v0")
        agent = RandomAgent(env, seed=42)
        results = evaluate_agent(agent, env, n_episodes=10)
        assert 0.0 <= results["survival_rate"] <= 1.0
        env.close()
