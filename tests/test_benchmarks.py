"""Tests for the benchmark suite."""

import gymnasium as gym

import sepsisim  # noqa: F401
from sepsisim.benchmarks.environments import BENCHMARK_ENVS


class TestBenchmarkEnvironments:
    def test_all_envs_defined(self):
        assert len(BENCHMARK_ENVS) == 9

    def test_tiers_present(self):
        tiers = {v["tier"] for v in BENCHMARK_ENVS.values()}
        assert "easy" in tiers
        assert "medium" in tiers
        assert "hard" in tiers

    def test_all_envs_creatable(self):
        for name, cfg in BENCHMARK_ENVS.items():
            env = gym.make(cfg["env_id"], **cfg.get("kwargs", {}))
            obs, _ = env.reset(seed=42)
            assert obs is not None, f"Failed to create {name}"
            env.close()

    def test_all_envs_steppable(self):
        for name, cfg in BENCHMARK_ENVS.items():
            env = gym.make(cfg["env_id"], **cfg.get("kwargs", {}))
            env.reset(seed=42)
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert obs is not None, f"Failed to step {name}"
            env.close()

    def test_descriptions_non_empty(self):
        for name, cfg in BENCHMARK_ENVS.items():
            assert len(cfg["description"]) > 10, f"Missing description for {name}"


class TestBenchmarkConsistency:
    def test_fluid_envs_share_obs_shape(self):
        shapes = set()
        for name, cfg in BENCHMARK_ENVS.items():
            if "FluidResuscitation" in name:
                env = gym.make(cfg["env_id"], **cfg.get("kwargs", {}))
                shapes.add(env.observation_space.shape)
                env.close()
        assert len(shapes) == 1

    def test_vaso_envs_share_obs_shape(self):
        shapes = set()
        for name, cfg in BENCHMARK_ENVS.items():
            if "VasopressorTitration" in name:
                env = gym.make(cfg["env_id"], **cfg.get("kwargs", {}))
                shapes.add(env.observation_space.shape)
                env.close()
        assert len(shapes) == 1

    def test_management_envs_share_obs_shape(self):
        shapes = set()
        for name, cfg in BENCHMARK_ENVS.items():
            if "SepsisManagement" in name:
                env = gym.make(cfg["env_id"], **cfg.get("kwargs", {}))
                shapes.add(env.observation_space.shape)
                env.close()
        assert len(shapes) == 1
