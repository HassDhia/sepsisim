"""Benchmark runner for comparing agents across environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import sepsisim  # noqa: F401
from sepsisim.agents.random_agent import RandomAgent
from sepsisim.agents.heuristic_agent import HeuristicAgent
from sepsisim.benchmarks.environments import BENCHMARK_ENVS


def run_benchmarks(
    n_episodes: int = 50,
    output_path: str | None = None,
) -> dict:
    """Run all benchmark evaluations."""
    results = {}

    for bench_name, bench_cfg in BENCHMARK_ENVS.items():
        print(f"Benchmarking: {bench_name}")
        env = gym.make(bench_cfg["env_id"], **bench_cfg.get("kwargs", {}))

        random_agent = RandomAgent(env, seed=42)
        random_results = random_agent.evaluate(n_episodes)

        heuristic_agent = HeuristicAgent(env)
        heuristic_results = heuristic_agent.evaluate(n_episodes)

        results[bench_name] = {
            "tier": bench_cfg["tier"],
            "description": bench_cfg["description"],
            "random": random_results,
            "heuristic": heuristic_results,
        }

        env.close()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run SepsiSim benchmarks")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default="results/benchmark_results.json")
    args = parser.parse_args()

    results = run_benchmarks(args.episodes, args.output)

    print("\n" + "=" * 70)
    print(f"{'Environment':<35} {'Random':>10} {'Heuristic':>12}")
    print("=" * 70)
    for name, data in results.items():
        r_mean = data["random"]["mean_reward"]
        h_mean = data["heuristic"]["mean_reward"]
        print(f"{name:<35} {r_mean:>10.1f} {h_mean:>12.1f}")


if __name__ == "__main__":
    main()
