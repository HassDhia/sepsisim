"""Training orchestration for all SepsiSim environments."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import sepsisim  # noqa: F401
from sepsisim.agents.random_agent import RandomAgent
from sepsisim.agents.heuristic_agent import HeuristicAgent
from sepsisim.training.configs import ENV_CONFIGS, SEED, EVAL_EPISODES


def train_all(output_dir: str = "results") -> dict:
    """Train and evaluate all agents on all environments.

    Returns:
        Complete training results for all environments and agents.
    """
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Training with seed={SEED}")

    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for env_id, cfg in ENV_CONFIGS.items():
        env_name = env_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Training: {env_name}")
        print(f"{'='*60}")

        env = gym.make(env_id)

        print(f"  Evaluating Random agent...")
        random_agent = RandomAgent(env, seed=SEED)
        random_results = random_agent.evaluate(n_episodes=EVAL_EPISODES)
        print(f"  Random: {random_results['mean_reward']:.2f} +/- {random_results['std_reward']:.2f}")

        print(f"  Evaluating Heuristic agent...")
        heuristic_agent = HeuristicAgent(env)
        heuristic_results = heuristic_agent.evaluate(n_episodes=EVAL_EPISODES)
        print(f"  Heuristic: {heuristic_results['mean_reward']:.2f} +/- {heuristic_results['std_reward']:.2f}")

        ppo_results = None
        try:
            from sepsisim.agents.ppo import train_ppo
            print(f"  Training PPO ({cfg['total_timesteps']} steps)...")
            ppo_results, model = train_ppo(env_id)
            print(f"  PPO: {ppo_results['mean_reward']:.2f} +/- {ppo_results['std_reward']:.2f}")

            model_path = output_path / "models" / f"{env_name}.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
        except ImportError:
            print("  PPO training skipped (install sepsisim[train])")
            ppo_results = {"mean_reward": 0.0, "std_reward": 0.0, "note": "skipped"}

        random_mean = random_results["mean_reward"]
        ppo_mean = ppo_results["mean_reward"] if ppo_results else 0.0
        if abs(random_mean) > 0.01:
            ratio = abs(ppo_mean - random_mean) / abs(random_mean)
        else:
            ratio = ppo_mean - random_mean
            ppo_results["ppo_vs_random_ratio_note"] = (
                "Random baseline near zero; using raw difference"
            )

        env_results = {
            "random": random_results,
            "heuristic": heuristic_results,
            "ppo": ppo_results,
            "ppo_vs_random_ratio": round(ratio, 3),
            "training_config": {
                "total_timesteps": cfg["total_timesteps"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "seed": SEED,
            },
        }
        results[env_name] = env_results
        env.close()

    results_path = output_path / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    train_all()
