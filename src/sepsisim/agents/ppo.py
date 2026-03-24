"""PPO agent wrapper and CLI entrypoint for SepsiSim training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

import sepsisim  # noqa: F401 (registers environments)
from sepsisim.training.configs import ENV_CONFIGS, SEED


def train_ppo(env_id: str, config: dict | None = None) -> dict:
    """Train a PPO agent on the given environment.

    Args:
        env_id: Gymnasium environment ID.
        config: Training hyperparameters (defaults from configs.py).

    Returns:
        Training results dictionary.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 required: pip install sepsisim[train]")
        sys.exit(1)

    cfg = ENV_CONFIGS.get(env_id, ENV_CONFIGS["sepsisim/FluidResuscitation-v0"])
    if config:
        cfg = {**cfg, **config}

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Training with seed={SEED}")

    def make_env():
        return gym.make(env_id)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        verbose=1,
        seed=SEED,
    )

    model.learn(total_timesteps=cfg["total_timesteps"])

    eval_env = gym.make(env_id)
    rewards = []
    lengths = []
    for ep in range(50):
        obs, _ = eval_env.reset(seed=ep)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)

    results = {
        "env_id": env_id,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "total_timesteps": cfg["total_timesteps"],
        "seed": SEED,
    }

    return results, model


def main():
    parser = argparse.ArgumentParser(description="Train PPO on SepsiSim")
    parser.add_argument(
        "--env",
        default="sepsisim/FluidResuscitation-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    args = parser.parse_args()

    config = {}
    if args.timesteps:
        config["total_timesteps"] = args.timesteps

    results, model = train_ppo(args.env, config)
    print(json.dumps(results, indent=2))

    save_path = Path("results/models") / f"{args.env.split('/')[-1]}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
