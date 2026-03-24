"""Shared training hyperparameter configurations.

This module is the SINGLE SOURCE OF TRUTH for training hyperparameters.
Both the CLI entrypoint (agents/ppo.py) and train_all.py import from here.
"""

ENV_CONFIGS = {
    "sepsisim/FluidResuscitation-v0": {
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "normalize_obs": True,
        "difficulty": "medium",
    },
    "sepsisim/VasopressorTitration-v0": {
        "total_timesteps": 300_000,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "normalize_obs": True,
        "difficulty": "medium",
    },
    "sepsisim/SepsisManagement-v0": {
        "total_timesteps": 500_000,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "normalize_obs": True,
        "difficulty": "medium",
    },
}

SEED = 42
EVAL_EPISODES = 50
