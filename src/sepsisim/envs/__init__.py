"""Gymnasium environment registration for SepsiSim."""

from gymnasium.envs.registration import register


def register_envs() -> None:
    """Register all SepsiSim environments with Gymnasium."""
    register(
        id="sepsisim/FluidResuscitation-v0",
        entry_point="sepsisim.envs.fluid_resuscitation:FluidResuscitationEnv",
        max_episode_steps=72,
    )
    register(
        id="sepsisim/VasopressorTitration-v0",
        entry_point="sepsisim.envs.vasopressor_titration:VasopressorTitrationEnv",
        max_episode_steps=72,
    )
    register(
        id="sepsisim/SepsisManagement-v0",
        entry_point="sepsisim.envs.sepsis_management:SepsisManagementEnv",
        max_episode_steps=72,
    )
