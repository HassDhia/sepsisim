"""Benchmark environment configurations with difficulty tiers."""

BENCHMARK_ENVS = {
    "FluidResuscitation-easy": {
        "env_id": "sepsisim/FluidResuscitation-v0",
        "kwargs": {"severity": "easy"},
        "tier": "easy",
        "description": "Mild sepsis, low bacteria load, fluid-responsive",
    },
    "FluidResuscitation-medium": {
        "env_id": "sepsisim/FluidResuscitation-v0",
        "kwargs": {"severity": "medium"},
        "tier": "medium",
        "description": "Moderate sepsis, standard fluid resuscitation challenge",
    },
    "FluidResuscitation-hard": {
        "env_id": "sepsisim/FluidResuscitation-v0",
        "kwargs": {"severity": "hard"},
        "tier": "hard",
        "description": "Severe sepsis, high bacteria load, limited fluid responsiveness",
    },
    "VasopressorTitration-easy": {
        "env_id": "sepsisim/VasopressorTitration-v0",
        "kwargs": {"severity": "easy"},
        "tier": "easy",
        "description": "Mild septic shock, low vasopressor requirements",
    },
    "VasopressorTitration-medium": {
        "env_id": "sepsisim/VasopressorTitration-v0",
        "kwargs": {"severity": "medium"},
        "tier": "medium",
        "description": "Moderate shock, requires careful vasopressor titration",
    },
    "VasopressorTitration-hard": {
        "env_id": "sepsisim/VasopressorTitration-v0",
        "kwargs": {"severity": "hard"},
        "tier": "hard",
        "description": "Severe refractory shock, high vasopressor demand",
    },
    "SepsisManagement-easy": {
        "env_id": "sepsisim/SepsisManagement-v0",
        "kwargs": {"severity": "easy"},
        "tier": "easy",
        "description": "Mild sepsis, combined management with clear clinical trajectory",
    },
    "SepsisManagement-medium": {
        "env_id": "sepsisim/SepsisManagement-v0",
        "kwargs": {"severity": "medium"},
        "tier": "medium",
        "description": "Moderate sepsis, complex multi-intervention optimization",
    },
    "SepsisManagement-hard": {
        "env_id": "sepsisim/SepsisManagement-v0",
        "kwargs": {"severity": "hard"},
        "tier": "hard",
        "description": "Severe sepsis, all interventions critical, narrow therapeutic window",
    },
}
