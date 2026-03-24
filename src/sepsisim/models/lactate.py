"""Lactate kinetics model for sepsis.

Models lactate production from tissue hypoperfusion and clearance
dependent on hepatic function and tissue perfusion.

SIMPLIFICATION: Using a single-compartment model rather than multi-compartment
(blood, muscle, liver). Clinical lactate kinetics involve organ-specific
production rates and hepatic extraction ratios. Acceptable for benchmark
simulation of gross lactate trends.

References:
    - Hernandez, G. et al. (2019). Lactate in sepsis. Intensive Care Medicine.
    - Bakker, J. et al. (2022). Lactate: where are we now? Critical Care.
"""

from __future__ import annotations

import numpy as np

PARAMETER_RANGES = {
    "baseline_lactate": {"min": 0.5, "max": 1.5, "unit": "mmol/L", "default": 1.0,
                         "source": "Normal physiology"},
    "max_production_rate": {"min": 2.0, "max": 8.0, "unit": "mmol/L/h", "default": 4.0,
                            "source": "Hernandez et al., 2019"},
    "clearance_rate": {"min": 0.1, "max": 0.5, "unit": "1/h", "default": 0.3,
                       "source": "Bakker et al., 2022"},
    "map_threshold": {"min": 55.0, "max": 70.0, "unit": "mmHg", "default": 65.0,
                      "source": "Rhodes et al., 2017"},
    "damage_production_gain": {"min": 0.5, "max": 5.0, "unit": "mmol/L/h", "default": 1.5,
                               "source": "Estimated from clinical data"},
}


class LactateModel:
    """Single-compartment lactate kinetics.

    Lactate rises when:
        - MAP drops below threshold (tissue hypoperfusion)
        - Tissue damage increases (direct cellular injury)

    Lactate clears when:
        - MAP is adequate (restored perfusion)
        - Tissue damage is low (intact hepatic function)
    """

    def __init__(
        self,
        baseline_lactate: float = 1.0,
        max_production_rate: float = 4.0,
        clearance_rate: float = 0.3,
        map_threshold: float = 65.0,
        damage_production_gain: float = 1.5,
    ) -> None:
        self.baseline_lactate = baseline_lactate
        self.max_production_rate = max_production_rate
        self.clearance_rate = clearance_rate
        self.map_threshold = map_threshold
        self.damage_production_gain = damage_production_gain

    def step(
        self,
        lactate: float,
        map_mmhg: float,
        tissue_damage: float,
        dt: float = 1.0,
    ) -> float:
        """Update lactate level.

        Args:
            lactate: current blood lactate (mmol/L).
            map_mmhg: current mean arterial pressure (mmHg).
            tissue_damage: tissue damage from inflammation model (0-1).
            dt: time step in hours.

        Returns:
            Updated lactate level (mmol/L).
        """
        hypoperfusion = max(0.0, 1.0 - map_mmhg / self.map_threshold)
        production = (self.max_production_rate * hypoperfusion
                      + self.damage_production_gain * tissue_damage)

        perfusion_factor = min(1.0, map_mmhg / self.map_threshold)
        liver_factor = max(0.3, 1.0 - 0.5 * tissue_damage)
        clearance = self.clearance_rate * perfusion_factor * liver_factor

        d_lactate = production - clearance * (lactate - self.baseline_lactate)
        new_lactate = lactate + d_lactate * dt

        return float(np.clip(new_lactate, 0.1, 30.0))

    def reset(
        self,
        severity: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Generate initial lactate level.

        Args:
            severity: initial sepsis severity (0-1).
            rng: optional RNG for stochastic variation.

        Returns:
            Initial lactate in mmol/L.
        """
        base = self.baseline_lactate + 3.0 * severity
        if rng is not None:
            base *= rng.uniform(0.8, 1.2)
        return float(np.clip(base, 0.5, 15.0))
