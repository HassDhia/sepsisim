"""Simplified cardiovascular model for sepsis hemodynamics.

Models the relationship between fluid administration, vasopressor dosing,
and mean arterial pressure (MAP) in the context of septic shock.

SIMPLIFICATION: Using a lumped-parameter model rather than a full
Frank-Starling + Windkessel circuit. Clinical models include chamber-specific
compliance, valve dynamics, and regional vascular beds. Acceptable for
benchmark simulation of gross hemodynamic trends.

References:
    - Surviving Sepsis Campaign: Rhodes et al., Intensive Care Medicine, 2017.
    - BioGears sepsis model: McDaniel et al., Frontiers in Physiology, 2019.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PARAMETER_RANGES = {
    "map_baseline": {"min": 65.0, "max": 100.0, "unit": "mmHg", "default": 75.0,
                     "source": "Rhodes et al., 2017"},
    "svr_baseline": {"min": 800.0, "max": 1200.0, "unit": "dyn.s/cm5", "default": 1000.0,
                     "source": "Standard physiology"},
    "co_baseline": {"min": 4.0, "max": 8.0, "unit": "L/min", "default": 5.0,
                    "source": "Standard physiology"},
    "fluid_map_gain": {"min": 2.0, "max": 15.0, "unit": "mmHg/L", "default": 8.0,
                       "source": "CLOVERS trial, NEJM 2023"},
    "vaso_map_gain": {"min": 5.0, "max": 30.0, "unit": "mmHg/(mcg/kg/min)", "default": 15.0,
                      "source": "Rhodes et al., 2017"},
    "fluid_redistribution_rate": {"min": 0.05, "max": 0.3, "unit": "1/h", "default": 0.15,
                                  "source": "Estimated from clinical data"},
    "map_target": {"min": 60.0, "max": 75.0, "unit": "mmHg", "default": 65.0,
                   "source": "Rhodes et al., 2017"},
}


class CardiovascularModel:
    """Lumped cardiovascular model for sepsis hemodynamics.

    State variables:
        map_mmhg: mean arterial pressure (mmHg)
        intravascular_volume: effective circulating volume (L, relative to baseline)
        vasopressor_effect: current vasopressor effect level (0-1)

    The model captures:
        - MAP depression from sepsis-induced vasodilation (via tissue damage)
        - MAP increase from IV fluid boluses (transient, redistributes over time)
        - MAP increase from vasopressor infusion (norepinephrine-like)
        - Diminishing returns on fluid responsiveness (Starling curve plateau)
    """

    def __init__(
        self,
        map_baseline: float = 75.0,
        svr_baseline: float = 1000.0,
        co_baseline: float = 5.0,
        fluid_map_gain: float = 8.0,
        vaso_map_gain: float = 15.0,
        fluid_redistribution_rate: float = 0.15,
        map_target: float = 65.0,
    ) -> None:
        self.map_baseline = map_baseline
        self.svr_baseline = svr_baseline
        self.co_baseline = co_baseline
        self.fluid_map_gain = fluid_map_gain
        self.vaso_map_gain = vaso_map_gain
        self.fluid_redistribution_rate = fluid_redistribution_rate
        self.map_target = map_target

    def compute_map(
        self,
        tissue_damage: float,
        intravascular_volume: float,
        vasopressor_dose: float,
        noise_rng: np.random.Generator | None = None,
    ) -> float:
        """Compute MAP given current physiological state.

        Args:
            tissue_damage: tissue damage level from inflammation model (0-1).
            intravascular_volume: net fluid volume above baseline (L).
            vasopressor_dose: norepinephrine-equivalent dose (mcg/kg/min).
            noise_rng: optional RNG for physiological noise.

        Returns:
            Current MAP in mmHg.
        """
        sepsis_depression = self.map_baseline * 0.3 * tissue_damage

        # SIMPLIFICATION: Linear fluid response with saturation.
        # Clinical reality follows Frank-Starling curve with complex preload dependence.
        fluid_effect = self.fluid_map_gain * intravascular_volume
        fluid_effect *= 1.0 / (1.0 + 0.3 * max(0, intravascular_volume))

        vaso_effect = self.vaso_map_gain * vasopressor_dose
        vaso_effect *= 1.0 / (1.0 + 0.05 * vasopressor_dose)

        map_val = self.map_baseline - sepsis_depression + fluid_effect + vaso_effect

        if noise_rng is not None:
            map_val += noise_rng.normal(0, 2.0)

        return float(np.clip(map_val, 20.0, 150.0))

    def update_volume(
        self,
        intravascular_volume: float,
        fluid_bolus_ml: float,
        dt: float = 1.0,
    ) -> float:
        """Update intravascular volume after fluid administration and redistribution.

        Args:
            intravascular_volume: current excess volume (L above baseline).
            fluid_bolus_ml: fluid administered this step (mL).
            dt: time step in hours.

        Returns:
            Updated intravascular volume.
        """
        volume = intravascular_volume + fluid_bolus_ml / 1000.0

        volume -= self.fluid_redistribution_rate * volume * dt

        return float(np.clip(volume, -1.0, 10.0))

    def compute_urine_output(
        self, map_mmhg: float, noise_rng: np.random.Generator | None = None
    ) -> float:
        """Estimate urine output based on MAP (simplified renal autoregulation).

        Args:
            map_mmhg: current mean arterial pressure.
            noise_rng: optional RNG for noise.

        Returns:
            Urine output in mL/kg/h.
        """
        # SIMPLIFICATION: Linear relationship between MAP and UO.
        # Clinical renal autoregulation maintains GFR across MAP 60-160 mmHg.
        if map_mmhg < 50:
            uo = 0.0
        elif map_mmhg < 65:
            uo = 0.3 * (map_mmhg - 50) / 15.0
        else:
            uo = 0.3 + 0.2 * min(1.0, (map_mmhg - 65) / 30.0)

        if noise_rng is not None:
            uo += noise_rng.normal(0, 0.05)

        return float(np.clip(uo, 0.0, 2.0))

    def reset(self) -> dict[str, float]:
        """Return initial cardiovascular state."""
        return {
            "intravascular_volume": 0.0,
            "vasopressor_dose": 0.0,
            "map_mmhg": self.map_baseline,
        }
