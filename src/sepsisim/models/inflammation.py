"""Acute inflammation ODE model based on Reynolds et al. (2006).

Implements a 4-variable system tracking bacteria, pro-inflammatory response,
anti-inflammatory response, and tissue damage. This is a simplified benchmark
model suitable for RL environment dynamics.

Reference:
    Reynolds, A., Rubin, J., Clermont, G., Day, J., Vodovotz, Y., & Bhatt, G.B.
    (2006). A reduced mathematical model of the acute inflammatory response:
    I. Derivation of model and analysis of anti-inflammation.
    Journal of Theoretical Biology, 242(1), 220-236.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# SIMPLIFICATION: Using lumped pro-/anti-inflammatory variables rather than
# individual cytokine species (TNF-alpha, IL-6, IL-10, etc.).
# Clinical models track 10+ cytokines. Acceptable for benchmark simulation.

PARAMETER_RANGES = {
    "k_growth": {"min": 0.1, "max": 1.0, "unit": "1/h", "default": 0.6,
                 "source": "Reynolds et al., 2006"},
    "k_kill": {"min": 0.1, "max": 1.0, "unit": "1/h", "default": 0.5,
               "source": "Reynolds et al., 2006"},
    "b_max": {"min": 0.5, "max": 2.0, "unit": "normalized", "default": 1.0,
              "source": "Reynolds et al., 2006 (normalized carrying capacity)"},
    "s_m": {"min": 0.001, "max": 0.1, "unit": "1/h", "default": 0.01,
            "source": "Reynolds et al., 2006"},
    "k_mb": {"min": 0.5, "max": 5.0, "unit": "1/h", "default": 1.5,
             "source": "Reynolds et al., 2006"},
    "k_md": {"min": 0.1, "max": 2.0, "unit": "1/h", "default": 0.2,
             "source": "Reynolds et al., 2006 (calibrated for RL timescale)"},
    "mu_m": {"min": 0.01, "max": 0.5, "unit": "1/h", "default": 0.2,
             "source": "Reynolds et al., 2006 (calibrated for RL timescale)"},
    "k_ma": {"min": 0.1, "max": 1.0, "unit": "1/h", "default": 0.5,
             "source": "Reynolds et al., 2006 (calibrated for RL timescale)"},
    "s_a": {"min": 0.001, "max": 0.1, "unit": "1/h", "default": 0.01,
            "source": "Reynolds et al., 2006"},
    "k_am": {"min": 0.1, "max": 2.0, "unit": "1/h", "default": 0.4,
             "source": "Reynolds et al., 2006"},
    "k_ad": {"min": 0.01, "max": 0.5, "unit": "1/h", "default": 0.1,
             "source": "Reynolds et al., 2006"},
    "mu_a": {"min": 0.01, "max": 0.5, "unit": "1/h", "default": 0.05,
             "source": "Reynolds et al., 2006"},
    "k_dm": {"min": 0.1, "max": 2.0, "unit": "1/h", "default": 0.5,
             "source": "Reynolds et al., 2006 (calibrated for RL timescale)"},
    "mu_d": {"min": 0.01, "max": 0.5, "unit": "1/h", "default": 0.12,
             "source": "Reynolds et al., 2006 (calibrated for RL timescale)"},
}


class InflammationModel:
    """4-ODE acute inflammation model.

    State variables:
        B: bacteria concentration (normalized, 0-1)
        M: pro-inflammatory mediator level (normalized)
        A: anti-inflammatory mediator level (normalized)
        D: tissue damage (normalized, 0-1, where 1 = organ failure)

    Parameters follow Reynolds et al. (2006) with defaults calibrated
    to produce clinically plausible sepsis trajectories.
    """

    def __init__(
        self,
        k_growth: float = 0.6,
        k_kill: float = 0.5,
        b_max: float = 1.0,
        s_m: float = 0.01,
        k_mb: float = 1.5,
        k_md: float = 0.2,
        mu_m: float = 0.2,
        k_ma: float = 0.5,
        s_a: float = 0.01,
        k_am: float = 0.4,
        k_ad: float = 0.1,
        mu_a: float = 0.05,
        k_dm: float = 0.5,
        mu_d: float = 0.12,
        antibiotic_efficacy: float = 0.0,
    ) -> None:
        self.k_growth = k_growth
        self.k_kill = k_kill
        self.b_max = b_max
        self.s_m = s_m
        self.k_mb = k_mb
        self.k_md = k_md
        self.mu_m = mu_m
        self.k_ma = k_ma
        self.s_a = s_a
        self.k_am = k_am
        self.k_ad = k_ad
        self.mu_a = mu_a
        self.k_dm = k_dm
        self.mu_d = mu_d
        self.antibiotic_efficacy = antibiotic_efficacy

        self._validate_params()

    def _validate_params(self) -> None:
        for name, rng in PARAMETER_RANGES.items():
            val = getattr(self, name, None)
            if val is not None and not (rng["min"] <= val <= rng["max"]):
                raise ValueError(
                    f"Parameter {name}={val} outside range "
                    f"[{rng['min']}, {rng['max']}] ({rng['source']})"
                )

    def derivatives(
        self, t: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute derivatives for the 4-ODE system."""
        B, M, A, D = np.clip(state, 0.0, None)

        sat_b = B / (1.0 + B)

        dB = (self.k_growth * B * (1.0 - B / self.b_max)
              - self.k_kill * M * sat_b
              - self.antibiotic_efficacy * B)

        dM = (self.s_m
              + self.k_mb * sat_b
              + self.k_md * D
              - self.mu_m * M
              - self.k_ma * M * A)

        dA = (self.s_a
              + self.k_am * M
              + self.k_ad * D
              - self.mu_a * A)

        dD = self.k_dm * M / (1.0 + M) - self.mu_d * D

        return np.array([dB, dM, dA, dD])

    def step(
        self, state: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        """Advance the model by dt hours.

        Args:
            state: [B, M, A, D] current state vector.
            dt: time step in hours.

        Returns:
            New state vector after dt hours.
        """
        sol = solve_ivp(
            self.derivatives,
            [0, dt],
            state,
            method="RK45",
            max_step=dt / 2,
            rtol=1e-6,
            atol=1e-9,
        )
        new_state = sol.y[:, -1]
        new_state = np.clip(new_state, 0.0, None)
        new_state[0] = np.clip(new_state[0], 0.0, self.b_max)
        new_state[3] = np.clip(new_state[3], 0.0, 1.0)
        return new_state

    def reset(
        self,
        bacteria_load: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate an initial sepsis state.

        Args:
            bacteria_load: initial bacterial burden (0-1 normalized).
            rng: random generator for stochastic variation.

        Returns:
            Initial [B, M, A, D] state.
        """
        if rng is not None:
            bacteria_load *= rng.uniform(0.8, 1.2)
        bacteria_load = np.clip(bacteria_load, 0.01, self.b_max)
        return np.array([
            bacteria_load,
            self.s_m / self.mu_m,
            self.s_a / self.mu_a,
            0.0,
        ])
