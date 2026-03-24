"""FluidResuscitation-v0: IV fluid bolus decisions to restore hemodynamics.

The agent decides how much IV crystalloid fluid to administer each hour
to a septic patient. The goal is to restore MAP above 65 mmHg and maintain
adequate urine output while avoiding fluid overload.

Observation space (7D continuous):
    - MAP (mmHg)
    - Lactate (mmol/L)
    - Urine output (mL/kg/h)
    - Tissue damage (0-1)
    - Intravascular volume excess (L)
    - Bacteria load (normalized)
    - SOFA score (0-12)

Action space (1D continuous):
    - Fluid bolus volume: 0-1000 mL per hour

Reward:
    - Positive for MAP in [65, 90] mmHg range
    - Positive for lactate decrease
    - Negative for fluid overload (volume > 4L)
    - Large negative for death (tissue damage >= 0.9)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sepsisim.models.cardiovascular import CardiovascularModel
from sepsisim.models.inflammation import InflammationModel
from sepsisim.models.lactate import LactateModel
from sepsisim.models.sofa import compute_sofa_score


class FluidResuscitationEnv(gym.Env):
    """Gymnasium environment for IV fluid resuscitation in sepsis."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        severity: str = "medium",
        dt: float = 1.0,
        antibiotic_given: bool = True,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.severity = severity
        self.antibiotic_given = antibiotic_given

        severity_map = {"easy": 0.2, "medium": 0.4, "hard": 0.6}
        self._init_bacteria = severity_map.get(severity, 0.4)

        self.observation_space = spaces.Box(
            low=np.array([20.0, 0.1, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([150.0, 30.0, 2.0, 1.0, 10.0, 1.0, 12.0], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1000.0], dtype=np.float32),
        )

        self._inflammation: InflammationModel | None = None
        self._cardio: CardiovascularModel | None = None
        self._lactate_model: LactateModel | None = None
        self._rng: np.random.Generator | None = None

        self._inf_state: NDArray[np.float64] | None = None
        self._lactate: float = 0.0
        self._volume: float = 0.0
        self._map: float = 0.0
        self._uo: float = 0.0
        self._total_fluid: float = 0.0
        self._step_count: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        abx_eff = 0.5 if self.antibiotic_given else 0.0
        self._inflammation = InflammationModel(antibiotic_efficacy=abx_eff)
        self._cardio = CardiovascularModel()
        self._lactate_model = LactateModel()

        self._inf_state = self._inflammation.reset(
            bacteria_load=self._init_bacteria, rng=self._rng
        )
        self._volume = 0.0
        self._total_fluid = 0.0
        tissue_damage = float(self._inf_state[3])

        self._map = self._cardio.compute_map(
            tissue_damage, self._volume, 0.0, self._rng
        )
        self._lactate = self._lactate_model.reset(
            severity=self._init_bacteria, rng=self._rng
        )
        self._uo = self._cardio.compute_urine_output(self._map, self._rng)
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        assert self._inflammation is not None
        assert self._cardio is not None
        assert self._lactate_model is not None
        assert self._inf_state is not None

        fluid_ml = float(np.clip(action[0], 0.0, 1000.0))
        self._total_fluid += fluid_ml

        self._inf_state = self._inflammation.step(self._inf_state, self.dt)
        tissue_damage = float(self._inf_state[3])
        bacteria = float(self._inf_state[0])

        self._volume = self._cardio.update_volume(
            self._volume, fluid_ml, self.dt
        )
        self._map = self._cardio.compute_map(
            tissue_damage, self._volume, 0.0, self._rng
        )
        self._lactate = self._lactate_model.step(
            self._lactate, self._map, tissue_damage, self.dt
        )
        self._uo = self._cardio.compute_urine_output(self._map, self._rng)

        reward = self._compute_reward(tissue_damage, bacteria)

        self._step_count += 1

        terminated = tissue_damage >= 0.9
        truncated = False

        if terminated:
            reward -= 50.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, tissue_damage: float, bacteria: float) -> float:
        """Reward function emphasizing MAP target and lactate clearance."""
        reward = 0.0

        if 65.0 <= self._map <= 90.0:
            reward += 2.0
        elif self._map < 50.0:
            reward -= 3.0
        elif self._map < 65.0:
            reward -= 1.0 * (65.0 - self._map) / 15.0

        if self._lactate < 2.0:
            reward += 1.0
        elif self._lactate > 4.0:
            reward -= 0.5 * (self._lactate - 4.0)

        if self._uo >= 0.5:
            reward += 0.5

        if self._volume > 4.0:
            reward -= 1.0 * (self._volume - 4.0)

        reward -= 0.1 * tissue_damage

        return reward

    def _get_obs(self) -> NDArray[np.float32]:
        assert self._inf_state is not None
        sofa = compute_sofa_score(
            self._map, 0.0, self._uo, self._lactate
        )
        return np.array([
            self._map,
            self._lactate,
            self._uo,
            float(self._inf_state[3]),
            self._volume,
            float(self._inf_state[0]),
            float(sofa),
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        assert self._inf_state is not None
        return {
            "map_mmhg": self._map,
            "lactate": self._lactate,
            "urine_output": self._uo,
            "tissue_damage": float(self._inf_state[3]),
            "bacteria": float(self._inf_state[0]),
            "total_fluid_ml": self._total_fluid,
            "intravascular_volume": self._volume,
            "step": self._step_count,
        }
