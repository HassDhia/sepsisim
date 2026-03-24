"""VasopressorTitration-v0: Norepinephrine dosing to maintain MAP in septic shock.

The agent titrates vasopressor (norepinephrine) infusion rate to maintain
MAP above 65 mmHg while minimizing excessive vasoconstriction and organ
ischemia risk.

Observation space (8D continuous):
    - MAP (mmHg)
    - Lactate (mmol/L)
    - Urine output (mL/kg/h)
    - Tissue damage (0-1)
    - Current vasopressor dose (mcg/kg/min)
    - Bacteria load (normalized)
    - SOFA score (0-12)
    - Hours on vasopressors

Action space (1D continuous):
    - Dose change: -0.1 to +0.1 mcg/kg/min per hour

Reward:
    - Positive for MAP in [65, 85] mmHg range
    - Negative for excessive vasopressor dose (ischemia risk)
    - Negative for MAP < 55 or > 100 (hypo/hypertension)
    - Large negative for death
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


class VasopressorTitrationEnv(gym.Env):
    """Gymnasium environment for vasopressor titration in septic shock."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        severity: str = "medium",
        dt: float = 1.0,
        antibiotic_given: bool = True,
        fluid_protocol: float = 250.0,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.severity = severity
        self.antibiotic_given = antibiotic_given
        self.fluid_protocol = fluid_protocol

        severity_map = {"easy": 0.25, "medium": 0.45, "hard": 0.65}
        self._init_bacteria = severity_map.get(severity, 0.45)

        self.observation_space = spaces.Box(
            low=np.array([20.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([150.0, 30.0, 2.0, 1.0, 2.0, 1.0, 12.0, 72.0], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array([-0.1], dtype=np.float32),
            high=np.array([0.1], dtype=np.float32),
        )

        self._inflammation: InflammationModel | None = None
        self._cardio: CardiovascularModel | None = None
        self._lactate_model: LactateModel | None = None
        self._rng: np.random.Generator | None = None

        self._inf_state: NDArray[np.float64] | None = None
        self._lactate: float = 0.0
        self._volume: float = 0.0
        self._vaso_dose: float = 0.0
        self._map: float = 0.0
        self._uo: float = 0.0
        self._step_count: int = 0
        self._hours_on_vaso: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        abx_eff = 0.15 if self.antibiotic_given else 0.0
        self._inflammation = InflammationModel(antibiotic_efficacy=abx_eff)
        self._cardio = CardiovascularModel()
        self._lactate_model = LactateModel()

        self._inf_state = self._inflammation.reset(
            bacteria_load=self._init_bacteria, rng=self._rng
        )
        self._volume = 1.0
        self._vaso_dose = 0.05
        self._hours_on_vaso = 0.0
        tissue_damage = float(self._inf_state[3])

        self._map = self._cardio.compute_map(
            tissue_damage, self._volume, self._vaso_dose, self._rng
        )
        self._lactate = self._lactate_model.reset(
            severity=self._init_bacteria, rng=self._rng
        )
        self._uo = self._cardio.compute_urine_output(self._map, self._rng)
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        assert self._inflammation is not None
        assert self._cardio is not None
        assert self._lactate_model is not None
        assert self._inf_state is not None

        dose_change = float(np.clip(action[0], -0.1, 0.1))
        self._vaso_dose = float(np.clip(self._vaso_dose + dose_change, 0.0, 2.0))

        if self._vaso_dose > 0:
            self._hours_on_vaso += self.dt

        self._inf_state = self._inflammation.step(self._inf_state, self.dt)
        tissue_damage = float(self._inf_state[3])

        self._volume = self._cardio.update_volume(
            self._volume, self.fluid_protocol, self.dt
        )
        self._map = self._cardio.compute_map(
            tissue_damage, self._volume, self._vaso_dose, self._rng
        )
        self._lactate = self._lactate_model.step(
            self._lactate, self._map, tissue_damage, self.dt
        )
        self._uo = self._cardio.compute_urine_output(self._map, self._rng)

        reward = self._compute_reward(tissue_damage)

        self._step_count += 1

        terminated = tissue_damage >= 0.9
        truncated = False

        if terminated:
            reward -= 50.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self, tissue_damage: float) -> float:
        reward = 0.0

        if 65.0 <= self._map <= 85.0:
            reward += 2.0
        elif self._map < 55.0:
            reward -= 3.0
        elif self._map < 65.0:
            reward -= 1.5 * (65.0 - self._map) / 10.0
        elif self._map > 100.0:
            reward -= 0.5 * (self._map - 100.0) / 20.0

        if self._lactate < 2.0:
            reward += 1.0
        elif self._lactate > 4.0:
            reward -= 0.5 * (self._lactate - 4.0)

        if self._vaso_dose > 0.5:
            reward -= 0.5 * (self._vaso_dose - 0.5)

        if self._uo >= 0.5:
            reward += 0.5

        reward -= 0.1 * tissue_damage

        return reward

    def _get_obs(self) -> NDArray[np.float32]:
        assert self._inf_state is not None
        sofa = compute_sofa_score(
            self._map, self._vaso_dose, self._uo, self._lactate
        )
        return np.array([
            self._map,
            self._lactate,
            self._uo,
            float(self._inf_state[3]),
            self._vaso_dose,
            float(self._inf_state[0]),
            float(sofa),
            self._hours_on_vaso,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        assert self._inf_state is not None
        return {
            "map_mmhg": self._map,
            "lactate": self._lactate,
            "urine_output": self._uo,
            "tissue_damage": float(self._inf_state[3]),
            "bacteria": float(self._inf_state[0]),
            "vasopressor_dose": self._vaso_dose,
            "hours_on_vasopressors": self._hours_on_vaso,
            "step": self._step_count,
        }
