"""Tests for the cardiovascular model."""

import numpy as np
import pytest

from sepsisim.models.cardiovascular import CardiovascularModel, PARAMETER_RANGES


class TestCardiovascularInit:
    def test_default_construction(self):
        model = CardiovascularModel()
        assert model.map_baseline == 75.0
        assert model.map_target == 65.0

    def test_parameter_ranges_exist(self):
        assert len(PARAMETER_RANGES) >= 5
        for name, rng in PARAMETER_RANGES.items():
            assert rng["min"] < rng["max"]


class TestComputeMAP:
    def test_baseline_no_damage(self):
        model = CardiovascularModel()
        m = model.compute_map(0.0, 0.0, 0.0)
        assert m == pytest.approx(75.0, abs=1.0)

    def test_damage_reduces_map(self):
        model = CardiovascularModel()
        m0 = model.compute_map(0.0, 0.0, 0.0)
        m1 = model.compute_map(0.5, 0.0, 0.0)
        assert m1 < m0

    def test_fluid_increases_map(self):
        model = CardiovascularModel()
        m0 = model.compute_map(0.3, 0.0, 0.0)
        m1 = model.compute_map(0.3, 2.0, 0.0)
        assert m1 > m0

    def test_vasopressor_increases_map(self):
        model = CardiovascularModel()
        m0 = model.compute_map(0.3, 0.0, 0.0)
        m1 = model.compute_map(0.3, 0.0, 0.2)
        assert m1 > m0

    def test_map_bounded(self):
        model = CardiovascularModel()
        m = model.compute_map(1.0, -1.0, 0.0)
        assert 20.0 <= m <= 150.0

    def test_map_with_noise(self):
        model = CardiovascularModel()
        rng = np.random.default_rng(42)
        values = [model.compute_map(0.3, 1.0, 0.1, rng) for _ in range(100)]
        assert np.std(values) > 0.1

    def test_fluid_diminishing_returns(self):
        model = CardiovascularModel()
        m1 = model.compute_map(0.3, 1.0, 0.0)
        m2 = model.compute_map(0.3, 2.0, 0.0)
        m3 = model.compute_map(0.3, 5.0, 0.0)
        gain_1_2 = m2 - m1
        gain_2_5 = m3 - m2
        assert gain_2_5 / 3.0 < gain_1_2


class TestUpdateVolume:
    def test_fluid_increases_volume(self):
        model = CardiovascularModel()
        v = model.update_volume(0.0, 500.0)
        assert v > 0.0

    def test_redistribution_decreases_volume(self):
        model = CardiovascularModel()
        v = model.update_volume(3.0, 0.0, dt=1.0)
        assert v < 3.0

    def test_volume_bounded(self):
        model = CardiovascularModel()
        v = model.update_volume(0.0, 20000.0)
        assert v <= 10.0


class TestUrineOutput:
    def test_low_map_low_uo(self):
        model = CardiovascularModel()
        uo = model.compute_urine_output(40.0)
        assert uo < 0.1

    def test_normal_map_adequate_uo(self):
        model = CardiovascularModel()
        uo = model.compute_urine_output(80.0)
        assert uo >= 0.3

    def test_uo_bounded(self):
        model = CardiovascularModel()
        uo = model.compute_urine_output(200.0)
        assert 0.0 <= uo <= 2.0


class TestCardiovascularReset:
    def test_reset_returns_dict(self):
        model = CardiovascularModel()
        state = model.reset()
        assert "intravascular_volume" in state
        assert "vasopressor_dose" in state
        assert "map_mmhg" in state
