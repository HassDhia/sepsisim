"""Tests for the inflammation ODE model."""

import numpy as np
import pytest

from sepsisim.models.inflammation import InflammationModel, PARAMETER_RANGES


class TestInflammationModelInit:
    def test_default_construction(self):
        model = InflammationModel()
        assert model.k_growth == 0.6
        assert model.k_kill == 0.5

    def test_custom_parameters(self):
        model = InflammationModel(k_growth=0.8, k_kill=0.3)
        assert model.k_growth == 0.8
        assert model.k_kill == 0.3

    def test_invalid_parameter_raises(self):
        with pytest.raises(ValueError, match="outside range"):
            InflammationModel(k_growth=50.0)

    def test_parameter_ranges_populated(self):
        assert len(PARAMETER_RANGES) >= 10
        for name, rng in PARAMETER_RANGES.items():
            assert "min" in rng
            assert "max" in rng
            assert "unit" in rng
            assert "source" in rng
            assert rng["min"] < rng["max"]


class TestInflammationReset:
    def test_reset_returns_4d(self):
        model = InflammationModel()
        state = model.reset()
        assert state.shape == (4,)

    def test_reset_bacteria_load(self):
        model = InflammationModel()
        state = model.reset(bacteria_load=0.5)
        assert state[0] == pytest.approx(0.5, rel=0.01)

    def test_reset_stochastic(self):
        model = InflammationModel()
        rng = np.random.default_rng(42)
        s1 = model.reset(rng=rng)
        rng2 = np.random.default_rng(99)
        s2 = model.reset(rng=rng2)
        assert not np.allclose(s1, s2)

    def test_reset_deterministic_same_seed(self):
        model = InflammationModel()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = model.reset(rng=rng1)
        s2 = model.reset(rng=rng2)
        np.testing.assert_array_equal(s1, s2)

    def test_reset_initial_damage_zero(self):
        model = InflammationModel()
        state = model.reset()
        assert state[3] == 0.0


class TestInflammationStep:
    def test_step_returns_4d(self):
        model = InflammationModel()
        state = model.reset()
        new_state = model.step(state, dt=1.0)
        assert new_state.shape == (4,)

    def test_step_bacteria_non_negative(self):
        model = InflammationModel()
        state = model.reset(bacteria_load=0.01)
        for _ in range(100):
            state = model.step(state, dt=0.5)
        assert state[0] >= 0.0

    def test_step_damage_bounded(self):
        model = InflammationModel()
        state = model.reset(bacteria_load=0.8)
        for _ in range(50):
            state = model.step(state, dt=1.0)
        assert 0.0 <= state[3] <= 1.0

    def test_antibiotic_reduces_bacteria(self):
        model_no_abx = InflammationModel(antibiotic_efficacy=0.0)
        model_abx = InflammationModel(antibiotic_efficacy=0.3)
        state = np.array([0.5, 0.1, 0.2, 0.0])
        s1 = model_no_abx.step(state.copy(), dt=5.0)
        s2 = model_abx.step(state.copy(), dt=5.0)
        assert s2[0] < s1[0]

    def test_derivatives_shape(self):
        model = InflammationModel()
        state = np.array([0.3, 0.1, 0.2, 0.05])
        deriv = model.derivatives(0.0, state)
        assert deriv.shape == (4,)

    def test_small_dt_stability(self):
        model = InflammationModel()
        state = model.reset(bacteria_load=0.3)
        for _ in range(200):
            state = model.step(state, dt=0.1)
        assert np.all(np.isfinite(state))

    def test_large_bacteria_controlled(self):
        model = InflammationModel(antibiotic_efficacy=0.2)
        state = model.reset(bacteria_load=0.9)
        for _ in range(50):
            state = model.step(state, dt=1.0)
        assert state[0] < 0.9
