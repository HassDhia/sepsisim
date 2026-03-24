"""Tests for the lactate kinetics model."""

import numpy as np
import pytest

from sepsisim.models.lactate import LactateModel, PARAMETER_RANGES


class TestLactateInit:
    def test_default_construction(self):
        model = LactateModel()
        assert model.baseline_lactate == 1.0
        assert model.clearance_rate == 0.3

    def test_parameter_ranges(self):
        assert len(PARAMETER_RANGES) >= 4
        for rng in PARAMETER_RANGES.values():
            assert rng["min"] < rng["max"]


class TestLactateStep:
    def test_low_map_increases_lactate(self):
        model = LactateModel()
        l1 = model.step(2.0, 40.0, 0.2, dt=1.0)
        assert l1 > 2.0

    def test_high_map_decreases_lactate(self):
        model = LactateModel()
        l1 = model.step(5.0, 80.0, 0.0, dt=1.0)
        assert l1 < 5.0

    def test_damage_increases_lactate(self):
        model = LactateModel()
        l0 = model.step(2.0, 70.0, 0.0, dt=1.0)
        l1 = model.step(2.0, 70.0, 0.5, dt=1.0)
        assert l1 > l0

    def test_lactate_bounded(self):
        model = LactateModel()
        l = model.step(0.01, 20.0, 0.9, dt=10.0)
        assert 0.1 <= l <= 30.0

    def test_lactate_non_negative(self):
        model = LactateModel()
        l = model.step(0.5, 100.0, 0.0, dt=10.0)
        assert l >= 0.1

    def test_stable_at_baseline(self):
        model = LactateModel()
        l = model.step(1.0, 80.0, 0.0, dt=0.1)
        assert abs(l - 1.0) < 0.5


class TestLactateReset:
    def test_reset_returns_float(self):
        model = LactateModel()
        l = model.reset(severity=0.5)
        assert isinstance(l, float)
        assert l > 0

    def test_severity_affects_initial(self):
        model = LactateModel()
        l_low = model.reset(severity=0.1)
        l_high = model.reset(severity=0.9)
        assert l_high > l_low

    def test_stochastic_reset(self):
        model = LactateModel()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        l1 = model.reset(severity=0.5, rng=rng1)
        l2 = model.reset(severity=0.5, rng=rng2)
        assert l1 != l2

    def test_reset_bounded(self):
        model = LactateModel()
        l = model.reset(severity=1.0)
        assert 0.5 <= l <= 15.0
