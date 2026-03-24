"""Tests for SOFA score computation."""

import pytest

from sepsisim.models.sofa import (
    cardiovascular_sofa,
    renal_sofa,
    lactate_sofa,
    compute_sofa_score,
)


class TestCardiovascularSOFA:
    def test_normal(self):
        assert cardiovascular_sofa(80.0, 0.0) == 0

    def test_low_map(self):
        assert cardiovascular_sofa(65.0, 0.0) == 1

    def test_any_vasopressor(self):
        assert cardiovascular_sofa(70.0, 0.05) == 2

    def test_high_vasopressor(self):
        assert cardiovascular_sofa(60.0, 0.15) == 3

    def test_very_high_vasopressor(self):
        assert cardiovascular_sofa(50.0, 0.5) == 4


class TestRenalSOFA:
    def test_normal_uo(self):
        assert renal_sofa(0.6) == 0

    def test_reduced_uo(self):
        assert renal_sofa(0.4) == 1

    def test_low_uo(self):
        assert renal_sofa(0.25) == 2

    def test_very_low_uo(self):
        assert renal_sofa(0.15) == 3

    def test_anuria(self):
        assert renal_sofa(0.05) == 4


class TestLactateSOFA:
    def test_normal(self):
        assert lactate_sofa(1.5) == 0

    def test_mild_elevation(self):
        assert lactate_sofa(3.0) == 1

    def test_moderate_elevation(self):
        assert lactate_sofa(5.0) == 2

    def test_high(self):
        assert lactate_sofa(8.0) == 3

    def test_very_high(self):
        assert lactate_sofa(12.0) == 4


class TestComputeSOFA:
    def test_healthy(self):
        score = compute_sofa_score(80.0, 0.0, 0.6, 1.0)
        assert score == 0

    def test_severe(self):
        score = compute_sofa_score(50.0, 0.5, 0.05, 12.0)
        assert score == 12

    def test_moderate(self):
        score = compute_sofa_score(65.0, 0.05, 0.3, 3.0)
        assert 2 <= score <= 6

    def test_score_range(self):
        score = compute_sofa_score(70.0, 0.1, 0.4, 2.5)
        assert 0 <= score <= 12

    def test_individual_components_sum(self):
        m, v, u, l = 60.0, 0.2, 0.15, 7.0
        total = compute_sofa_score(m, v, u, l)
        expected = cardiovascular_sofa(m, v) + renal_sofa(u) + lactate_sofa(l)
        assert total == expected
