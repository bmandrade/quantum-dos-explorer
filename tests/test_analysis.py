# -*- coding: utf-8 -*-
"""Tests for quantum_dos.analysis."""

import numpy as np
import pytest

from quantum_dos.analysis import analytical_dos, classify_axis, classify_regime
from quantum_dos.models import ConfinementThresholds, QuantumBox

THRESHOLDS = ConfinementThresholds()  # confined_nm=2.0, bulk_nm=8.0


# --------------------------------------------------------------------------
# classify_axis
# --------------------------------------------------------------------------

class TestClassifyAxis:
    def test_axis_shorter_than_confined_threshold_is_confined(self):
        assert classify_axis(1.0, THRESHOLDS) == "confined"

    def test_axis_between_thresholds_is_crossover(self):
        assert classify_axis(5.0, THRESHOLDS) == "crossover"

    def test_axis_longer_than_bulk_threshold_is_bulk_like(self):
        assert classify_axis(10.0, THRESHOLDS) == "bulk-like"

    def test_boundary_values_belong_to_the_documented_side(self):
        # confined_nm=2.0 -> exactly 2.0 is NOT "confined" (strict <), so
        # it falls through to "crossover".
        assert classify_axis(2.0, THRESHOLDS) == "crossover"
        # bulk_nm=8.0 -> exactly 8.0 IS "bulk-like" (the crossover branch
        # only catches strictly-less-than bulk_nm).
        assert classify_axis(8.0, THRESHOLDS) == "bulk-like"


# --------------------------------------------------------------------------
# classify_regime
# --------------------------------------------------------------------------

class TestClassifyRegime:
    def test_all_axes_confined_gives_0d(self):
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "0-D"

    def test_one_bulk_axis_with_others_confined_gives_1d(self):
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "1-D"

    def test_two_bulk_axes_with_one_confined_gives_2d(self):
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=1.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "2-D"

    def test_all_three_bulk_axes_gives_3d(self):
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "3-D"

    def test_regime_is_determined_solely_by_the_count_of_bulk_like_axes(self):
        # classify_regime only counts axes above bulk_nm; it does not
        # additionally require the *other* axes to be confined. So one
        # confined (1 nm) + one crossover (5 nm) + one bulk-like (10 nm)
        # axis still reports "1-D", same as (confined, confined, bulk-like)
        # would. This mirrors the original script's algorithm structure
        # exactly (only the threshold values were unified) and is a known,
        # documented simplification -- see classify_regime's docstring.
        box = QuantumBox(lx_nm=1.0, ly_nm=5.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "1-D"

    def test_all_axes_in_crossover_band_gives_crossover(self):
        box = QuantumBox(lx_nm=4.0, ly_nm=5.0, lz_nm=6.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "crossover"

    def test_custom_thresholds_change_classification(self):
        # With tighter thresholds, a box that was "crossover" under the
        # defaults can become cleanly classified.
        box = QuantumBox(lx_nm=4.0, ly_nm=4.0, lz_nm=4.0, effective_mass=1.0)
        loose = ConfinementThresholds(confined_nm=1.0, bulk_nm=3.0)
        assert classify_regime(box, loose) == "3-D"


# --------------------------------------------------------------------------
# analytical_dos
# --------------------------------------------------------------------------

class TestAnalyticalDOS:
    def test_3d_shape_is_sqrt_of_energy(self):
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
        egrid = np.linspace(0.0, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_allclose(shape, np.sqrt(egrid))

    def test_2d_shape_is_constant(self):
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=1.0, effective_mass=1.0)
        egrid = np.linspace(0.1, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_allclose(shape, np.ones_like(egrid))

    def test_1d_shape_decreases_with_energy(self):
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=10.0, effective_mass=1.0)
        egrid = np.linspace(0.5, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        assert np.all(np.diff(shape) < 0)

    def test_0d_shape_is_all_zero(self):
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        egrid = np.linspace(0.1, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_array_equal(shape, np.zeros_like(egrid))

    def test_output_is_finite_across_the_full_energy_range_for_all_dimensionalities(self):
        boxes = [
            QuantumBox(1.0, 1.0, 1.0, 1.0),   # 0-D
            QuantumBox(1.0, 1.0, 10.0, 1.0),  # 1-D
            QuantumBox(10.0, 10.0, 1.0, 1.0),  # 2-D
            QuantumBox(10.0, 10.0, 10.0, 1.0),  # 3-D
        ]
        egrid = np.linspace(0.0, 12.0, 500)
        for box in boxes:
            shape = analytical_dos(egrid, box, THRESHOLDS)
            assert np.all(np.isfinite(shape))
