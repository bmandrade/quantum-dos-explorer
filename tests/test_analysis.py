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
        """An axis below the confined threshold must be "confined".

        WHY: a direction much shorter than the confinement length behaves
        quantum-mechanically (discrete levels), so the classifier must
        label it "confined". This checks the lowest of the three bands.

        GIVEN an axis length of 1 nm (below confined_nm = 2 nm),
        WHEN the axis is classified,
        THEN the result is "confined".
        """
        assert classify_axis(1.0, THRESHOLDS) == "confined"

    def test_axis_between_thresholds_is_crossover(self):
        """An axis between the two thresholds must be "crossover".

        WHY: between the confined and bulk cutoffs the system is neither
        fully quantum nor fully bulk-like, so the classifier must report
        the intermediate "crossover" band. This checks the middle band.

        GIVEN an axis length of 5 nm (between 2 and 8 nm),
        WHEN the axis is classified,
        THEN the result is "crossover".
        """
        assert classify_axis(5.0, THRESHOLDS) == "crossover"

    def test_axis_longer_than_bulk_threshold_is_bulk_like(self):
        """An axis above the bulk threshold must be "bulk-like".

        WHY: a direction much longer than the bulk length behaves like an
        extended (bulk) dimension, so the classifier must label it
        "bulk-like". This checks the highest of the three bands.

        GIVEN an axis length of 10 nm (above bulk_nm = 8 nm),
        WHEN the axis is classified,
        THEN the result is "bulk-like".
        """
        assert classify_axis(10.0, THRESHOLDS) == "bulk-like"

    def test_boundary_values_belong_to_the_documented_side(self):
        """Values exactly on a threshold must fall on the documented side.

        WHY: boundary behaviour is where off-by-one classification errors
        hide, and the documented convention uses strict "<" comparisons.
        A length exactly at confined_nm is therefore NOT confined (it
        falls through to crossover), and a length exactly at bulk_nm IS
        bulk-like (the crossover branch only catches values strictly below
        bulk_nm). Both boundaries are checked together because they test
        the single concept of consistent edge handling.

        GIVEN axis lengths exactly equal to confined_nm (2 nm) and to
        bulk_nm (8 nm),
        WHEN each is classified,
        THEN 2 nm is "crossover" and 8 nm is "bulk-like".
        """
        assert classify_axis(2.0, THRESHOLDS) == "crossover"
        assert classify_axis(8.0, THRESHOLDS) == "bulk-like"


# --------------------------------------------------------------------------
# classify_regime
# --------------------------------------------------------------------------

class TestClassifyRegime:
    def test_all_axes_confined_gives_0d(self):
        """A box confined in all three axes must be classified 0-D.

        WHY: when every direction is quantum-confined the system is a
        quantum dot, i.e. zero-dimensional. This checks the fully
        confined corner of the classification.

        GIVEN a 1x1x1 nm box (all axes confined),
        WHEN the regime is classified,
        THEN the result is "0-D".
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "0-D"

    def test_one_bulk_axis_with_others_confined_gives_1d(self):
        """A box extended along one axis only must be classified 1-D.

        WHY: one bulk-like direction with the other two confined is a
        quantum wire -- one-dimensional. This checks that a single
        extended axis gives 1-D.

        GIVEN a 1x1x10 nm box (one bulk-like axis),
        WHEN the regime is classified,
        THEN the result is "1-D".
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "1-D"

    def test_two_bulk_axes_with_one_confined_gives_2d(self):
        """A box extended along two axes must be classified 2-D.

        WHY: two bulk-like directions with one confined is a quantum well
        -- two-dimensional. This checks the two-extended-axis case.

        GIVEN a 10x10x1 nm box (two bulk-like axes),
        WHEN the regime is classified,
        THEN the result is "2-D".
        """
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=1.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "2-D"

    def test_all_three_bulk_axes_gives_3d(self):
        """A box extended along all three axes must be classified 3-D.

        WHY: three bulk-like directions is a bulk solid --
        three-dimensional. This checks the fully extended corner,
        completing the set of clean 0/1/2/3-D cases.

        GIVEN a 10x10x10 nm box (all axes bulk-like),
        WHEN the regime is classified,
        THEN the result is "3-D".
        """
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "3-D"

    def test_regime_is_determined_solely_by_the_count_of_bulk_like_axes(self):
        """The regime must depend only on the number of bulk-like axes.

        WHY: this documents and locks in a deliberate simplification of
        the classifier. It counts how many axes are bulk-like and does
        not additionally require the remaining axes to be confined, so a
        box with one confined, one crossover, and one bulk-like axis is
        still reported as 1-D -- the same as (confined, confined,
        bulk-like). This mirrors the original prototype's algorithm (only
        the threshold values were unified) and is called out here so the
        behaviour is understood rather than mistaken for a bug.

        GIVEN a 1x5x10 nm box (one confined, one crossover, one bulk-like
        axis),
        WHEN the regime is classified,
        THEN the result is "1-D", decided purely by the single bulk-like
        axis.
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=5.0, lz_nm=10.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "1-D"

    def test_all_axes_in_crossover_band_gives_crossover(self):
        """A box with no bulk-like or confined axes must be "crossover".

        WHY: when no axis is bulk-like and not all axes are confined, the
        box does not fit a clean integer dimensionality and must be
        reported as "crossover". This checks that the catch-all branch is
        reached when every axis lies in the intermediate band.

        GIVEN a 4x5x6 nm box (all axes in the crossover band),
        WHEN the regime is classified,
        THEN the result is "crossover".
        """
        box = QuantumBox(lx_nm=4.0, ly_nm=5.0, lz_nm=6.0, effective_mass=1.0)
        assert classify_regime(box, THRESHOLDS) == "crossover"

    def test_custom_thresholds_change_classification(self):
        """Custom thresholds must change how a box is classified.

        WHY: the thresholds are a user-configurable input, so the same
        box must be able to classify differently under different cutoffs.
        A 4 nm cube is "crossover" under the defaults but becomes a clean
        3-D case under tighter thresholds. This confirms the classifier
        actually uses the thresholds it is given rather than hard-coded
        constants.

        GIVEN a 4x4x4 nm box and thresholds confined_nm=1, bulk_nm=3,
        WHEN the regime is classified with those custom thresholds,
        THEN the result is "3-D".
        """
        box = QuantumBox(lx_nm=4.0, ly_nm=4.0, lz_nm=4.0, effective_mass=1.0)
        loose = ConfinementThresholds(confined_nm=1.0, bulk_nm=3.0)
        assert classify_regime(box, loose) == "3-D"


# --------------------------------------------------------------------------
# analytical_dos
# --------------------------------------------------------------------------

class TestAnalyticalDOS:
    def test_3d_shape_is_sqrt_of_energy(self):
        """The 3-D analytical DOS must follow a square-root law.

        WHY: the free-electron DOS of a bulk (3-D) system is proportional
        to sqrt(E). For a box classified 3-D, analytical_dos must return
        exactly that shape, so this compares it against sqrt(E) on a small
        grid.

        GIVEN a 10x10x10 nm (3-D) box and a short energy grid,
        WHEN the analytical DOS is evaluated,
        THEN it equals sqrt(E) at every grid point.
        """
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
        egrid = np.linspace(0.0, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_allclose(shape, np.sqrt(egrid))

    def test_2d_shape_is_constant(self):
        """The 2-D analytical DOS must be energy-independent.

        WHY: the free-electron DOS of a 2-D system is a constant (a step
        that is flat above the band edge). For a box classified 2-D,
        analytical_dos must return a constant shape, checked here as all
        ones.

        GIVEN a 10x10x1 nm (2-D) box and a short energy grid,
        WHEN the analytical DOS is evaluated,
        THEN every value is the same constant.
        """
        box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=1.0, effective_mass=1.0)
        egrid = np.linspace(0.1, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_allclose(shape, np.ones_like(egrid))

    def test_1d_shape_decreases_with_energy(self):
        """The 1-D analytical DOS must decrease with energy.

        WHY: the free-electron DOS of a 1-D system falls off as
        1/sqrt(E), so it must be monotonically decreasing. Rather than
        pin the exact 1/sqrt(E) values, this checks the qualitative shape
        -- that the DOS strictly decreases across the grid -- which is the
        defining feature of the 1-D case and robust to the small energy
        floor used to avoid a singularity at E = 0.

        GIVEN a 1x1x10 nm (1-D) box and a short energy grid above zero,
        WHEN the analytical DOS is evaluated,
        THEN each value is smaller than the one before it.
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=10.0, effective_mass=1.0)
        egrid = np.linspace(0.5, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        assert np.all(np.diff(shape) < 0)

    def test_0d_shape_is_all_zero(self):
        """The 0-D analytical DOS must be identically zero.

        WHY: a 0-D system has a discrete (delta-function) spectrum with no
        continuous analytical DOS, so analytical_dos returns zeros for a
        fully confined box, signalling that the smooth analytical limit
        does not apply. Exact equality to zero is the correct check here,
        since the function returns literal zeros rather than a computed
        value.

        GIVEN a 1x1x1 nm (0-D) box and a short energy grid,
        WHEN the analytical DOS is evaluated,
        THEN every value is exactly zero.
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        egrid = np.linspace(0.1, 4.0, 5)
        shape = analytical_dos(egrid, box, THRESHOLDS)
        np.testing.assert_array_equal(shape, np.zeros_like(egrid))

    def test_output_is_finite_across_the_full_energy_range_for_all_dimensionalities(self):
        """The analytical DOS must stay finite for every dimensionality.

        WHY: the 1-D shape (1/sqrt(E)) could blow up near E = 0 if the
        singularity were not guarded, so this checks that for all four
        dimensionalities the analytical DOS is finite across the whole
        energy range including the lowest energies. It guards against an
        unguarded division producing inf or NaN for any regime.

        GIVEN one representative box for each of 0-, 1-, 2-, and 3-D and a
        full 0-12 eV grid,
        WHEN the analytical DOS is evaluated for each,
        THEN every value is finite in all four cases.
        """
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
