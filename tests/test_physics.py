# -*- coding: utf-8 -*-
"""Tests for quantum_dos.physics (scientific core, no matplotlib)."""

import numpy as np
import pytest

from quantum_dos.constants import BOLTZMANN, ELECTRON_MASS, EV_TO_J, HBAR
from quantum_dos.models import QuantumBox
from quantum_dos.physics import (
    broadened_dos,
    calculate_dos,
    energy_levels,
    fermi_dirac,
    fermi_energy,
)


def _analytic_ground_state_energy_eV(box: QuantumBox) -> float:
    """Direct evaluation of Eq. (2): E(1,1,1) for a box, in eV."""
    mass_kg = box.effective_mass * ELECTRON_MASS
    l_m = np.array(box.dimensions_nm) * 1e-9
    coeff = (np.pi**2 * HBAR**2) / (2.0 * mass_kg * l_m**2) / EV_TO_J
    return float(np.sum(coeff))  # nx=ny=nz=1


def _count_local_maxima(y: np.ndarray) -> int:
    """Count strict local maxima in a 1-D array.

    A point is a local maximum if it is strictly greater than both of
    its immediate neighbours. Used as a simple proxy for the number of
    resolved peaks in a broadened DOS curve.
    """
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))


def _full_untruncated_dos(
    energies_eV: np.ndarray, egrid_eV: np.ndarray, sigma_eV: float
) -> np.ndarray:
    """Reference DOS: every state's Gaussian summed at every grid point.

    This is the naive, obviously correct broadening (no tail
    truncation). It is deliberately slow and simple so it can serve as
    an independent oracle that the optimized broadened_dos is checked
    against.
    """
    two_sigma_sq = 2.0 * sigma_eV**2
    delta = egrid_eV[:, None] - energies_eV[None, :]
    return np.sum(np.exp(-(delta**2) / two_sigma_sq), axis=1)


# --------------------------------------------------------------------------
# energy_levels
# --------------------------------------------------------------------------

class TestEnergyLevels:
    def test_ground_state_energy_matches_particle_in_a_box_formula(self):
        """The lowest enumerated level must equal the analytic ground state.

        WHY: this is the oracle test for the whole energy spectrum. The
        particle-in-a-box ground state E(1,1,1) has a known closed form,
        so if the numerically enumerated lowest level matches it, we can
        trust that the energy formula and the nm->m unit conversion are
        correct.

        GIVEN a 5x5x5 nm box with the free-electron mass,
        WHEN its energy levels are enumerated,
        THEN the smallest level equals the analytic E(1,1,1), to a tight
        floating-point tolerance rather than exactly, since the two are
        computed by different code paths and will not be bit-identical.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        levels = energy_levels(box, e_max_eV=12.0, sigma_eV=0.05)
        expected_ground_state = _analytic_ground_state_energy_eV(box)
        assert np.isclose(levels.min(), expected_ground_state, rtol=1e-8)

    def test_levels_are_returned_sorted_ascending(self):
        """Enumerated energy levels must come back in ascending order.

        WHY: downstream code relies on this ordering. The Fermi energy is
        found by filling levels from the lowest up, and the broadening
        uses searchsorted on the levels; both are only correct if the
        array is sorted. This test pins that ordering as a guaranteed
        property of energy_levels rather than an accident of how it
        happens to build the array.

        GIVEN a box with three different side lengths and a non-unit mass
        (so the three per-axis energy scales differ and the levels are
        not trivially ordered),
        WHEN its energy levels are enumerated,
        THEN every level is greater than or equal to the one before it.
        """
        box = QuantumBox(lx_nm=4.0, ly_nm=6.0, lz_nm=8.0, effective_mass=0.7)
        levels = energy_levels(box, e_max_eV=10.0, sigma_eV=0.05)
        assert np.all(np.diff(levels) >= 0)

    def test_no_level_exceeds_the_requested_cutoff_margin(self):
        """Enumeration must not return levels above the cutoff plus its margin.

        WHY: energy_levels enumerates states only up to e_max plus a
        margin of 4*sigma, so that the Gaussian tails of states just
        below e_max are still represented on the grid without wasting
        work on states whose contribution is negligible. This test
        guarantees that the upper bound is actually respected -- i.e. the
        filtering step is present and uses the documented margin -- so
        the state count stays bounded and the broadened DOS is not
        silently truncated.

        GIVEN a 3x3x3 nm box with e_max = 8 eV and sigma = 0.1 eV,
        WHEN its energy levels are enumerated,
        THEN no level exceeds e_max + 4*sigma, with a tiny slack to
        absorb floating-point round-off at the boundary.
        """
        e_max, sigma = 8.0, 0.1
        box = QuantumBox(lx_nm=3.0, ly_nm=3.0, lz_nm=3.0, effective_mass=1.0)
        levels = energy_levels(box, e_max_eV=e_max, sigma_eV=sigma)
        assert np.all(levels <= e_max + 4.0 * sigma + 1e-9)

    def test_larger_box_gives_smaller_level_spacing_than_smaller_box(self):
        """A larger box must have more closely spaced low-lying levels.

        WHY: this checks a qualitative physical trend the software exists
        to illustrate -- weaker confinement (a larger box) packs the
        energy levels more densely. It is a direction test: rather than
        pinning an exact number, it fixes the sign of the change between
        two boxes, so it stays meaningful even as other details of the
        model evolve, and it would catch a sign error or an L-dependence
        that went the wrong way.

        GIVEN a small 2x2x2 nm box and a large 10x10x10 nm box, identical
        except for size,
        WHEN the lowest two levels of each are enumerated,
        THEN the gap between them is smaller for the large box than for
        the small one.
        """
        small_box = QuantumBox(lx_nm=2.0, ly_nm=2.0, lz_nm=2.0, effective_mass=1.0)
        large_box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)

        levels_small = energy_levels(small_box, e_max_eV=12.0, sigma_eV=0.05)
        levels_large = energy_levels(large_box, e_max_eV=12.0, sigma_eV=0.05)

        spacing_small = levels_small[1] - levels_small[0]
        spacing_large = levels_large[1] - levels_large[0]
        assert spacing_large < spacing_small

    def test_heavier_effective_mass_gives_smaller_ground_state_energy(self):
        """A heavier effective mass must lower the ground-state energy.

        WHY: the confinement energy scales as 1/m*, so a heavier electron
        sits in a lower ground state for the same box. This is the second
        direction test for the energy model (the first varied the box
        size; this one varies the mass), and together they check that the
        two independent factors in the energy formula both act in the
        physically correct direction. A flipped sign or a mass placed in
        the numerator instead of the denominator would fail here.

        GIVEN two 5x5x5 nm boxes differing only in effective mass, one
        light (0.2 m_e) and one heavy (2.0 m_e),
        WHEN the lowest level of each is enumerated,
        THEN the heavy box has the lower ground-state energy.
        """
        light_box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=0.2)
        heavy_box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=2.0)

        ground_light = energy_levels(light_box, e_max_eV=20.0, sigma_eV=0.05).min()
        ground_heavy = energy_levels(heavy_box, e_max_eV=20.0, sigma_eV=0.05).min()
        assert ground_heavy < ground_light

    def test_rejects_non_positive_e_max(self):
        """A non-positive energy cutoff must be rejected with ValueError.

        WHY: e_max defines the energy window to enumerate; a zero or
        negative cutoff is physically meaningless and would otherwise
        produce an empty or nonsensical spectrum. The course guidance is
        to report invalid input by raising an exception rather than
        asserting or silently degrading, so this test fixes that
        contract: callers are told clearly, via ValueError, when they
        pass a bad cutoff.

        GIVEN a valid box but an energy cutoff of 0 eV,
        WHEN energy levels are requested,
        THEN energy_levels raises ValueError.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            energy_levels(box, e_max_eV=0.0, sigma_eV=0.05)

    def test_rejects_non_positive_sigma(self):
        """A non-positive broadening width must be rejected with ValueError.

        WHY: sigma is the Gaussian broadening width and appears in the
        denominator (exp(-dE^2 / (2 sigma^2))), so a zero or negative
        value would divide by zero or invert the Gaussian. As with the
        e_max check, the contract is to raise ValueError on invalid input
        rather than let a meaningless computation proceed. This is the
        sigma counterpart of the e_max validation test.

        GIVEN a valid box but a broadening width of -0.01 eV,
        WHEN energy levels are requested,
        THEN energy_levels raises ValueError.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            energy_levels(box, e_max_eV=10.0, sigma_eV=-0.01)


# --------------------------------------------------------------------------
# broadened_dos
# --------------------------------------------------------------------------

class TestBroadenedDOS:
    def test_output_shape_matches_energy_grid(self):
        """The broadened DOS must have the same shape as the energy grid.

        WHY: broadened_dos returns one DOS value per energy-grid point,
        and everything downstream (plotting, the Fermi-level lookup, the
        occupied-DOS shading) pairs the two arrays element by element. If
        the shapes ever diverged, those pairings would misalign or crash,
        so this test pins the one-value-per-grid-point contract.

        GIVEN a 500-point energy grid and two discrete levels to broaden,
        WHEN the broadened DOS is computed,
        THEN its shape equals the energy grid's shape.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        dos = broadened_dos(np.array([5.0, 6.0]), egrid, sigma_eV=0.05)
        assert dos.shape == egrid.shape

    def test_dos_values_are_all_finite(self):
        """Every broadened-DOS value must be finite.

        WHY: the DOS is a sum of Gaussians; a NaN or inf would silently
        poison the plot and the DOS-at-Fermi-energy readout, and such a
        value would most likely signal an overflow or a divide-by-zero
        in the broadening. Checking finiteness on its own (separate from
        the non-negativity check) keeps the failure reason unambiguous:
        if this test fails, the problem is a non-finite value, nothing
        else.

        GIVEN five discrete levels on a 500-point grid,
        WHEN the broadened DOS is computed,
        THEN all of its values are finite.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dos = broadened_dos(energies, egrid, sigma_eV=0.1)
        assert np.all(np.isfinite(dos))

    def test_dos_values_are_non_negative(self):
        """Every broadened-DOS value must be non-negative.

        WHY: a density of states is a count of states per energy and
        cannot be negative; the broadened DOS is a sum of non-negative
        Gaussians, so any negative value would indicate a sign error in
        the construction. This is kept separate from the finiteness
        check so that a failure points to exactly one cause -- a negative
        value.

        GIVEN five discrete levels on a 500-point grid,
        WHEN the broadened DOS is computed,
        THEN all of its values are >= 0.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dos = broadened_dos(energies, egrid, sigma_eV=0.1)
        assert np.all(dos >= 0)

    def test_empty_state_array_gives_all_zero_dos(self):
        """With no states to broaden, the DOS must be all zeros.

        WHY: an empty spectrum is a real edge case -- it is what a caller
        gets if every enumerated level fell outside the energy window --
        and the function must degrade gracefully rather than raise or
        return a malformed array. A DOS built from zero Gaussians is
        identically zero, so this test fixes that behavior. The two
        assertions describe a single outcome (a correctly shaped,
        all-zero result): the shape check is only there so the zero check
        is meaningful, so this remains one test for one concept.

        GIVEN an empty array of state energies and a 500-point grid,
        WHEN the broadened DOS is computed,
        THEN the result has the grid's shape and every value is exactly
        zero -- exact, not approximate, because these are untouched zeros
        from array initialization, not the result of floating-point
        arithmetic.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        dos = broadened_dos(np.array([]), egrid, sigma_eV=0.05)
        assert dos.shape == egrid.shape
        assert np.all(dos == 0.0)

    def test_single_state_dos_peaks_at_that_state_energy(self):
        """A single broadened level must peak at that level's energy.

        WHY: this is the simplest meaningful check that the broadening is
        centered correctly. A single Gaussian placed at energy E_0 has
        its maximum at E_0, so the broadened DOS of one state must peak
        there. If the sign of the exponent, or the dE = E_grid - E_state
        offset, were wrong, the peak would land somewhere else and this
        test would catch it.

        GIVEN one state at 6.0 eV broadened onto a fine 5000-point grid,
        WHEN the broadened DOS is computed,
        THEN the grid point of maximum DOS is at 6.0 eV, within one grid
        spacing -- the peak can only be located to the resolution of the
        grid, so the tolerance is the grid step, not an arbitrary number.
        """
        egrid = np.linspace(0.01, 12.0, 5000)
        dos = broadened_dos(np.array([6.0]), egrid, sigma_eV=0.05)
        peak_energy = egrid[np.argmax(dos)]
        assert np.isclose(peak_energy, 6.0, atol=egrid[1] - egrid[0])

    def test_larger_sigma_produces_a_smoother_curve_with_fewer_local_maxima(self):
        """A larger broadening width must smooth out closely spaced peaks.

        WHY: sigma controls how much each discrete level is smeared, and
        the whole point of the broadening is that increasing it merges
        nearby levels into a smoother curve. This is a direction test on
        sigma: with several closely spaced levels, a small sigma resolves
        them as separate bumps while a large sigma blends them into
        essentially one. Counting local maxima is a simple, robust proxy
        for "number of resolved peaks", so a decrease in that count as
        sigma grows confirms the smoothing behaves correctly.

        GIVEN five levels spaced 0.1 eV apart, broadened once with a
        small sigma (0.01 eV) and once with a large sigma (0.5 eV),
        WHEN the number of local maxima of each curve is counted,
        THEN the sharp (small-sigma) curve has strictly more local maxima
        than the smooth (large-sigma) curve.
        """
        energies = np.array([5.0, 5.1, 5.2, 5.3, 5.4])
        egrid = np.linspace(4.5, 6.0, 2000)

        dos_sharp = broadened_dos(energies, egrid, sigma_eV=0.01)
        dos_smooth = broadened_dos(energies, egrid, sigma_eV=0.5)

        assert _count_local_maxima(dos_sharp) > _count_local_maxima(dos_smooth)

    def test_rejects_non_positive_sigma(self):
        """broadened_dos must reject a non-positive broadening width.

        WHY: sigma appears as 1/(2 sigma^2) in the Gaussian, so a zero or
        negative width is undefined (division by zero or an inverted
        Gaussian). broadened_dos validates its own sigma independently of
        energy_levels, because it is part of the public API and may be
        called directly, so the invalid-input contract must hold here
        too.

        GIVEN a single state and a valid grid but sigma = 0,
        WHEN the broadened DOS is requested,
        THEN broadened_dos raises ValueError.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        with pytest.raises(ValueError):
            broadened_dos(np.array([1.0]), egrid, sigma_eV=0.0)

    def test_rejects_non_positive_truncation_sigma(self):
        """broadened_dos must reject a non-positive truncation window.

        WHY: truncation_sigma sets how many standard deviations around
        each level are included before the Gaussian tail is cut off. A
        zero or negative window would include no points (or an inverted
        range) and produce a meaningless or empty DOS, so it is invalid
        input and must raise rather than silently return garbage. This
        guards the knob that controls the performance optimization, so a
        bad value is reported instead of quietly corrupting the result.

        GIVEN a single state and a valid grid but truncation_sigma = 0,
        WHEN the broadened DOS is requested,
        THEN broadened_dos raises ValueError.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        with pytest.raises(ValueError):
            broadened_dos(np.array([1.0]), egrid, sigma_eV=0.05, truncation_sigma=0.0)

    def test_truncated_matches_full_untruncated_sum(self):
        """The truncated broadening must match a full, untruncated sum.

        WHY: this is the oracle test that justifies the performance
        optimization. broadened_dos only evaluates each state's Gaussian
        within +/- truncation_sigma * sigma of a grid point, skipping the
        far tails. The optimization is only legitimate if it does not
        change the result, so we compare it against an independent,
        obviously correct reference -- the full double sum over every
        state and every grid point. With the default 8-sigma window the
        discarded tail is exp(-8^2/2) = exp(-32) ~ 1.3e-14 of a
        Gaussian's peak, so the two agree to about 13 significant
        figures. We therefore assert a tight relative tolerance (1e-12)
        rather than exact equality: the tiny difference is the
        deliberately discarded tail, not a bug, and floating-point sums
        are never bit-identical anyway. Checking several sigma values
        ensures the agreement holds across broadening widths, not just
        one lucky case.

        GIVEN 5000 random levels (fixed seed, so the test is
        deterministic) on a 500-point grid,
        WHEN the truncated and full-sum DOS are computed for several
        sigma values,
        THEN they agree to a relative tolerance of 1e-12 for every sigma.
        """
        rng = np.random.default_rng(0)  # fixed seed -> deterministic test
        energies = np.sort(rng.uniform(0.0, 12.0, size=5000))
        egrid = np.linspace(0.01, 12.0, 500)

        for sigma in (0.01, 0.05, 0.1, 0.3):
            reference = _full_untruncated_dos(energies, egrid, sigma)
            optimized = broadened_dos(energies, egrid, sigma)
            np.testing.assert_allclose(
                optimized, reference, rtol=1e-12, atol=1e-12,
                err_msg=f"truncated DOS differs from full sum at sigma={sigma}",
            )

    def test_smaller_truncation_window_still_captures_the_peak(self):
        """Even a tight truncation window must capture a level's own peak.

        WHY: the truncation only discards the far tails of each Gaussian,
        so however small the window is set, the state sitting exactly on
        a grid point (dE = 0) must still contribute its full peak value
        of 1. This guards against an off-by-one or an over-aggressive
        window that accidentally excluded the centre of the Gaussian --
        which would drop the peak and badly distort the DOS. Using a
        deliberately small window (3 sigma) makes the test meaningful: it
        would still pass trivially with the default 8 sigma.

        GIVEN one state at exactly 6.0 eV (a grid point) and a tight
        3-sigma truncation window,
        WHEN the broadened DOS is computed,
        THEN the peak value is essentially 1.0 (the Gaussian's maximum),
        checked as > 0.99 to allow for the grid not landing perfectly on
        the centre.
        """
        egrid = np.linspace(0.01, 12.0, 500)
        dos = broadened_dos(np.array([6.0]), egrid, sigma_eV=0.05,
                            truncation_sigma=3.0)
        assert dos.max() > 0.99


# --------------------------------------------------------------------------
# fermi_energy
# --------------------------------------------------------------------------

class TestFermiEnergy:
    def test_known_filling_case_fills_two_electrons_per_state(self):
        """Filling N electrons must occupy N/2 states (spin degeneracy).

        WHY: each spatial state holds two electrons (spin up and down),
        so the Fermi energy is the energy of the highest occupied state
        after placing the electrons two at a time. This is a known-answer
        test: ten evenly spaced states filled with ten electrons must
        occupy exactly five states, putting the Fermi energy at the fifth
        one. It pins the central filling rule the Fermi energy depends on.

        GIVEN ten evenly spaced states [1..10] eV and ten electrons,
        WHEN the Fermi energy is computed,
        THEN it equals the energy of the fifth state (5.0 eV).
        """
        energies = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        ef = fermi_energy(energies, n_electrons=10)
        assert np.isclose(ef, 5.0)

    def test_odd_electron_count_rounds_up_to_next_full_state(self):
        """An odd electron count must still occupy a whole top state.

        WHY: with two electrons per state, an odd electron count leaves
        the highest state only half-filled, but it is still the highest
        occupied state and so still sets the Fermi energy. Nine electrons
        therefore occupy ceil(9/2) = 5 states, giving the same Fermi
        energy as ten. This checks the rounding of the half-filled top
        state, a boundary case the simple even-count test does not cover.

        GIVEN ten evenly spaced states [1..10] eV and nine electrons,
        WHEN the Fermi energy is computed,
        THEN it equals the energy of the fifth state (5.0 eV), the same
        as the ten-electron case.
        """
        energies = np.arange(1.0, 11.0)
        ef = fermi_energy(energies, n_electrons=9)
        assert np.isclose(ef, 5.0)

    def test_single_electron_fills_only_the_ground_state(self):
        """A single electron must place the Fermi energy at the lowest state.

        WHY: the smallest non-trivial filling is one electron, which
        occupies only the ground state, so the Fermi energy must be the
        lowest level. This is the lower-limit case of the filling rule
        and guards against an off-by-one in how the highest occupied
        state is indexed (e.g. returning the second state, or an empty
        fill).

        GIVEN ten evenly spaced states [1..10] eV and one electron,
        WHEN the Fermi energy is computed,
        THEN it equals the energy of the first state (1.0 eV).
        """
        energies = np.arange(1.0, 11.0)
        ef = fermi_energy(energies, n_electrons=1)
        assert np.isclose(ef, 1.0)

    def test_raises_when_requested_electrons_exceed_available_states(self):
        """Asking to fill more states than exist must raise, not clip.

        WHY: this is a regression test for a real bug in the original
        prototype. That version silently clipped the fill index to the
        last enumerated state when the electron count exceeded the number
        of available states, which quietly reported a wrong Fermi energy
        whenever the energy cutoff was too low for the requested filling.
        The corrected behaviour is to raise a clear error so the caller
        knows to raise the cutoff rather than trust a bogus number. The
        test is kept so the bug cannot return.

        GIVEN only ten states but a request to place 1000 electrons,
        WHEN the Fermi energy is computed,
        THEN a ValueError is raised whose message reports that not enough
        states were enumerated.
        """
        energies = np.arange(1.0, 11.0)  # only 10 states available
        with pytest.raises(ValueError, match="Not enough enumerated states"):
            fermi_energy(energies, n_electrons=1000)

    def test_rejects_non_positive_electron_count(self):
        """A non-positive electron count must be rejected with ValueError.

        WHY: filling a box with zero or a negative number of electrons is
        physically meaningless and has no well-defined Fermi energy, so
        the function must reject it as invalid input rather than return a
        spurious value. This completes the input-validation contract for
        fermi_energy.

        GIVEN a valid set of states but an electron count of 0,
        WHEN the Fermi energy is computed,
        THEN a ValueError is raised.
        """
        energies = np.arange(1.0, 11.0)
        with pytest.raises(ValueError):
            fermi_energy(energies, n_electrons=0)


# --------------------------------------------------------------------------
# fermi_dirac
# --------------------------------------------------------------------------

class TestFermiDirac:
    def test_occupation_approaches_one_well_below_fermi_energy(self):
        """States well below the Fermi energy must be almost fully occupied.

        WHY: the Fermi-Dirac distribution approaches 1 for energies far
        below the Fermi level -- those states are essentially always
        filled. This checks the low-energy limit of the occupation
        function. It is one of a pair with the high-energy test below;
        together they pin the two asymptotes of the distribution.

        GIVEN an energy 5 eV below the Fermi level at room temperature,
        WHEN the Fermi-Dirac occupation is evaluated,
        THEN the occupation is very close to 1 (> 0.999).
        """
        occ = fermi_dirac(np.array([-5.0]), fermi_energy_eV=0.0, temperature_K=300.0)
        assert occ[0] > 0.999

    def test_occupation_approaches_zero_well_above_fermi_energy(self):
        """States well above the Fermi energy must be almost empty.

        WHY: the Fermi-Dirac distribution approaches 0 for energies far
        above the Fermi level -- those states are essentially never
        filled. This checks the high-energy limit, the counterpart of the
        low-energy test above.

        GIVEN an energy 5 eV above the Fermi level at room temperature,
        WHEN the Fermi-Dirac occupation is evaluated,
        THEN the occupation is very close to 0 (< 0.001).
        """
        occ = fermi_dirac(np.array([5.0]), fermi_energy_eV=0.0, temperature_K=300.0)
        assert occ[0] < 0.001

    def test_occupation_is_exactly_half_at_the_fermi_energy_for_finite_temperature(self):
        """Occupation at the Fermi energy must be exactly one half.

        WHY: by definition of the Fermi level, a state exactly at the
        Fermi energy is half-occupied at any finite temperature -- this
        is the fixed point of the distribution and is independent of
        temperature. Testing it pins the centre of the Fermi-Dirac curve,
        complementing the two asymptote tests.

        GIVEN an energy equal to the Fermi energy (2.5 eV) at room
        temperature,
        WHEN the Fermi-Dirac occupation is evaluated,
        THEN the occupation is 0.5.
        """
        occ = fermi_dirac(np.array([2.5]), fermi_energy_eV=2.5, temperature_K=300.0)
        assert np.isclose(occ[0], 0.5)

    def test_zero_temperature_gives_a_sharp_step_function(self):
        """At absolute zero the occupation must be a sharp step.

        WHY: the zero-temperature limit is a special case the function
        must handle explicitly, because the general formula divides by
        the temperature. At T = 0 every state below the Fermi energy is
        full (1), every state above is empty (0), and the state exactly
        at the Fermi energy is half-filled (0.5). This verifies the
        limiting behaviour is implemented correctly rather than producing
        a divide-by-zero.

        GIVEN one energy below, one at, and one above the Fermi level,
        all at T = 0,
        WHEN the Fermi-Dirac occupation is evaluated,
        THEN the occupations are [1.0, 0.5, 0.0] respectively.
        """
        energies = np.array([-1.0, 0.0, 1.0])
        occ = fermi_dirac(energies, fermi_energy_eV=0.0, temperature_K=0.0)
        np.testing.assert_allclose(occ, [1.0, 0.5, 0.0])

    def test_higher_temperature_smears_occupation_near_fermi_energy(self):
        """Raising the temperature must soften the step near the Fermi level.

        WHY: temperature controls how sharp the occupation step is. At a
        fixed energy just above the Fermi level, a hotter system has more
        thermally excited electrons there, so its occupation is pulled up
        toward 0.5 compared to a colder, sharper step. This is a
        direction test on temperature: it fixes the sign of the change
        rather than an exact value, confirming that temperature enters
        the formula with the correct effect.

        GIVEN a probe energy 0.05 eV above the Fermi level, evaluated at a
        cold (50 K) and a hot (2000 K) temperature,
        WHEN the Fermi-Dirac occupation is evaluated at each,
        THEN the hot occupation is larger than the cold one.
        """
        e_probe = 0.05  # eV above Ef
        occ_cold = fermi_dirac(np.array([e_probe]), fermi_energy_eV=0.0, temperature_K=50.0)
        occ_hot = fermi_dirac(np.array([e_probe]), fermi_energy_eV=0.0, temperature_K=2000.0)
        assert occ_hot[0] > occ_cold[0]

    def test_does_not_overflow_for_energies_far_from_fermi_level(self):
        """Extreme energies must saturate the occupation without overflowing.

        WHY: the Fermi-Dirac formula contains exp((E - Ef)/kT), which for
        a very large energy gap would overflow to infinity and poison the
        result. The implementation clips the exponent to a safe range, so
        this test checks that extreme energies produce finite, saturated
        occupations (0 for far above, 1 for far below) instead of inf or
        NaN. The two assertions describe one concept -- safe saturation
        -- with the finiteness check as the precondition for the value
        check, so this stays a single test. The values are compared with
        a tiny absolute tolerance rather than exact 0.0/1.0, because the
        clipped exponent saturates to negligibly-small-but-nonzero
        numbers, not to exact zero.

        GIVEN energies very far above and below the Fermi level,
        WHEN the Fermi-Dirac occupation is evaluated at T = 1 K,
        THEN all results are finite and equal to [0, 1] to within 1e-100.
        """
        occ = fermi_dirac(np.array([1e6, -1e6]), fermi_energy_eV=0.0, temperature_K=1.0)
        assert np.all(np.isfinite(occ))
        np.testing.assert_allclose(occ, [0.0, 1.0], atol=1e-100)

    def test_rejects_negative_temperature(self):
        """A negative temperature must be rejected with ValueError.

        WHY: a negative absolute temperature is unphysical and would make
        the occupation formula behave nonsensically (states above the
        Fermi level would become more occupied than those below). The
        function must reject it as invalid input. Note that T = 0 is
        allowed and handled separately (the sharp-step test above); only
        strictly negative temperatures are rejected.

        GIVEN a temperature of -1 K,
        WHEN the Fermi-Dirac occupation is requested,
        THEN a ValueError is raised.
        """
        with pytest.raises(ValueError):
            fermi_dirac(np.array([0.0]), fermi_energy_eV=0.0, temperature_K=-1.0)


# --------------------------------------------------------------------------
# calculate_dos (integration of the pieces above)
# --------------------------------------------------------------------------

class TestCalculateDOS:
    def test_result_arrays_have_the_expected_grid_shape(self):
        """calculate_dos must return energy and DOS arrays of grid length.

        WHY: calculate_dos is the top-level entry point that ties the
        pieces together, and its DOSResult must expose an energy grid and
        a DOS array of matching length (the default 500 points) so that
        callers can pair them directly. This is the shape half of the
        output contract, kept separate from the value checks so a failure
        points unambiguously at the array shapes.

        GIVEN a default 5x5x5 nm box at silver's electron density,
        WHEN calculate_dos is called,
        THEN both energy_eV and dos have shape (500,).
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        assert result.energy_eV.shape == (500,)
        assert result.dos.shape == (500,)

    def test_result_dos_is_all_finite(self):
        """The DOS returned by calculate_dos must be finite everywhere.

        WHY: a NaN or inf anywhere in the assembled DOS would corrupt the
        plot and any downstream use. This checks finiteness of the full
        pipeline's output on its own, so a failure means a non-finite
        value and nothing else.

        GIVEN a default 5x5x5 nm box at silver's electron density,
        WHEN calculate_dos is called,
        THEN every DOS value is finite.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        assert np.all(np.isfinite(result.dos))

    def test_result_dos_is_non_negative(self):
        """The DOS returned by calculate_dos must be non-negative.

        WHY: a density of states cannot be negative; this verifies the
        physical validity of the assembled output on its own, separate
        from the finiteness check, so a failure isolates a negative value
        as the single cause.

        GIVEN a default 5x5x5 nm box at silver's electron density,
        WHEN calculate_dos is called,
        THEN every DOS value is >= 0.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        assert np.all(result.dos >= 0)

    def test_result_metadata_is_sane(self):
        """calculate_dos must report a finite Fermi energy and some states.

        WHY: besides the arrays, DOSResult carries a Fermi energy and a
        count of enumerated states. A finite Fermi energy and a positive
        state count are the minimal sanity conditions on that metadata --
        a non-finite Fermi energy or zero states would mean the filling
        or enumeration silently failed. This covers the result's metadata
        separately from its arrays.

        GIVEN a default 5x5x5 nm box at silver's electron density,
        WHEN calculate_dos is called,
        THEN the Fermi energy is finite and at least one state was
        enumerated.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        assert np.isfinite(result.fermi_energy_eV)
        assert result.n_states_enumerated > 0

    def test_custom_n_points_changes_grid_resolution(self):
        """Requesting n_points must set the length of the energy grid.

        WHY: n_points is a user-facing parameter controlling the energy
        grid resolution, so passing a non-default value must actually
        change the number of grid points produced. This confirms the
        parameter is wired through to the grid rather than ignored in
        favour of the default.

        GIVEN a default box but n_points = 100,
        WHEN calculate_dos is called,
        THEN the returned energy grid has 100 points.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(
            box, sigma_eV=0.05, electron_density_m3=5.86e28, n_points=100
        )
        assert result.energy_eV.shape == (100,)

    def test_same_inputs_give_identical_results_every_call(self):
        """Identical inputs must give bit-for-bit identical results.

        WHY: reproducibility is a core requirement -- the same inputs
        must always produce the same numbers, which means there must be
        no hidden randomness or state anywhere in the pipeline. Here
        exact equality is the correct check (not a tolerance): two runs
        of a deterministic computation on identical inputs must return
        bit-identical arrays, and any difference would reveal an
        uncontrolled source of variation.

        GIVEN the same box and parameters used for two separate calls,
        WHEN calculate_dos is run twice,
        THEN the two DOS arrays are exactly equal and the two Fermi
        energies are exactly equal.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result1 = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        result2 = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        np.testing.assert_array_equal(result1.dos, result2.dos)
        assert result1.fermi_energy_eV == result2.fermi_energy_eV

    def test_raises_clear_error_when_electron_density_too_high_for_cutoff(self):
        """A filling that exceeds the enumerated states must raise clearly.

        WHY: this is the end-to-end version of the Fermi-energy
        "not enough states" regression. A large, light-effective-mass
        box at silver's electron density needs far more filled states
        than are enumerated below the default 12 eV cutoff, so
        calculate_dos must surface the same clear error through the full
        pipeline rather than returning a silently clipped Fermi energy.
        It confirms the error propagates all the way to the public entry
        point.

        GIVEN a 20x20x20 nm box with a light effective mass (0.1 m_e) at
        silver's electron density,
        WHEN calculate_dos is called,
        THEN a ValueError is raised reporting that not enough states were
        enumerated.
        """
        box = QuantumBox(lx_nm=20.0, ly_nm=20.0, lz_nm=20.0, effective_mass=0.1)
        with pytest.raises(ValueError, match="Not enough enumerated states"):
            calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(sigma_eV=0.0, electron_density_m3=5.86e28),
            dict(sigma_eV=-1.0, electron_density_m3=5.86e28),
            dict(sigma_eV=0.05, electron_density_m3=0.0),
            dict(sigma_eV=0.05, electron_density_m3=-1.0),
            dict(sigma_eV=0.05, electron_density_m3=5.86e28, e_max_eV=0.0),
            dict(sigma_eV=0.05, electron_density_m3=5.86e28, n_points=1),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        """Each invalid parameter must make calculate_dos raise ValueError.

        WHY: calculate_dos is the public entry point, so every parameter
        it exposes must be validated: a non-positive broadening, a
        non-positive electron density, a non-positive energy cutoff, and
        a grid with too few points are all physically or numerically
        invalid and must be rejected rather than silently mishandled.
        The cases are parametrized because they test one single concept
        -- invalid input is rejected -- and listing them together keeps
        that contract in one place while still reporting which specific
        case failed.

        GIVEN a valid box but one invalid parameter per case,
        WHEN calculate_dos is called,
        THEN it raises ValueError.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            calculate_dos(box, **kwargs)
