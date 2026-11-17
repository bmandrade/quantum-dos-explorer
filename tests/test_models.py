# -*- coding: utf-8 -*-
"""Tests for quantum_dos.constants and quantum_dos.models."""

import numpy as np
import pytest

from quantum_dos import ConfinementThresholds, DOSResult, QuantumBox
from quantum_dos.constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    EV_TO_J,
    HBAR,
    SILVER_ELECTRON_DENSITY_M3,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

class TestConstants:
    def test_hbar_matches_codata_reduced_planck_constant(self):
        """The reduced Planck constant must match its CODATA value.

        WHY: every energy in the model is proportional to hbar^2, so a
        wrong value would rescale the entire spectrum. This pins hbar to
        the accepted CODATA figure. A relative tolerance is used rather
        than exact equality because the constant is stored rounded to a
        fixed number of digits.

        GIVEN the stored HBAR constant,
        WHEN it is compared with the CODATA reduced Planck constant,
        THEN they agree to within a relative tolerance of 1e-6.
        """
        assert np.isclose(HBAR, 1.0545718e-34, rtol=1e-6)

    def test_electron_mass_matches_codata_free_electron_mass(self):
        """The electron mass must match its CODATA value.

        WHY: the effective mass is expressed in units of the free-electron
        mass, so this constant sets the absolute energy scale. Pinning it
        guards against a typo in a value that silently affects every
        result.

        GIVEN the stored ELECTRON_MASS constant,
        WHEN it is compared with the CODATA free-electron mass,
        THEN they agree to within a relative tolerance of 1e-6.
        """
        assert np.isclose(ELECTRON_MASS, 9.1093837e-31, rtol=1e-6)

    def test_ev_to_j_matches_codata_elementary_charge(self):
        """The eV-to-joule factor must match the elementary charge.

        WHY: energies are converted between joules and electron-volts
        using this factor, so an error here would shift every energy by a
        constant scale. It must equal the elementary charge in coulombs.

        GIVEN the stored EV_TO_J constant,
        WHEN it is compared with the CODATA elementary charge,
        THEN they agree to within a relative tolerance of 1e-6.
        """
        assert np.isclose(EV_TO_J, 1.6021766e-19, rtol=1e-6)

    def test_boltzmann_matches_codata_value(self):
        """The Boltzmann constant must match its CODATA value.

        WHY: the Fermi-Dirac occupation depends on kB*T, so this constant
        sets how temperature enters the model. Pinning it protects the
        temperature dependence from a mistyped value.

        GIVEN the stored BOLTZMANN constant,
        WHEN it is compared with the CODATA Boltzmann constant,
        THEN they agree to within a relative tolerance of 1e-6.
        """
        assert np.isclose(BOLTZMANN, 1.380649e-23, rtol=1e-6)

    def test_all_constants_are_positive_and_finite(self):
        """Every physical constant must be a positive, finite number.

        WHY: all of these constants appear in products, ratios, or
        exponents where a zero, negative, or non-finite value would make
        the physics nonsensical. This is a single sanity invariant --
        "all constants are usable positive numbers" -- applied to the
        whole set at once, so a failure flags whichever constant is
        malformed.

        GIVEN the full set of physical constants,
        WHEN each is inspected,
        THEN every one is finite and strictly positive.
        """
        for value in (HBAR, ELECTRON_MASS, EV_TO_J, BOLTZMANN, SILVER_ELECTRON_DENSITY_M3):
            assert np.isfinite(value)
            assert value > 0

    def test_silver_electron_density_has_plausible_bulk_metal_magnitude(self):
        """Silver's electron density must have a bulk-metal magnitude.

        WHY: this illustrative constant is used to fill the box with a
        realistic number of electrons. Rather than pin an exact figure,
        the test checks it lies in the range typical of bulk metal
        conduction-electron densities (about 1e28-1e29 m^-3), which would
        catch a wrong power of ten -- the most likely error for such a
        value.

        GIVEN the stored silver electron density,
        WHEN its magnitude is checked,
        THEN it falls between 1e27 and 1e30 m^-3.
        """
        assert 1e27 < SILVER_ELECTRON_DENSITY_M3 < 1e30


# --------------------------------------------------------------------------
# QuantumBox
# --------------------------------------------------------------------------

class TestQuantumBox:
    def test_valid_box_stores_dimensions_and_mass_unchanged(self):
        """A valid box must store its dimensions and mass as given.

        WHY: QuantumBox is the parameter object passed throughout the
        API, so it must preserve exactly the values it is constructed
        with. This is a single concept -- faithful storage of the inputs
        -- checked across the fields that make up the box.

        GIVEN a box built with distinct side lengths and a non-unit mass,
        WHEN its stored attributes are read back,
        THEN the dimensions and effective mass equal the constructor
        arguments.
        """
        box = QuantumBox(lx_nm=5.0, ly_nm=6.0, lz_nm=7.0, effective_mass=0.5)
        assert box.dimensions_nm == (5.0, 6.0, 7.0)
        assert box.effective_mass == 0.5

    def test_volume_nm3_is_product_of_side_lengths(self):
        """The box volume in nm^3 must be the product of its sides.

        WHY: the volume is used to compute how many electrons fill the
        box, so it must equal Lx*Ly*Lz. This checks the volume property
        against a hand-computed value.

        GIVEN a 2x3x4 nm box,
        WHEN its volume_nm3 is read,
        THEN it equals 24 nm^3.
        """
        box = QuantumBox(lx_nm=2.0, ly_nm=3.0, lz_nm=4.0, effective_mass=1.0)
        assert np.isclose(box.volume_nm3, 24.0)

    def test_volume_m3_converts_nm3_to_m3_correctly(self):
        """The box volume in m^3 must apply the nm->m conversion correctly.

        WHY: electron filling uses the volume in SI units, so the
        nm^3-to-m^3 conversion (a factor of 1e-27) must be applied
        correctly. A unit-conversion slip here would be a cube of the
        linear error, so it is worth pinning explicitly.

        GIVEN a 1x1x1 nm box,
        WHEN its volume_m3 is read,
        THEN it equals 1e-27 m^3.
        """
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        assert np.isclose(box.volume_m3, 1e-27)

    @pytest.mark.parametrize("bad_lx", [0.0, -1.0, -100.0])
    def test_rejects_non_positive_lx(self, bad_lx):
        """A non-positive box dimension must be rejected with ValueError.

        WHY: a zero or negative side length is not a physical box and
        would divide by zero in the energy formula (E ~ 1/L^2), so the
        constructor must reject it as invalid input. Several bad values
        are parametrized because they all exercise the one rule -- side
        lengths must be positive.

        GIVEN a box construction with a non-positive Lx,
        WHEN QuantumBox is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=bad_lx, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)

    @pytest.mark.parametrize("bad_mass", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_effective_mass(self, bad_mass):
        """An invalid effective mass must be rejected with ValueError.

        WHY: the effective mass must be a positive, finite number: zero
        or negative is unphysical and would divide by zero or flip the
        sign of the energy, while NaN or inf would poison every result.
        The parametrized cases cover all of these invalid forms under the
        single rule that the mass must be positive and finite.

        GIVEN a box construction with a non-positive or non-finite mass,
        WHEN QuantumBox is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=bad_mass)

    def test_rejects_non_finite_dimension(self):
        """A non-finite box dimension must be rejected with ValueError.

        WHY: a NaN or infinite side length is not a valid box and would
        propagate non-finite values through every calculation, so it must
        be caught at construction. This complements the non-positive
        dimension test: together they cover the two ways a dimension can
        be invalid.

        GIVEN a box construction with a NaN dimension,
        WHEN QuantumBox is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=float("nan"), ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)


# --------------------------------------------------------------------------
# ConfinementThresholds
# --------------------------------------------------------------------------

class TestConfinementThresholds:
    def test_default_thresholds_match_documented_values(self):
        """The default thresholds must match their documented values.

        WHY: the confinement thresholds decide how a box is classified
        (confined, crossover, bulk-like), and the documentation and GUI
        rely on the stated defaults (2 nm and 8 nm). This pins those
        defaults so they cannot drift away from what is documented.

        GIVEN a ConfinementThresholds built with no arguments,
        WHEN its thresholds are read,
        THEN confined_nm is 2.0 and bulk_nm is 8.0.
        """
        th = ConfinementThresholds()
        assert th.confined_nm == 2.0
        assert th.bulk_nm == 8.0

    def test_rejects_confined_nm_greater_than_or_equal_to_bulk_nm(self):
        """The confined threshold must be strictly below the bulk threshold.

        WHY: the classification only makes sense if the "confined" cutoff
        is below the "bulk-like" cutoff, leaving a crossover band between
        them. An equal or inverted pair would make the crossover region
        empty or negative, so the constructor must reject it. Both the
        equal and the inverted case are checked, since they are the two
        ways the ordering can be violated.

        GIVEN thresholds where confined_nm is equal to or greater than
        bulk_nm,
        WHEN ConfinementThresholds is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=8.0, bulk_nm=8.0)
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=9.0, bulk_nm=8.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_rejects_non_positive_confined_nm(self, bad_value):
        """A non-positive confined threshold must be rejected.

        WHY: a length threshold must be a positive distance; zero or
        negative is meaningless as a confinement cutoff. This enforces
        that rule for confined_nm.

        GIVEN a non-positive confined_nm,
        WHEN ConfinementThresholds is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=bad_value, bulk_nm=8.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_rejects_non_positive_bulk_nm(self, bad_value):
        """A non-positive bulk threshold must be rejected.

        WHY: as with the confined threshold, the bulk cutoff must be a
        positive distance. This is the bulk_nm counterpart of the
        previous test, completing the validation of both thresholds.

        GIVEN a non-positive bulk_nm,
        WHEN ConfinementThresholds is instantiated,
        THEN it raises ValueError.
        """
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=2.0, bulk_nm=bad_value)


# --------------------------------------------------------------------------
# DOSResult
# --------------------------------------------------------------------------

class TestDOSResult:
    def test_stores_all_fields_unchanged(self):
        """A DOSResult must store every field exactly as given.

        WHY: DOSResult is the plain container returned by the public API,
        so its only job is to carry the energy grid, the DOS, the Fermi
        energy, and the state count back to the caller unchanged. This is
        a single concept -- faithful storage -- checked across all four
        fields; exact equality is used because the values should be
        passed through untouched, not recomputed.

        GIVEN a DOSResult built from known arrays and scalars,
        WHEN its fields are read back,
        THEN each field equals exactly what was passed in.
        """
        energy = np.array([1.0, 2.0, 3.0])
        dos = np.array([0.1, 0.2, 0.3])
        result = DOSResult(
            energy_eV=energy, dos=dos, fermi_energy_eV=2.0, n_states_enumerated=42
        )
        np.testing.assert_array_equal(result.energy_eV, energy)
        np.testing.assert_array_equal(result.dos, dos)
        assert result.fermi_energy_eV == 2.0
        assert result.n_states_enumerated == 42
