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
        assert np.isclose(HBAR, 1.0545718e-34, rtol=1e-6)

    def test_electron_mass_matches_codata_free_electron_mass(self):
        assert np.isclose(ELECTRON_MASS, 9.1093837e-31, rtol=1e-6)

    def test_ev_to_j_matches_codata_elementary_charge(self):
        assert np.isclose(EV_TO_J, 1.6021766e-19, rtol=1e-6)

    def test_boltzmann_matches_codata_value(self):
        assert np.isclose(BOLTZMANN, 1.380649e-23, rtol=1e-6)

    def test_all_constants_are_positive_and_finite(self):
        for value in (HBAR, ELECTRON_MASS, EV_TO_J, BOLTZMANN, SILVER_ELECTRON_DENSITY_M3):
            assert np.isfinite(value)
            assert value > 0

    def test_silver_electron_density_has_plausible_bulk_metal_magnitude(self):
        # Bulk metal conduction-electron densities are typically 1e28-1e29 m^-3.
        assert 1e27 < SILVER_ELECTRON_DENSITY_M3 < 1e30


# --------------------------------------------------------------------------
# QuantumBox
# --------------------------------------------------------------------------

class TestQuantumBox:
    def test_valid_box_stores_dimensions_and_mass_unchanged(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=6.0, lz_nm=7.0, effective_mass=0.5)
        assert box.dimensions_nm == (5.0, 6.0, 7.0)
        assert box.effective_mass == 0.5

    def test_volume_nm3_is_product_of_side_lengths(self):
        box = QuantumBox(lx_nm=2.0, ly_nm=3.0, lz_nm=4.0, effective_mass=1.0)
        assert np.isclose(box.volume_nm3, 24.0)

    def test_volume_m3_converts_nm3_to_m3_correctly(self):
        box = QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)
        # 1 nm^3 = (1e-9 m)^3 = 1e-27 m^3
        assert np.isclose(box.volume_m3, 1e-27)

    @pytest.mark.parametrize("bad_lx", [0.0, -1.0, -100.0])
    def test_rejects_non_positive_lx(self, bad_lx):
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=bad_lx, ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)

    @pytest.mark.parametrize("bad_mass", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_effective_mass(self, bad_mass):
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=1.0, ly_nm=1.0, lz_nm=1.0, effective_mass=bad_mass)

    def test_rejects_non_finite_dimension(self):
        with pytest.raises(ValueError):
            QuantumBox(lx_nm=float("nan"), ly_nm=1.0, lz_nm=1.0, effective_mass=1.0)


# --------------------------------------------------------------------------
# ConfinementThresholds
# --------------------------------------------------------------------------

class TestConfinementThresholds:
    def test_default_thresholds_match_documented_values(self):
        th = ConfinementThresholds()
        assert th.confined_nm == 2.0
        assert th.bulk_nm == 8.0

    def test_rejects_confined_nm_greater_than_or_equal_to_bulk_nm(self):
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=8.0, bulk_nm=8.0)
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=9.0, bulk_nm=8.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_rejects_non_positive_confined_nm(self, bad_value):
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=bad_value, bulk_nm=8.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_rejects_non_positive_bulk_nm(self, bad_value):
        with pytest.raises(ValueError):
            ConfinementThresholds(confined_nm=2.0, bulk_nm=bad_value)


# --------------------------------------------------------------------------
# DOSResult
# --------------------------------------------------------------------------

class TestDOSResult:
    def test_stores_all_fields_unchanged(self):
        energy = np.array([1.0, 2.0, 3.0])
        dos = np.array([0.1, 0.2, 0.3])
        result = DOSResult(
            energy_eV=energy, dos=dos, fermi_energy_eV=2.0, n_states_enumerated=42
        )
        np.testing.assert_array_equal(result.energy_eV, energy)
        np.testing.assert_array_equal(result.dos, dos)
        assert result.fermi_energy_eV == 2.0
        assert result.n_states_enumerated == 42
