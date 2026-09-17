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


# --------------------------------------------------------------------------
# energy_levels
# --------------------------------------------------------------------------

class TestEnergyLevels:
    def test_ground_state_energy_matches_particle_in_a_box_formula(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        levels = energy_levels(box, e_max_eV=12.0, sigma_eV=0.05)
        expected_ground_state = _analytic_ground_state_energy_eV(box)
        assert np.isclose(levels.min(), expected_ground_state, rtol=1e-8)

    def test_levels_are_returned_sorted_ascending(self):
        box = QuantumBox(lx_nm=4.0, ly_nm=6.0, lz_nm=8.0, effective_mass=0.7)
        levels = energy_levels(box, e_max_eV=10.0, sigma_eV=0.05)
        assert np.all(np.diff(levels) >= 0)

    def test_no_level_exceeds_the_requested_cutoff_margin(self):
        e_max, sigma = 8.0, 0.1
        box = QuantumBox(lx_nm=3.0, ly_nm=3.0, lz_nm=3.0, effective_mass=1.0)
        levels = energy_levels(box, e_max_eV=e_max, sigma_eV=sigma)
        assert np.all(levels <= e_max + 4.0 * sigma + 1e-9)

    def test_larger_box_gives_smaller_level_spacing_than_smaller_box(self):
        # Weaker confinement (larger L) -> denser energy levels near the bottom
        # of the spectrum, i.e. smaller spacing between the lowest levels.
        small_box = QuantumBox(lx_nm=2.0, ly_nm=2.0, lz_nm=2.0, effective_mass=1.0)
        large_box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)

        levels_small = energy_levels(small_box, e_max_eV=12.0, sigma_eV=0.05)
        levels_large = energy_levels(large_box, e_max_eV=12.0, sigma_eV=0.05)

        spacing_small = levels_small[1] - levels_small[0]
        spacing_large = levels_large[1] - levels_large[0]
        assert spacing_large < spacing_small

    def test_heavier_effective_mass_gives_smaller_ground_state_energy(self):
        # E ~ 1 / m*, so increasing effective mass should lower the ground
        # state energy (smaller kinetic-energy spacing), all else equal.
        light_box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=0.2)
        heavy_box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=2.0)

        ground_light = energy_levels(light_box, e_max_eV=20.0, sigma_eV=0.05).min()
        ground_heavy = energy_levels(heavy_box, e_max_eV=20.0, sigma_eV=0.05).min()
        assert ground_heavy < ground_light

    def test_rejects_non_positive_e_max(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            energy_levels(box, e_max_eV=0.0, sigma_eV=0.05)

    def test_rejects_non_positive_sigma(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            energy_levels(box, e_max_eV=10.0, sigma_eV=-0.01)


# --------------------------------------------------------------------------
# broadened_dos
# --------------------------------------------------------------------------

class TestBroadenedDOS:
    def test_output_shape_matches_energy_grid(self):
        egrid = np.linspace(0.01, 12.0, 500)
        dos = broadened_dos(np.array([5.0, 6.0]), egrid, sigma_eV=0.05)
        assert dos.shape == egrid.shape

    def test_dos_values_are_finite_and_non_negative(self):
        egrid = np.linspace(0.01, 12.0, 500)
        energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dos = broadened_dos(energies, egrid, sigma_eV=0.1)
        assert np.all(np.isfinite(dos))
        assert np.all(dos >= 0)

    def test_empty_state_array_gives_all_zero_dos(self):
        egrid = np.linspace(0.01, 12.0, 500)
        dos = broadened_dos(np.array([]), egrid, sigma_eV=0.05)
        assert dos.shape == egrid.shape
        assert np.all(dos == 0.0)

    def test_single_state_dos_peaks_at_that_state_energy(self):
        egrid = np.linspace(0.01, 12.0, 5000)
        dos = broadened_dos(np.array([6.0]), egrid, sigma_eV=0.05)
        peak_energy = egrid[np.argmax(dos)]
        assert np.isclose(peak_energy, 6.0, atol=egrid[1] - egrid[0])

    def test_larger_sigma_produces_a_smoother_curve_with_fewer_local_maxima(self):
        # Several closely-spaced discrete levels: small sigma resolves them
        # as separate peaks, large sigma merges them into one smooth bump.
        energies = np.array([5.0, 5.1, 5.2, 5.3, 5.4])
        egrid = np.linspace(4.5, 6.0, 2000)

        dos_sharp = broadened_dos(energies, egrid, sigma_eV=0.01)
        dos_smooth = broadened_dos(energies, egrid, sigma_eV=0.5)

        def n_local_maxima(y):
            return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))

        assert n_local_maxima(dos_sharp) > n_local_maxima(dos_smooth)

    def test_rejects_non_positive_sigma(self):
        egrid = np.linspace(0.01, 12.0, 500)
        with pytest.raises(ValueError):
            broadened_dos(np.array([1.0]), egrid, sigma_eV=0.0)


# --------------------------------------------------------------------------
# fermi_energy
# --------------------------------------------------------------------------

class TestFermiEnergy:
    def test_known_filling_case_fills_two_electrons_per_state(self):
        # 10 evenly spaced states, 10 electrons -> exactly 5 states filled
        # (2 electrons per state, spin degeneracy) -> Ef = energy of 5th state.
        energies = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        ef = fermi_energy(energies, n_electrons=10)
        assert np.isclose(ef, 5.0)

    def test_odd_electron_count_rounds_up_to_next_full_state(self):
        energies = np.arange(1.0, 11.0)
        # 9 electrons needs ceil(9/2) = 5 states filled -> same Ef as 10 electrons.
        ef = fermi_energy(energies, n_electrons=9)
        assert np.isclose(ef, 5.0)

    def test_single_electron_fills_only_the_ground_state(self):
        energies = np.arange(1.0, 11.0)
        ef = fermi_energy(energies, n_electrons=1)
        assert np.isclose(ef, 1.0)

    def test_raises_when_requested_electrons_exceed_available_states(self):
        # Regression test: the original script silently clipped the fill
        # index to the last enumerated state instead of signalling that the
        # energy cutoff was too low to represent the requested filling.
        energies = np.arange(1.0, 11.0)  # only 10 states available
        with pytest.raises(ValueError, match="Not enough enumerated states"):
            fermi_energy(energies, n_electrons=1000)

    def test_rejects_non_positive_electron_count(self):
        energies = np.arange(1.0, 11.0)
        with pytest.raises(ValueError):
            fermi_energy(energies, n_electrons=0)


# --------------------------------------------------------------------------
# fermi_dirac
# --------------------------------------------------------------------------

class TestFermiDirac:
    def test_occupation_approaches_one_well_below_fermi_energy(self):
        occ = fermi_dirac(np.array([-5.0]), fermi_energy_eV=0.0, temperature_K=300.0)
        assert occ[0] > 0.999

    def test_occupation_approaches_zero_well_above_fermi_energy(self):
        occ = fermi_dirac(np.array([5.0]), fermi_energy_eV=0.0, temperature_K=300.0)
        assert occ[0] < 0.001

    def test_occupation_is_exactly_half_at_the_fermi_energy_for_finite_temperature(self):
        occ = fermi_dirac(np.array([2.5]), fermi_energy_eV=2.5, temperature_K=300.0)
        assert np.isclose(occ[0], 0.5)

    def test_zero_temperature_gives_a_sharp_step_function(self):
        energies = np.array([-1.0, 0.0, 1.0])
        occ = fermi_dirac(energies, fermi_energy_eV=0.0, temperature_K=0.0)
        np.testing.assert_allclose(occ, [1.0, 0.5, 0.0])

    def test_higher_temperature_smears_occupation_near_fermi_energy(self):
        e_probe = 0.05  # eV above Ef
        occ_cold = fermi_dirac(np.array([e_probe]), fermi_energy_eV=0.0, temperature_K=50.0)
        occ_hot = fermi_dirac(np.array([e_probe]), fermi_energy_eV=0.0, temperature_K=2000.0)
        # At higher T the step is smeared out, so occupation just above Ef
        # is pulled closer to 0.5 compared to a cold, sharp step.
        assert occ_hot[0] > occ_cold[0]

    def test_does_not_overflow_for_energies_far_from_fermi_level(self):
        occ = fermi_dirac(np.array([1e6, -1e6]), fermi_energy_eV=0.0, temperature_K=1.0)
        assert np.all(np.isfinite(occ))
        # The exponent is clipped at +/-500, so occupation saturates to
        # numerically-negligible-but-not-exactly-zero / not-exactly-one
        # values rather than exact 0.0 / 1.0 -- use a tight absolute
        # tolerance rather than exact equality.
        np.testing.assert_allclose(occ, [0.0, 1.0], atol=1e-100)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError):
            fermi_dirac(np.array([0.0]), fermi_energy_eV=0.0, temperature_K=-1.0)


# --------------------------------------------------------------------------
# calculate_dos (integration of the pieces above)
# --------------------------------------------------------------------------

class TestCalculateDOS:
    def test_returns_expected_energy_grid_and_finite_non_negative_dos(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)

        assert result.energy_eV.shape == (500,)
        assert result.dos.shape == (500,)
        assert np.all(np.isfinite(result.dos))
        assert np.all(result.dos >= 0)
        assert np.isfinite(result.fermi_energy_eV)
        assert result.n_states_enumerated > 0

    def test_custom_n_points_changes_grid_resolution(self):
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result = calculate_dos(
            box, sigma_eV=0.05, electron_density_m3=5.86e28, n_points=100
        )
        assert result.energy_eV.shape == (100,)

    def test_same_inputs_give_identical_results_every_call(self):
        # Reproducibility: no hidden randomness anywhere in the pipeline.
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        result1 = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        result2 = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
        np.testing.assert_array_equal(result1.dos, result2.dos)
        assert result1.fermi_energy_eV == result2.fermi_energy_eV

    def test_raises_clear_error_when_electron_density_too_high_for_cutoff(self):
        # Same bug-reproduction case identified during the Step 1 audit:
        # a light-effective-mass, large box at silver's electron density
        # needs far more filled states than are enumerated below 12 eV.
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
        box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
        with pytest.raises(ValueError):
            calculate_dos(box, **kwargs)
