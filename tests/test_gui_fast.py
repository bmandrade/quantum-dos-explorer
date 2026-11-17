# -*- coding: utf-8 -*-
"""Tests for the DOS caching logic used by the optimized GUI.

The optimized GUI (``quantum_dos.gui_fast``) avoids recomputation by
caching the last DOS result and deciding, from what changed, how much
work actually needs redoing. That decision logic lives in ``DosCache``
and is pure computation -- it has no dependency on matplotlib and is
tested here directly. The GUI rendering itself is a visualization layer
and is checked by hand rather than by automated tests.

The central promise of the cache is correctness: however it decides to
reuse or recompute, the result it returns must always be numerically
identical to calling ``calculate_dos`` directly. These tests pin that
promise, and separately check that the cache does skip work when it
safely can.
"""

import numpy as np
import pytest

from quantum_dos import QuantumBox, calculate_dos
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3
from quantum_dos.gui_fast import DosCache


class TestDosCache:
    def test_first_call_matches_calculate_dos_exactly(self):
        """A fresh cache must return exactly what calculate_dos returns.

        WHY: the cache is only ever allowed to be an optimization, never
        a change in results. On the first call there is nothing cached,
        so it must compute and return the identical DOS, Fermi energy,
        and state count as a direct call. Exact equality is the right
        check: the cache must pass the computed values through untouched.

        GIVEN an empty cache and a 5x5x5 nm box,
        WHEN a DOS is requested from the cache and computed directly,
        THEN the two results are identical in every field.
        """
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        cached = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        direct = calculate_dos(box, sigma_eV=0.05,
                               electron_density_m3=SILVER_ELECTRON_DENSITY_M3)
        np.testing.assert_array_equal(cached.dos, direct.dos)
        assert cached.fermi_energy_eV == direct.fermi_energy_eV
        assert cached.n_states_enumerated == direct.n_states_enumerated

    def test_repeated_identical_call_returns_same_cached_object(self):
        """Asking twice with identical inputs must not recompute.

        WHY: the whole point of the cache is to skip work when nothing
        relevant changed. Requesting the same inputs twice should hand
        back the very same cached object rather than recomputing, so this
        checks object identity (is) as direct evidence that no second
        computation happened.

        GIVEN a cache queried once,
        WHEN it is queried again with identical inputs,
        THEN it returns the exact same object instance.
        """
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        first = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        second = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert first is second

    def test_density_only_change_reuses_curve_but_updates_fermi_energy(self):
        """Changing only the density must reuse the curve but refill Ef.

        WHY: the electron density affects which states are filled (the
        Fermi energy) but not the shape of the DOS curve, so when only
        the density changes the cache can reuse the expensive DOS curve
        and recompute just the Fermi energy. This test verifies both that
        the optimization happens (same curve) and that it stays correct
        (the reused-curve result still matches a full direct call).

        GIVEN a cached result at one density,
        WHEN the same box is queried at a different density,
        THEN the DOS curve is unchanged, the Fermi energy differs, and
        the result still equals a direct calculate_dos call.
        """
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        r1 = cache.get(box, sigma_eV=0.05, density_m3=5.86e28)
        dos1 = r1.dos.copy()
        r2 = cache.get(box, sigma_eV=0.05, density_m3=8.0e28)

        np.testing.assert_array_equal(r2.dos, dos1)
        assert r2.fermi_energy_eV != r1.fermi_energy_eV

        direct = calculate_dos(box, sigma_eV=0.05, electron_density_m3=8.0e28)
        np.testing.assert_array_equal(r2.dos, direct.dos)
        assert np.isclose(r2.fermi_energy_eV, direct.fermi_energy_eV)

    def test_sigma_change_triggers_a_real_recompute(self):
        """Changing the broadening must force a full DOS recompute.

        WHY: sigma changes the shape of every broadened peak, so the
        cached curve is no longer valid and must be recomputed. This
        checks that a sigma change produces a different curve and that
        the recomputed curve still matches a direct call -- i.e. the
        cache invalidates correctly on sigma rather than wrongly reusing
        a stale curve.

        GIVEN a cached result at one sigma,
        WHEN the same box is queried at a different sigma,
        THEN the DOS curve changes and matches a direct calculate_dos
        call.
        """
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        r1 = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(box, sigma_eV=0.20, density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert not np.array_equal(r1.dos, r2.dos)
        direct = calculate_dos(box, sigma_eV=0.20,
                               electron_density_m3=SILVER_ELECTRON_DENSITY_M3)
        np.testing.assert_array_equal(r2.dos, direct.dos)

    def test_geometry_change_triggers_a_real_recompute(self):
        """Changing the box size must force a full DOS recompute.

        WHY: the box dimensions set the whole energy spectrum, so a
        different geometry must not reuse the cached curve. This checks
        that changing the box produces a different DOS, confirming the
        cache keys on geometry.

        GIVEN a cached result for one box,
        WHEN a differently sized box is queried,
        THEN the DOS curve is different.
        """
        cache = DosCache()
        r1 = cache.get(QuantumBox(5, 5, 5, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(QuantumBox(8, 8, 8, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert not np.array_equal(r1.dos, r2.dos)

    def test_mass_change_triggers_a_real_recompute(self):
        """Changing the effective mass must force a full DOS recompute.

        WHY: the effective mass scales the whole energy spectrum, so like
        geometry it must invalidate the cached curve. This checks that
        changing the mass produces a different DOS, confirming the cache
        keys on mass as well.

        GIVEN a cached result for one effective mass,
        WHEN the same box is queried with a different mass,
        THEN the DOS curve is different.
        """
        cache = DosCache()
        r1 = cache.get(QuantumBox(5, 5, 5, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(QuantumBox(5, 5, 5, 0.5), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert not np.array_equal(r1.dos, r2.dos)

    def test_cache_result_always_matches_direct_call_over_a_mixed_sequence(self):
        """Over a realistic mix of changes the cache must never drift.

        WHY: the individual tests check one kind of change at a time, but
        a real session interleaves them -- a density change, then a sigma
        change, then a geometry change, a repeat, a mass change. This
        drives the cache through such a sequence and asserts at every
        step that its result still equals a fresh direct call, so no
        combination of reuse decisions can let the cached result silently
        diverge from the truth.

        GIVEN a cache driven through a mixed sequence of parameter
        changes,
        WHEN each step's result is compared with a direct calculate_dos
        call,
        THEN they agree at every step.
        """
        cache = DosCache()
        steps = [
            (QuantumBox(5, 5, 5, 1.0), 0.05, 5.86e28),   # initial
            (QuantumBox(5, 5, 5, 1.0), 0.05, 7.0e28),    # density only
            (QuantumBox(5, 5, 5, 1.0), 0.10, 7.0e28),    # sigma changed
            (QuantumBox(6, 5, 5, 1.0), 0.10, 7.0e28),    # geometry changed
            (QuantumBox(6, 5, 5, 1.0), 0.10, 7.0e28),    # no change
            (QuantumBox(6, 5, 5, 0.8), 0.10, 7.0e28),    # mass changed
        ]
        for box, sigma, density in steps:
            cached = cache.get(box, sigma_eV=sigma, density_m3=density)
            direct = calculate_dos(box, sigma_eV=sigma,
                                   electron_density_m3=density)
            np.testing.assert_array_equal(cached.dos, direct.dos)
            assert np.isclose(cached.fermi_energy_eV, direct.fermi_energy_eV)

    def test_cache_propagates_calculate_dos_errors(self):
        """Errors from calculate_dos must propagate through the cache.

        WHY: if an input is invalid (here a filling that exceeds the
        enumerated states), calculate_dos raises. The cache must let that
        error propagate rather than swallowing it or, worse, caching a
        broken result. This confirms the failure path is preserved
        through the caching layer.

        GIVEN inputs that make calculate_dos raise,
        WHEN they are requested through the cache,
        THEN the same ValueError is raised.
        """
        cache = DosCache()
        with pytest.raises(ValueError):
            cache.get(QuantumBox(20, 20, 20, 0.1), sigma_eV=0.05,
                      density_m3=SILVER_ELECTRON_DENSITY_M3)
