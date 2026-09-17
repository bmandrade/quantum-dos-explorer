# -*- coding: utf-8 -*-
"""Tests for the optimized GUI (quantum_dos.gui_fast).

Two groups:

- ``TestDosCache`` tests the pure caching logic directly (no matplotlib
  needed): it must return results numerically identical to calling
  ``calculate_dos`` directly, while skipping recomputation when it can.
- ``TestGuiFastSmoke`` builds the real optimized GUI on the Agg backend
  and drives its callbacks, mirroring the reference GUI smoke test.
"""

import numpy as np
import pytest

from quantum_dos import QuantumBox, calculate_dos
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3
from quantum_dos.gui_fast import DosCache


# --------------------------------------------------------------------------
# DosCache -- pure logic, no matplotlib
# --------------------------------------------------------------------------

class TestDosCache:
    def test_first_call_matches_calculate_dos_exactly(self):
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        cached = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        direct = calculate_dos(box, sigma_eV=0.05,
                               electron_density_m3=SILVER_ELECTRON_DENSITY_M3)
        np.testing.assert_array_equal(cached.dos, direct.dos)
        assert cached.fermi_energy_eV == direct.fermi_energy_eV
        assert cached.n_states_enumerated == direct.n_states_enumerated

    def test_repeated_identical_call_returns_same_cached_object(self):
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        first = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        second = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        # Nothing changed -> the very same object is handed back (no recompute).
        assert first is second

    def test_density_only_change_reuses_curve_but_updates_fermi_energy(self):
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        r1 = cache.get(box, sigma_eV=0.05, density_m3=5.86e28)
        dos1 = r1.dos.copy()
        r2 = cache.get(box, sigma_eV=0.05, density_m3=8.0e28)

        # DOS curve identical (reused), Fermi energy updated.
        np.testing.assert_array_equal(r2.dos, dos1)
        assert r2.fermi_energy_eV != r1.fermi_energy_eV

        # ... and the reused-curve result still matches a direct full call.
        direct = calculate_dos(box, sigma_eV=0.05, electron_density_m3=8.0e28)
        np.testing.assert_array_equal(r2.dos, direct.dos)
        assert np.isclose(r2.fermi_energy_eV, direct.fermi_energy_eV)

    def test_sigma_change_triggers_a_real_recompute(self):
        box = QuantumBox(5, 5, 5, 1.0)
        cache = DosCache()
        r1 = cache.get(box, sigma_eV=0.05, density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(box, sigma_eV=0.20, density_m3=SILVER_ELECTRON_DENSITY_M3)
        # Different broadening -> different curve, and it matches a direct call.
        assert not np.array_equal(r1.dos, r2.dos)
        direct = calculate_dos(box, sigma_eV=0.20,
                               electron_density_m3=SILVER_ELECTRON_DENSITY_M3)
        np.testing.assert_array_equal(r2.dos, direct.dos)

    def test_geometry_change_triggers_a_real_recompute(self):
        cache = DosCache()
        r1 = cache.get(QuantumBox(5, 5, 5, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(QuantumBox(8, 8, 8, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert not np.array_equal(r1.dos, r2.dos)

    def test_mass_change_triggers_a_real_recompute(self):
        cache = DosCache()
        r1 = cache.get(QuantumBox(5, 5, 5, 1.0), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        r2 = cache.get(QuantumBox(5, 5, 5, 0.5), sigma_eV=0.05,
                       density_m3=SILVER_ELECTRON_DENSITY_M3)
        assert not np.array_equal(r1.dos, r2.dos)

    def test_cache_result_always_matches_direct_call_over_a_mixed_sequence(self):
        # Drive the cache through a realistic mixed sequence of changes and
        # assert that at every step it agrees with a fresh direct call.
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
        # A curve input that makes calculate_dos raise must still raise
        # through the cache (not be swallowed or mis-cached).
        cache = DosCache()
        with pytest.raises(ValueError):
            cache.get(QuantumBox(20, 20, 20, 0.1), sigma_eV=0.05,
                      density_m3=SILVER_ELECTRON_DENSITY_M3)


# --------------------------------------------------------------------------
# Headless smoke test of the optimized GUI
# --------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

from quantum_dos.gui_fast import build_app  # noqa: E402


@pytest.fixture
def app():
    handles = build_app(plt, Slider, Button)
    yield handles
    plt.close(handles["fig"])


class TestGuiFastSmoke:
    def test_application_starts_and_populates_dos(self, app):
        assert app["fig"] is not None
        ydata = app["artists"]["line_dos"].get_ydata()
        assert np.any(ydata > 0)
        assert np.all(np.isfinite(ydata))

    def test_changing_dimension_updates_dos_after_compute(self, app):
        before = app["artists"]["line_dos"].get_ydata().copy()
        app["sliders"]["lx"].set_val(12.0)
        # The slider callback only *schedules* a debounced recompute; call
        # the compute directly to get the deterministic post-state.
        app["compute_and_draw"]()
        after = app["artists"]["line_dos"].get_ydata()
        assert not np.array_equal(before, after)

    def test_changing_temperature_keeps_curve_finite(self, app):
        app["sliders"]["temp"].set_val(900.0)
        app["compute_and_draw"]()
        assert np.all(np.isfinite(app["artists"]["line_dos"].get_ydata()))

    def test_changing_density_moves_fermi_line_without_changing_curve(self, app):
        curve_before = app["artists"]["line_dos"].get_ydata().copy()
        ef_before = app["artists"]["vline_ef"].get_xdata()[0]
        app["sliders"]["density"].set_val(12.0)
        app["compute_and_draw"]()
        curve_after = app["artists"]["line_dos"].get_ydata()
        ef_after = app["artists"]["vline_ef"].get_xdata()[0]
        # Density changes Ef but not the DOS curve.
        np.testing.assert_array_equal(curve_before, curve_after)
        assert not np.isclose(ef_before, ef_after)

    def test_reset_restores_defaults(self, app):
        app["sliders"]["lx"].set_val(15.0)
        app["reset"]()
        assert np.isclose(app["sliders"]["lx"].val, 5.0)

    def test_invalid_parameters_show_message_not_crash(self, app):
        app["sliders"]["lx"].set_val(20.0)
        app["sliders"]["ly"].set_val(20.0)
        app["sliders"]["lz"].set_val(20.0)
        app["sliders"]["mass"].set_val(0.1)
        app["sliders"]["density"].set_val(15.0)
        app["compute_and_draw"]()
        info_text = app["info"].get_text()
        assert "Cannot compute" in info_text or "enumerated states" in info_text
