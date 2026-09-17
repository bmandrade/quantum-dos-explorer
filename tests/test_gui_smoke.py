# -*- coding: utf-8 -*-
"""Headless smoke test for the interactive GUI.

The assignment asks for a manual smoke test of the GUI (does it start,
do sliders update the plot, does reset work, no exceptions during
normal interaction). This module provides the *automatable* subset of
that: it builds the real GUI on a non-interactive ("Agg") matplotlib
backend and drives the actual slider/reset callbacks, checking that
the plotted artists respond as expected -- without opening a window or
requiring a display.

It deliberately does NOT try to assert pixel-level rendering details
(the assignment explicitly warns against brittle visual tests); it
checks behavior, not appearance.

The whole module is skipped if matplotlib is not installed, so the
core test suite still runs in a matplotlib-free environment.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

from quantum_dos.gui import build_app  # noqa: E402


@pytest.fixture
def app():
    """Build the GUI on the Agg backend and clean it up afterwards."""
    handles = build_app(plt, Slider, Button)
    yield handles
    plt.close(handles["fig"])


class TestGuiBuilds:
    def test_application_starts_without_exception(self, app):
        # If build_app returned, the figure was constructed and the
        # initial update() ran without raising.
        assert app["fig"] is not None
        assert set(app["sliders"]) == {
            "lx", "ly", "lz", "mass", "temp", "sigma", "density"
        }

    def test_initial_dos_curve_is_populated(self, app):
        ydata = app["artists"]["line_dos"].get_ydata()
        assert np.any(ydata > 0)
        assert np.all(np.isfinite(ydata))


class TestSlidersUpdatePlot:
    def test_changing_a_dimension_updates_the_dos_curve(self, app):
        before = app["artists"]["line_dos"].get_ydata().copy()
        app["sliders"]["lx"].set_val(12.0)  # fires the on_changed callback
        after = app["artists"]["line_dos"].get_ydata()
        assert not np.array_equal(before, after)

    def test_changing_effective_mass_updates_the_dos_curve(self, app):
        before = app["artists"]["line_dos"].get_ydata().copy()
        app["sliders"]["mass"].set_val(2.0)
        after = app["artists"]["line_dos"].get_ydata()
        assert not np.array_equal(before, after)

    def test_changing_sigma_changes_the_broadening(self, app):
        before = app["artists"]["line_dos"].get_ydata().copy()
        app["sliders"]["sigma"].set_val(0.4)
        after = app["artists"]["line_dos"].get_ydata()
        assert not np.array_equal(before, after)

    def test_changing_density_moves_the_fermi_level(self, app):
        before = app["artists"]["vline_ef"].get_xdata()[0]
        app["sliders"]["density"].set_val(12.0)
        after = app["artists"]["vline_ef"].get_xdata()[0]
        assert not np.isclose(before, after)

    def test_changing_temperature_does_not_raise_and_keeps_curve_finite(self, app):
        app["sliders"]["temp"].set_val(900.0)
        ydata = app["artists"]["line_dos"].get_ydata()
        assert np.all(np.isfinite(ydata))


class TestReset:
    def test_reset_restores_default_slider_values(self, app):
        app["sliders"]["lx"].set_val(15.0)
        app["sliders"]["mass"].set_val(1.8)
        app["reset"]()
        assert np.isclose(app["sliders"]["lx"].val, 5.0)
        assert np.isclose(app["sliders"]["mass"].val, 1.0)


class TestInvalidParametersHandledGracefully:
    def test_density_too_high_for_cutoff_shows_message_not_crash(self, app):
        # Drive the sliders into the regime that makes calculate_dos raise
        # (large light-mass box at high density). The GUI must catch it and
        # display a message rather than propagating the exception.
        app["sliders"]["lx"].set_val(20.0)
        app["sliders"]["ly"].set_val(20.0)
        app["sliders"]["lz"].set_val(20.0)
        app["sliders"]["mass"].set_val(0.1)
        app["sliders"]["density"].set_val(15.0)
        # Should not have raised. Info text should mention the problem.
        info_text = app["info"].get_text()
        assert "Cannot compute" in info_text or "enumerated states" in info_text
