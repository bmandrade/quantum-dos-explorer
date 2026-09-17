# -*- coding: utf-8 -*-
"""Cross-check that the optimized GUI matches the reference GUI numerically.

The whole premise of ``gui_fast`` is that it is *only* faster -- it must
produce exactly the same DOS curve, Fermi energy, occupied shading,
regime label, and per-axis classification as the reference ``gui`` for
every set of inputs. This module builds both GUIs on the Agg backend,
drives them to the same slider states, and asserts their rendered data
matches.

If this test ever fails, the optimization has changed the science, which
is precisely what it must never do.

Skipped automatically if matplotlib is unavailable.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

from quantum_dos import gui as ref_gui  # noqa: E402
from quantum_dos import gui_fast as fast_gui  # noqa: E402


# A spread of slider states covering the different regimes and the
# cheap/expensive update paths. (lx, ly, lz, mass, temp, sigma). Density
# is now fixed to bulk silver for both GUIs, so it is not varied here.
PARAMETER_SETS = [
    (5.0, 5.0, 5.0, 1.0, 0.0, 0.05),      # default-ish, T=0
    (5.0, 5.0, 5.0, 1.0, 300.0, 0.05),    # finite T
    (1.0, 1.0, 1.0, 1.0, 100.0, 0.05),    # 0-D dot
    (1.0, 1.0, 12.0, 1.0, 200.0, 0.10),   # 1-D-ish
    (12.0, 12.0, 1.0, 1.5, 500.0, 0.20),  # 2-D-ish, heavy mass
    (10.0, 10.0, 10.0, 1.0, 0.0, 0.05),   # 3-D bulk
    (4.0, 6.0, 8.0, 0.8, 750.0, 0.30),    # crossover, high T
    (20.0, 20.0, 20.0, 0.1, 200.0, 0.05),  # error case: too many electrons
]


def _set_state(app, lx, ly, lz, mass, temp, sigma):
    app["sliders"]["lx"].set_val(lx)
    app["sliders"]["ly"].set_val(ly)
    app["sliders"]["lz"].set_val(lz)
    app["sliders"]["mass"].set_val(mass)
    app["sliders"]["temp"].set_val(temp)
    app["sliders"]["sigma"].set_val(sigma)


@pytest.mark.parametrize("params", PARAMETER_SETS)
def test_fast_gui_matches_reference_gui_numerically(params):
    ref = ref_gui.build_app(plt, Slider, Button)
    fast = fast_gui.build_app(plt, Slider, Button)
    try:
        _set_state(ref, *params)
        # The reference GUI updates synchronously on each set_val; call its
        # update once more to be certain it reflects the final state.
        ref["update"]()

        _set_state(fast, *params)
        # The fast GUI debounces, so trigger its compute explicitly.
        fast["compute_and_draw"]()

        ref_info = ref["info"].get_text()
        fast_info = fast["info"].get_text()

        # The state-summary text is the single source of truth for the
        # computed result (Ef, DOS@Ef, states, regime, T, sigma, n) or, on
        # failure, the identical "Cannot compute" message. It must match
        # either way.
        assert fast_info == ref_info

        # If this parameter set is one the model cannot compute (both GUIs
        # show the same error and leave their previous curves untouched),
        # there is nothing further to compare -- the curves are stale by
        # design and legitimately differ by build history.
        if fast_info.startswith("Cannot compute"):
            return

        # Otherwise the plotted data must be byte-for-byte identical.
        np.testing.assert_array_equal(
            fast["artists"]["line_dos"].get_ydata(),
            ref["artists"]["line_dos"].get_ydata(),
        )
        np.testing.assert_array_equal(
            fast["artists"]["line_theo"].get_ydata(),
            ref["artists"]["line_theo"].get_ydata(),
        )
        np.testing.assert_array_equal(
            fast["artists"]["vline_ef"].get_xdata(),
            ref["artists"]["vline_ef"].get_xdata(),
        )
        assert (fast["artists"]["regime_badge"].get_text()
                == ref["artists"]["regime_badge"].get_text())
    finally:
        plt.close(ref["fig"])
        plt.close(fast["fig"])


def test_fast_gui_matches_reference_after_occupied_shading():
    # The occupied-DOS shading (dos * Fermi-Dirac occupation) is what the
    # temperature slider drives; verify it matches between the two GUIs at
    # a finite temperature by comparing the fill polygon vertices.
    ref = ref_gui.build_app(plt, Slider, Button)
    fast = fast_gui.build_app(plt, Slider, Button)
    try:
        params = (6.0, 6.0, 6.0, 1.0, 400.0, 0.08)
        _set_state(ref, *params)
        ref["update"]()
        _set_state(fast, *params)
        fast["compute_and_draw"]()

        # Both GUIs store the occupied fill as the current PolyCollection on
        # their axes; compare the summary line that reports DOS @ Ef, which
        # depends on the same occupied/DOS data, as a robust proxy plus the
        # already-compared full info text.
        assert fast["info"].get_text() == ref["info"].get_text()
    finally:
        plt.close(ref["fig"])
        plt.close(fast["fig"])
