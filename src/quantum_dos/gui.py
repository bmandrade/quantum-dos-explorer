# -*- coding: utf-8 -*-
"""
Interactive Matplotlib GUI for Quantum DOS Explorer.

This module is a **thin presentation layer**. It contains no physics
equations: every number it plots comes from the scientific API in
:mod:`quantum_dos.physics` and :mod:`quantum_dos.analysis`. Its job is
only to (1) read parameters from sliders, (2) call the API, (3) draw
the results, and (4) map the API's *semantic* classifications (e.g.
"2-D", "confined") onto colors and labels, which are presentation
choices that deliberately do not live in the scientific layer.

Launch it with::

    quantum-dos-gui

or::

    python -m quantum_dos.gui

matplotlib is required only for this module; it is an optional
dependency of the package (``pip install "quantum-dos-explorer[gui]"``).
The scientific core does not import matplotlib.
"""

from __future__ import annotations

import numpy as np

from .analysis import analytical_dos, classify_axis, classify_regime
from .constants import SILVER_ELECTRON_DENSITY_M3
from .models import ConfinementThresholds, QuantumBox
from .physics import calculate_dos, fermi_dirac

__all__ = ["main"]


# ── Default control values ────────────────────────────────────────────────
DEFAULTS = dict(
    lx_nm=5.0,
    ly_nm=5.0,
    lz_nm=5.0,
    effective_mass=1.0,
    temperature_K=0.0,
    sigma_eV=0.05,
    # Electron density is now a user-controlled slider (see module notes
    # and the project CHANGELOG). The slider ranges over a few x10^28 m^-3;
    # its default is bulk silver's value, matching the original script's
    # previously-hidden assumption, but now visible and adjustable.
    density_1e28=SILVER_ELECTRON_DENSITY_M3 / 1e28,
)

# Fixed numerical grid (kept equal to the original script's values).
E_MAX_EV = 12.0
N_POINTS = 500

# Shared confinement thresholds used for BOTH the regime badge and the
# per-axis readout coloring — one consistent convention (unlike the
# original script's three separate hardcoded threshold sets).
THRESHOLDS = ConfinementThresholds(confined_nm=2.0, bulk_nm=8.0)

# Below this positive temperature, the GUI substitutes the exact T=0
# limit. The Fermi-Dirac function supports T=0 directly (it returns the
# sharp step); this constant only guards the narrow open interval
# (0, GUI_MIN_TEMPERATURE_K), where an extremely (but not infinitely)
# sharp step would add nothing visually. Exactly 0 K is passed through
# as the true zero-temperature limit. See physics.fermi_dirac.
GUI_MIN_TEMPERATURE_K = 1.0


def _display_temperature_K(slider_value_K: float) -> float:
    """Map a temperature slider value to the temperature used for
    occupation. Exactly 0 is the true T=0 limit; values in the open
    interval (0, GUI_MIN_TEMPERATURE_K) are lifted to GUI_MIN_TEMPERATURE_K
    to avoid a needlessly sharp step; everything else passes through."""
    if slider_value_K <= 0.0:
        return 0.0
    return max(slider_value_K, GUI_MIN_TEMPERATURE_K)


# ── Presentation constants (colors live here, NOT in analysis.py) ─────────
BG = "#0F1117"
PANEL = "#1A1D27"
BLUE = "#4FC3F7"
TEXT = "#E8EAF6"
MUTED = "#7986CB"
GRID = "#252836"
RED = "#FF7043"
GREY = "#78909C"
OCC = "#1E88E5"

C_LX = "#A5D6A7"  # green — Lx slider accent
C_LY = "#FFD54F"  # amber — Ly slider accent
C_LZ = "#EF9A9A"  # rose  — Lz slider accent

# Semantic classification -> color mappings. These map the strings the
# scientific layer returns onto GUI colors. The scientific layer never
# returns colors itself; this is the single place that decides them.
REGIME_COLORS = {
    "0-D": "#CE93D8",       # purple
    "1-D": "#80CBC4",       # teal
    "2-D": "#A5D6A7",       # green
    "3-D": "#4FC3F7",       # blue
    "crossover": "#FFD54F",  # amber
}
REGIME_LABELS = {
    "0-D": "0-D  quantum dot",
    "1-D": "1-D  quantum wire",
    "2-D": "2-D  quantum well",
    "3-D": "3-D  bulk",
    "crossover": "crossover regime",
}
AXIS_COLORS = {
    "confined": "#CE93D8",   # purple
    "crossover": "#FFD54F",  # amber
    "bulk-like": "#4FC3F7",  # blue
}


def build_app(plt, Slider, Button):
    """Build the GUI figure, widgets, and update logic.

    Separated from :func:`main` so it can be exercised by a headless
    smoke test (with a non-interactive matplotlib backend) without
    entering the blocking event loop. Returns a dict of handles that a
    test — or ``main`` — can use.

    Parameters
    ----------
    plt : module
        ``matplotlib.pyplot`` (injected so this function never imports
        matplotlib at module import time).
    Slider, Button : type
        ``matplotlib.widgets.Slider`` / ``.Button``.

    Returns
    -------
    dict
        Keys: ``fig``, ``ax``, ``sliders`` (dict of the seven sliders),
        ``reset_button``, ``update`` (the update callback), ``reset``
        (the reset callback), ``artists`` (dict of persistent artists),
        and ``info`` (the state-summary text artist).
    """
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
    })

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    try:
        fig.canvas.manager.set_window_title("Quantum DOS Explorer")
    except Exception:
        pass

    # ── Divider helpers ────────────────────────────────────────────────
    def _vdiv(x):
        a = fig.add_axes([x, 0.0, 0.002, 1.0])
        a.set_facecolor(GRID)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_xticks([]); a.set_yticks([]); a.set_navigate(False)

    def _hdiv(y):
        a = fig.add_axes([0.04, y, 0.23, 0.001])
        a.set_facecolor(GRID)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_xticks([]); a.set_yticks([]); a.set_navigate(False)

    _vdiv(0.305)

    # ── Main plot ──────────────────────────────────────────────────────
    ax = fig.add_axes([0.40, 0.10, 0.57, 0.83])
    ax.set_xlabel("Energy  (eV)", fontsize=12)
    ax.set_ylabel("DOS  g(E)  (arb. units)", fontsize=12)
    ax.set_xlim(0, E_MAX_EV)
    ax.set_ylim(bottom=0)
    # Keep y-tick labels compact so wide values (large boxes) don't
    # collide with the sliders; scientific notation with a shared offset.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.yaxis.get_offset_text().set_fontsize(8)

    fig.text(0.155, 0.963, "Quantum DOS Explorer",
             ha="center", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.155, 0.940, "Free-electron model  |  Gaussian broadening",
             ha="center", fontsize=8, color=MUTED)

    egrid = np.linspace(0.01, E_MAX_EV, N_POINTS)

    # ── Persistent artists ─────────────────────────────────────────────
    line_theo, = ax.plot(egrid, np.zeros(N_POINTS), ":", color=GREY, lw=1.5,
                         zorder=4, label="Analytic limit", alpha=0.8)
    fill_holder = [ax.fill_between(egrid, 0, np.zeros(N_POINTS),
                                    color=OCC, alpha=0.35, zorder=3,
                                    label="Occupied")]
    line_dos, = ax.plot(egrid, np.zeros(N_POINTS), color=BLUE, lw=2.0,
                        zorder=5, label="Total DOS")
    vline_ef = ax.axvline(x=1.0, color=RED, lw=1.3, ls="--",
                          zorder=6, label="Fermi level", alpha=0.9)

    regime_badge = ax.text(
        0.015, 0.97, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL,
                  edgecolor=BLUE, alpha=0.85, linewidth=1.2),
    )
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.85, borderpad=0.6, handlelength=1.5)

    # ── Slider factory ─────────────────────────────────────────────────
    def _sl(rect, label, lo, hi, val, step, color=BLUE):
        a = fig.add_axes(rect)
        a.set_facecolor(BG)
        sl = Slider(a, label, lo, hi, valinit=val, valstep=step, color=color)
        sl.label.set_color(TEXT); sl.label.set_fontsize(10)
        sl.valtext.set_color(color); sl.valtext.set_fontsize(10)
        try:
            sl.poly.set_alpha(0.85)
        except AttributeError:
            pass
        return sl

    def _section(y, label):
        fig.text(0.155, y, label, ha="center", fontsize=9,
                 color=MUTED, fontstyle="italic")

    # ── Box-geometry sliders ───────────────────────────────────────────
    _section(0.910, "Box geometry")
    sl_lx = _sl([0.06, 0.868, 0.23, 0.026], "Lx (nm)", 0.5, 20.0,
                DEFAULTS["lx_nm"], 0.5, C_LX)
    sl_ly = _sl([0.06, 0.818, 0.23, 0.026], "Ly (nm)", 0.5, 20.0,
                DEFAULTS["ly_nm"], 0.5, C_LY)
    sl_lz = _sl([0.06, 0.768, 0.23, 0.026], "Lz (nm)", 0.5, 20.0,
                DEFAULTS["lz_nm"], 0.5, C_LZ)
    _hdiv(0.750)

    # ── Physics sliders ────────────────────────────────────────────────
    _section(0.738, "Physics parameters")
    sl_mass = _sl([0.06, 0.700, 0.23, 0.026], "m* (xme)", 0.1, 2.0,
                  DEFAULTS["effective_mass"], 0.05)
    sl_temp = _sl([0.06, 0.656, 0.23, 0.026], "T   (K)", 0.0, 1000.0,
                  DEFAULTS["temperature_K"], 10.0)
    sl_sigma = _sl([0.06, 0.612, 0.23, 0.026], "sig (eV)", 0.01, 0.5,
                   DEFAULTS["sigma_eV"], 0.01)
    # Electron-density slider, in units of 1e28 m^-3 (so the readout is
    # a small, legible number). Range spans typical metal densities.
    sl_density = _sl([0.06, 0.568, 0.23, 0.026], "n (1e28)", 0.5, 15.0,
                     DEFAULTS["density_1e28"], 0.1)
    _hdiv(0.550)

    # ── Reset button ───────────────────────────────────────────────────
    ax_rst = fig.add_axes([0.07, 0.500, 0.20, 0.040])
    btn_reset = Button(ax_rst, "Reset all", color=GRID, hovercolor="#2a2d3e")
    btn_reset.label.set_color(MUTED); btn_reset.label.set_fontsize(10)

    # ── Info box ───────────────────────────────────────────────────────
    ax_info = fig.add_axes([0.02, 0.048, 0.27, 0.430])
    ax_info.set_facecolor(PANEL)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    for sp in ax_info.spines.values():
        sp.set_edgecolor(GRID)
    ax_info.text(0.5, 0.975, "State summary", transform=ax_info.transAxes,
                 ha="center", va="top", fontsize=8, color=MUTED,
                 fontstyle="italic")
    info = ax_info.text(0.08, 0.90, "", transform=ax_info.transAxes,
                        va="top", ha="left", fontsize=9.5,
                        color=TEXT, linespacing=1.95)

    # ── Update callback (presentation only; all numbers from the API) ──
    def update(_=None):
        lx, ly, lz = sl_lx.val, sl_ly.val, sl_lz.val
        mass = sl_mass.val
        sigma = sl_sigma.val
        temperature = _display_temperature_K(sl_temp.val)
        density_m3 = sl_density.val * 1e28

        box = QuantumBox(lx_nm=lx, ly_nm=ly, lz_nm=lz, effective_mass=mass)

        # Call the scientific API. If the parameters are physically
        # invalid or the cutoff is too low for the requested filling,
        # show the message in the info box instead of crashing.
        try:
            result = calculate_dos(
                box, sigma_eV=sigma, electron_density_m3=density_m3,
                e_max_eV=E_MAX_EV, n_points=N_POINTS,
            )
        except ValueError as exc:
            info.set_text(f"Cannot compute:\n\n{exc}")
            fig.canvas.draw_idle()
            return

        dos = result.dos
        ef = result.fermi_energy_eV
        occupation = fermi_dirac(result.energy_eV, ef, temperature)

        # Analytical limiting shape, rescaled to the numerical peak for
        # visual comparison (the two are different quantities; see
        # analysis.analytical_dos docstring).
        g = analytical_dos(result.energy_eV, box, THRESHOLDS)
        dos_peak = max(float(dos.max()), 1e-30)
        g_peak = max(float(g.max()), 1e-30)

        line_dos.set_ydata(dos)
        line_theo.set_ydata(g / g_peak * dos_peak)
        vline_ef.set_xdata([ef, ef])

        fill_holder[0].remove()
        fill_holder[0] = ax.fill_between(result.energy_eV, 0, dos * occupation,
                                          color=OCC, alpha=0.35, zorder=3)
        ax.set_ylim(0, dos_peak * 1.2)

        # Regime badge: semantic classification -> label + color (mapping
        # owned by the GUI, not the science layer).
        regime = classify_regime(box, THRESHOLDS)
        regime_color = REGIME_COLORS[regime]
        regime_badge.set_text(REGIME_LABELS[regime])
        regime_badge.set_color(regime_color)
        regime_badge.get_bbox_patch().set_edgecolor(regime_color)

        # Per-axis confinement coloring on the dimension readouts.
        for slider, length in [(sl_lx, lx), (sl_ly, ly), (sl_lz, lz)]:
            slider.valtext.set_color(AXIS_COLORS[classify_axis(length, THRESHOLDS)])

        idx = int(np.argmin(np.abs(result.energy_eV - ef)))
        dos_at_ef = float(dos[idx])

        info.set_text(
            f"Lx : {lx:5.1f} nm  [{classify_axis(lx, THRESHOLDS)}]\n"
            f"Ly : {ly:5.1f} nm  [{classify_axis(ly, THRESHOLDS)}]\n"
            f"Lz : {lz:5.1f} nm  [{classify_axis(lz, THRESHOLDS)}]\n"
            f"\n"
            f"Regime   :  {REGIME_LABELS[regime]}\n"
            f"Ef          :  {ef:.3f} eV\n"
            f"DOS @ Ef :  {dos_at_ef:.1f}\n"
            f"States    :  {result.n_states_enumerated}\n"
            f"\n"
            f"m*    :  {mass:.2f} me\n"
            f"T      :  {temperature:.0f} K\n"
            f"sig   :  {sigma:.2f} eV\n"
            f"n      :  {density_m3:.2e} m^-3"
        )
        fig.canvas.draw_idle()

    def reset(_=None):
        for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma, sl_density):
            slider.reset()

    for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma, sl_density):
        slider.on_changed(update)
    btn_reset.on_clicked(reset)

    update()

    return {
        "fig": fig,
        "ax": ax,
        "sliders": {
            "lx": sl_lx, "ly": sl_ly, "lz": sl_lz, "mass": sl_mass,
            "temp": sl_temp, "sigma": sl_sigma, "density": sl_density,
        },
        "reset_button": btn_reset,
        "update": update,
        "reset": reset,
        "artists": {
            "line_dos": line_dos, "line_theo": line_theo,
            "vline_ef": vline_ef, "regime_badge": regime_badge,
        },
        "info": info,
    }


def main() -> None:
    """Launch the interactive DOS explorer window."""
    # Imported here (not at module top) so that merely importing this
    # module does not hard-require matplotlib; the import error is only
    # raised if someone actually tries to launch the GUI.
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    build_app(plt, Slider, Button)
    plt.show()


if __name__ == "__main__":
    main()
