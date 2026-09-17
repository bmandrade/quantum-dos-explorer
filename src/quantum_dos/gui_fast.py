# -*- coding: utf-8 -*-
"""
Performance-optimized interactive GUI for Quantum DOS Explorer.

This is a drop-in alternative to :mod:`quantum_dos.gui` with identical
appearance, controls, and -- crucially -- identical numerical results,
but much smoother interaction. The original ``gui`` module is preserved
unchanged; this module adds three presentation-layer optimizations, none
of which alter the physics:

1. **Result caching.** The expensive DOS curve is recomputed only when
   an input that actually changes it (geometry, effective mass, sigma)
   changes. Moving the temperature slider triggers *no* recompute (it
   only re-shades the occupied region); moving the density slider reuses
   the cached DOS curve and only re-derives the cheap Fermi energy.
2. **Debounced updates.** Rapid slider drags are coalesced so one drag
   triggers one recompute instead of dozens.
3. **Blitting.** Only the changed artists are redrawn, not the whole
   figure.

Launch it with::

    quantum-dos-gui-fast

or::

    python -m quantum_dos.gui_fast

Like :mod:`quantum_dos.gui`, matplotlib is imported lazily so the
scientific core stays matplotlib-free. Requires the ``[gui]`` extra.

Why a separate module? So the naive reference GUI (``gui.py``) remains
available and unchanged for comparison, and so the optimization can be
demonstrated side by side. The physics all comes from the same
scientific API; this file, like ``gui.py``, contains no physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .analysis import analytical_dos, classify_axis, classify_regime
from .constants import SILVER_ELECTRON_DENSITY_M3
from .models import ConfinementThresholds, DOSResult, QuantumBox
from .physics import calculate_dos, fermi_dirac, fermi_energy

# Reuse the presentation constants and defaults from the reference GUI so
# the two windows look identical and there is a single source of truth
# for colors, labels, thresholds, and default values.
from .gui import (
    AXIS_COLORS,
    BG,
    BLUE,
    C_LX,
    C_LY,
    C_LZ,
    DEFAULTS,
    E_MAX_EV,
    GRID,
    GUI_MIN_TEMPERATURE_K,
    MUTED,
    N_POINTS,
    OCC,
    PANEL,
    RED,
    REGIME_COLORS,
    REGIME_LABELS,
    TEXT,
    THRESHOLDS,
    GREY,
)

__all__ = ["main", "build_app", "DosCache"]

# Debounce interval: slider events arriving within this window are
# coalesced into a single recompute. 60 ms is below the threshold of
# perceptible lag but comfortably absorbs a fast drag's event storm.
DEBOUNCE_SECONDS = 0.06


@dataclass
class _CurveKey:
    """The set of inputs that determine the DOS *curve* (not Ef or
    occupation). If this is unchanged, the cached curve can be reused."""

    lx_nm: float
    ly_nm: float
    lz_nm: float
    effective_mass: float
    sigma_eV: float


class DosCache:
    """Caches the last DOS computation and decides the cheapest update.

    The three tiers, cheapest first:

    - **temperature only changed** -> no physics recompute at all; the
      DOS curve and Fermi energy are reused, only occupation is
      re-evaluated by the caller (microseconds).
    - **density changed** (curve inputs unchanged) -> reuse the cached
      DOS curve; recompute only the Fermi energy from the cached,
      already-sorted state energies (also microseconds).
    - **a curve input changed** (geometry / mass / sigma) -> full
      :func:`calculate_dos`.

    This class holds no matplotlib state; it is pure caching logic and
    is unit-tested directly (see tests/test_gui_fast.py).
    """

    def __init__(self) -> None:
        self._key: Optional[_CurveKey] = None
        self._result: Optional[DOSResult] = None
        self._sorted_energies: Optional[np.ndarray] = None
        self._density_m3: Optional[float] = None

    def get(
        self, box: QuantumBox, sigma_eV: float, density_m3: float
    ) -> DOSResult:
        """Return a DOSResult for these inputs, recomputing as little as
        possible. The returned object is always numerically identical to
        calling ``calculate_dos(box, sigma_eV, density_m3)`` directly."""
        key = _CurveKey(box.lx_nm, box.ly_nm, box.lz_nm,
                        box.effective_mass, sigma_eV)

        if self._key == key and self._result is not None:
            # Curve inputs unchanged. Does density differ?
            if density_m3 == self._density_m3:
                return self._result  # nothing changed at all
            # Only density changed: reuse the DOS curve, re-fill Ef.
            n_electrons = density_m3 * box.volume_m3
            ef = fermi_energy(self._sorted_energies, n_electrons)
            self._result = DOSResult(
                energy_eV=self._result.energy_eV,
                dos=self._result.dos,
                fermi_energy_eV=ef,
                n_states_enumerated=self._result.n_states_enumerated,
            )
            self._density_m3 = density_m3
            return self._result

        # A curve input changed: full recompute.
        result = calculate_dos(
            box, sigma_eV=sigma_eV, electron_density_m3=density_m3,
            e_max_eV=E_MAX_EV, n_points=N_POINTS,
        )
        # Cache the sorted energies too, so a later density-only change can
        # re-fill Ef without re-enumerating. Recomputing them here is cheap
        # relative to the broadening that calculate_dos already did.
        from .physics import energy_levels
        self._sorted_energies = energy_levels(box, e_max_eV=E_MAX_EV,
                                              sigma_eV=sigma_eV)
        self._key = key
        self._result = result
        self._density_m3 = density_m3
        return result


def build_app(plt, Slider, Button):
    """Build the optimized GUI figure, widgets, and update logic.

    Mirrors :func:`quantum_dos.gui.build_app` but adds the caching,
    debounce, and blitting machinery. Returns a dict of handles for
    headless testing (see tests/test_gui_fast.py). Injecting ``plt`` /
    widgets keeps this module import free of matplotlib.
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
        fig.canvas.manager.set_window_title("Quantum DOS Explorer (fast)")
    except Exception:
        pass

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

    ax = fig.add_axes([0.36, 0.10, 0.61, 0.83])
    ax.set_xlabel("Energy  (eV)", fontsize=12)
    ax.set_ylabel("DOS  g(E)  (arb. units)", fontsize=12)
    ax.set_xlim(0, E_MAX_EV)
    ax.set_ylim(bottom=0)

    fig.text(0.155, 0.963, "Quantum DOS Explorer",
             ha="center", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.155, 0.940, "Free-electron model  |  optimized",
             ha="center", fontsize=8, color=MUTED)

    egrid = np.linspace(0.01, E_MAX_EV, N_POINTS)

    line_theo, = ax.plot(egrid, np.zeros(N_POINTS), ":", color=GREY, lw=1.5,
                         zorder=4, label="Analytic limit", alpha=0.8,
                         animated=True)
    fill_holder = [ax.fill_between(egrid, 0, np.zeros(N_POINTS),
                                    color=OCC, alpha=0.35, zorder=3,
                                    label="Occupied", animated=True)]
    line_dos, = ax.plot(egrid, np.zeros(N_POINTS), color=BLUE, lw=2.0,
                        zorder=5, label="Total DOS", animated=True)
    vline_ef = ax.axvline(x=1.0, color=RED, lw=1.3, ls="--",
                          zorder=6, label="Fermi level", alpha=0.9,
                          animated=True)

    regime_badge = ax.text(
        0.015, 0.97, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL,
                  edgecolor=BLUE, alpha=0.85, linewidth=1.2),
        animated=True,
    )
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.85, borderpad=0.6, handlelength=1.5)

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

    _section(0.910, "Box geometry")
    sl_lx = _sl([0.06, 0.868, 0.23, 0.026], "Lx (nm)", 0.5, 20.0,
                DEFAULTS["lx_nm"], 0.5, C_LX)
    sl_ly = _sl([0.06, 0.818, 0.23, 0.026], "Ly (nm)", 0.5, 20.0,
                DEFAULTS["ly_nm"], 0.5, C_LY)
    sl_lz = _sl([0.06, 0.768, 0.23, 0.026], "Lz (nm)", 0.5, 20.0,
                DEFAULTS["lz_nm"], 0.5, C_LZ)
    _hdiv(0.750)

    _section(0.738, "Physics parameters")
    sl_mass = _sl([0.06, 0.700, 0.23, 0.026], "m* (xme)", 0.1, 2.0,
                  DEFAULTS["effective_mass"], 0.05)
    sl_temp = _sl([0.06, 0.656, 0.23, 0.026], "T   (K)", 0.0, 1000.0,
                  DEFAULTS["temperature_K"], 10.0)
    sl_sigma = _sl([0.06, 0.612, 0.23, 0.026], "sig (eV)", 0.01, 0.5,
                   DEFAULTS["sigma_eV"], 0.01)
    sl_density = _sl([0.06, 0.568, 0.23, 0.026], "n (1e28)", 0.5, 15.0,
                     DEFAULTS["density_1e28"], 0.1)
    _hdiv(0.550)

    ax_rst = fig.add_axes([0.07, 0.500, 0.20, 0.040])
    btn_reset = Button(ax_rst, "Reset all", color=GRID, hovercolor="#2a2d3e")
    btn_reset.label.set_color(MUTED); btn_reset.label.set_fontsize(10)

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

    cache = DosCache()
    # Background captured for blitting; refreshed on full draws / resizes.
    # 'pending' tracks whether a debounced recompute is queued.
    state = {"background": None, "timer": None, "pending": False}

    def _compute_and_draw():
        lx, ly, lz = sl_lx.val, sl_ly.val, sl_lz.val
        mass = sl_mass.val
        sigma = sl_sigma.val
        temperature = max(sl_temp.val, GUI_MIN_TEMPERATURE_K)
        density_m3 = sl_density.val * 1e28

        box = QuantumBox(lx_nm=lx, ly_nm=ly, lz_nm=lz, effective_mass=mass)

        try:
            result = cache.get(box, sigma_eV=sigma, density_m3=density_m3)
        except ValueError as exc:
            info.set_text(f"Cannot compute:\n\n{exc}")
            fig.canvas.draw_idle()
            return

        dos = result.dos
        ef = result.fermi_energy_eV
        occupation = fermi_dirac(result.energy_eV, ef, temperature)

        g = analytical_dos(result.energy_eV, box, THRESHOLDS)
        dos_peak = max(float(dos.max()), 1e-30)
        g_peak = max(float(g.max()), 1e-30)

        line_dos.set_ydata(dos)
        line_theo.set_ydata(g / g_peak * dos_peak)
        vline_ef.set_xdata([ef, ef])

        fill_holder[0].remove()
        fill_holder[0] = ax.fill_between(result.energy_eV, 0, dos * occupation,
                                          color=OCC, alpha=0.35, zorder=3,
                                          animated=True)
        ax.set_ylim(0, dos_peak * 1.2)

        regime = classify_regime(box, THRESHOLDS)
        regime_color = REGIME_COLORS[regime]
        regime_badge.set_text(REGIME_LABELS[regime])
        regime_badge.set_color(regime_color)
        regime_badge.get_bbox_patch().set_edgecolor(regime_color)

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

        # Because set_ylim can change the axes, a full draw is needed to
        # refresh the blit background; then subsequent blits are cheap.
        _full_draw()

    def _full_draw():
        """Redraw the whole canvas and re-capture the blit background."""
        fig.canvas.draw()
        try:
            state["background"] = fig.canvas.copy_from_bbox(ax.bbox)
        except Exception:
            state["background"] = None

    def _blit_artists():
        """Redraw only the animated artists over the cached background."""
        bg = state["background"]
        if bg is None:
            fig.canvas.draw_idle()
            return
        fig.canvas.restore_region(bg)
        for artist in (fill_holder[0], line_theo, line_dos, vline_ef, regime_badge):
            ax.draw_artist(artist)
        fig.canvas.blit(ax.bbox)

    def _schedule(_=None):
        """Debounce slider events: restart a single reusable one-shot
        timer on every event, so only the last event in a rapid burst
        survives to trigger one ``_compute_and_draw``.

        Rationale: while dragging a slider, matplotlib emits a stream of
        ``on_changed`` events (often dozens per second). Without
        debouncing, each would launch a full recompute, so the UI would
        fall behind the cursor. By resetting the timer on each event and
        computing only once it has been quiet for ``DEBOUNCE_SECONDS``,
        a whole drag collapses to a single recompute at its end.
        """
        timer = state["timer"]
        if timer is None:
            # Create the reusable timer once, on first use.
            timer = fig.canvas.new_timer(interval=int(DEBOUNCE_SECONDS * 1000))
            timer.single_shot = True
            timer.add_callback(_on_debounce_fire)
            state["timer"] = timer
        else:
            timer.stop()
        state["pending"] = True
        timer.start()

    def _on_debounce_fire():
        """Called by the debounce timer once a drag has gone quiet."""
        state["pending"] = False
        _compute_and_draw()

    def reset(_=None):
        for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma, sl_density):
            slider.reset()

    for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma, sl_density):
        slider.on_changed(_schedule)
    btn_reset.on_clicked(reset)

    # Initial synchronous draw (no debounce) so the window is populated
    # immediately on launch.
    _compute_and_draw()

    return {
        "fig": fig,
        "ax": ax,
        "sliders": {
            "lx": sl_lx, "ly": sl_ly, "lz": sl_lz, "mass": sl_mass,
            "temp": sl_temp, "sigma": sl_sigma, "density": sl_density,
        },
        "reset_button": btn_reset,
        "compute_and_draw": _compute_and_draw,
        "schedule": _schedule,
        "on_debounce_fire": _on_debounce_fire,
        "state": state,
        "reset": reset,
        "cache": cache,
        "artists": {
            "line_dos": line_dos, "line_theo": line_theo,
            "vline_ef": vline_ef, "regime_badge": regime_badge,
        },
        "info": info,
    }


def main() -> None:
    """Launch the optimized interactive DOS explorer window."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    build_app(plt, Slider, Button)
    plt.show()


if __name__ == "__main__":
    main()
