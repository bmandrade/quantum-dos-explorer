#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a publication-quality static figure of the DOS (requires
matplotlib).

Unlike the interactive GUI, this produces a single clean figure and
saves it to a file, suitable for a report or slide. It overlays the
Gaussian-broadened numerical DOS, its analytical limiting shape, and
the Fermi level for a chosen box.

Run with::

    python examples/publication_figure.py

By default the figure is written to ``dos_figure.png`` in the current
working directory. Pass a different path as the first argument::

    python examples/publication_figure.py my_figure.png

Requires the optional GUI dependency::

    pip install "quantum-dos-explorer[gui]"
"""

import sys

import numpy as np

from quantum_dos import QuantumBox, analytical_dos, calculate_dos, classify_regime
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3


def main(output_path: str = "dos_figure.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(
            "matplotlib is required for this example. Install it with:\n"
            '    pip install "quantum-dos-explorer[gui]"'
        )

    box = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
    result = calculate_dos(
        box, sigma_eV=0.1, electron_density_m3=SILVER_ELECTRON_DENSITY_M3
    )

    # Analytical limiting shape, rescaled to the numerical peak so the two
    # can be shown on the same axes (they are different quantities; the
    # rescaling is purely for visual comparison of *shape*).
    g = analytical_dos(result.energy_eV, box)
    scale = result.dos.max() / max(g.max(), 1e-30)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(result.energy_eV, result.dos, lw=2.0, color="#1f6feb",
            label="Numerical DOS (broadened)")
    ax.plot(result.energy_eV, g * scale, ls=":", lw=1.6, color="#666",
            label="Analytical limit (rescaled)")
    ax.axvline(result.fermi_energy_eV, color="#d1242f", ls="--", lw=1.3,
               label=f"Fermi level ({result.fermi_energy_eV:.2f} eV)")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS  g(E)  (arb. units)")
    ax.set_title(
        f"DOS of a {box.lx_nm:g}x{box.ly_nm:g}x{box.lz_nm:g} nm box "
        f"({classify_regime(box)} regime)"
    )
    ax.set_xlim(0, result.energy_eV.max())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "dos_figure.png"
    main(out)
