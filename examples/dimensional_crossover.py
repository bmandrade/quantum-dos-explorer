#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensional crossover sweep (no GUI).

Reproduces qualitatively the central idea of Batabyal & Dev,
Physica E 64 (2014) 224-233: as a confined box grows along one or more
directions, its DOS evolves from discrete 0-D-like peaks toward the
smooth higher-dimensional limiting shapes.

Here we grow a thin 2-atom-thick slab (Lz = 0.5 nm) laterally, mirroring
their Fig. 1 (0-D -> 2-D evolution), and report how the Fermi energy and
the dimensional-regime classification change with lateral size.

Run with::

    python examples/dimensional_crossover.py

Uses only the scientific core (no matplotlib).
"""

from quantum_dos import QuantumBox, calculate_dos, classify_regime
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3


def main() -> None:
    lz_nm = 0.5  # fixed 2-atom-thick slab, as in the reference paper
    lateral_sizes_nm = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    print(f"Thin slab, Lz = {lz_nm} nm, grown laterally (Lx = Ly = L):\n")
    header = f"{'L (nm)':>8} | {'regime':>10} | {'Ef (eV)':>9} | {'states':>8}"
    print(header)
    print("-" * len(header))

    for L in lateral_sizes_nm:
        box = QuantumBox(lx_nm=L, ly_nm=L, lz_nm=lz_nm, effective_mass=1.0)
        result = calculate_dos(
            box, sigma_eV=0.1, electron_density_m3=SILVER_ELECTRON_DENSITY_M3
        )
        print(
            f"{L:8.1f} | {classify_regime(box):>10} | "
            f"{result.fermi_energy_eV:9.3f} | {result.n_states_enumerated:8d}"
        )

    print(
        "\nAs L increases, the two lateral directions cross from 'confined' "
        "through 'crossover' to 'bulk-like', and the regime label evolves "
        "accordingly. (Threshold values are configurable via "
        "ConfinementThresholds.)"
    )


if __name__ == "__main__":
    main()
