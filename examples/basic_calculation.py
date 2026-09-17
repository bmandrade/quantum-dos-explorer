#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic usage of the Quantum DOS Explorer scientific API (no GUI).

Run with::

    python examples/basic_calculation.py

This computes the density of states and Fermi energy of a 5 x 5 x 5 nm
box (free-electron mass) filled at bulk silver's electron density, then
prints a short summary. It uses only the pure-NumPy scientific core --
matplotlib is not required.
"""

from quantum_dos import QuantumBox, calculate_dos, classify_regime
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3


def main() -> None:
    box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)

    result = calculate_dos(
        box,
        sigma_eV=0.05,
        electron_density_m3=SILVER_ELECTRON_DENSITY_M3,
    )

    print(f"Box volume         : {box.volume_nm3:g} nm^3")
    print(f"Dimensional regime : {classify_regime(box)}")
    print(f"States enumerated  : {result.n_states_enumerated}")
    print(f"Fermi energy       : {result.fermi_energy_eV:.4f} eV")
    print(f"Energy grid points : {result.energy_eV.size}")
    print(f"Peak DOS (arb.)    : {result.dos.max():.2f}")


if __name__ == "__main__":
    main()
