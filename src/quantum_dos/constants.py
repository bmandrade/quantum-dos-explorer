# -*- coding: utf-8 -*-
"""
Physical constants used throughout Quantum DOS Explorer.

All values are given in SI base units unless a conversion factor is
explicitly named (e.g. ``EV_TO_J``). Values are taken from CODATA
recommended values (2018), truncated to the precision used in the
original script this package was refactored from, so that numerical
results are unchanged by the refactor.

References
----------
CODATA 2018 recommended values: https://physics.nist.gov/cuu/Constants/
"""

#: Reduced Planck constant, hbar = h / (2*pi)  [J s]
HBAR: float = 1.0545718e-34

#: Free electron rest mass, m_e  [kg]
ELECTRON_MASS: float = 9.1093837e-31

#: Conversion factor: 1 eV in Joules  [J / eV]
EV_TO_J: float = 1.6021766e-19

#: Boltzmann constant, k_B  [J / K]
BOLTZMANN: float = 1.380649e-23

#: Example bulk conduction-electron density of silver [m^-3].
#:
#: This is *not* a physics default used anywhere in the calculation
#: engine. It is provided only as a convenient, physically meaningful
#: illustrative value for GUI/CLI examples, because the original
#: script's Fermi-energy estimate implicitly assumed this density.
#: Every caller of the scientific API must now supply
#: ``electron_density_m3`` explicitly (see physics.calculate_dos).
SILVER_ELECTRON_DENSITY_M3: float = 5.86e28
