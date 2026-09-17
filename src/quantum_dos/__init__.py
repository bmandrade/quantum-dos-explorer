# -*- coding: utf-8 -*-
"""
Quantum DOS Explorer — explore how quantum confinement changes the
electronic density of states of a rectangular quantum box.

Public scientific API (no matplotlib dependency)::

    from quantum_dos import QuantumBox, calculate_dos

    box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
    result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
    print(result.fermi_energy_eV)
"""

from .analysis import analytical_dos, classify_axis, classify_regime
from .models import ConfinementThresholds, DOSResult, QuantumBox
from .physics import broadened_dos, calculate_dos, energy_levels, fermi_dirac, fermi_energy

__all__ = [
    "QuantumBox",
    "ConfinementThresholds",
    "DOSResult",
    "energy_levels",
    "broadened_dos",
    "fermi_energy",
    "fermi_dirac",
    "calculate_dos",
    "analytical_dos",
    "classify_axis",
    "classify_regime",
]

__version__ = "0.1.0"
