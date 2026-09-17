# -*- coding: utf-8 -*-
"""
Data structures for Quantum DOS Explorer.

These dataclasses carry explicit units in their field names so that
the scientific API is unambiguous, and validate their inputs on
construction so that invalid physical parameters fail fast with a
clear error instead of silently propagating NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QuantumBox:
    """
    Geometry and effective mass of a rectangular quantum box.

    Parameters
    ----------
    lx_nm, ly_nm, lz_nm : float
        Box side lengths along x, y, z, in nanometres. Must be > 0.
    effective_mass : float
        Electron effective mass, in units of the free-electron mass
        ``m_e`` (i.e. ``effective_mass=1.0`` means the free-electron
        mass). Must be > 0.

    Notes
    -----
    The box uses Dirichlet boundary conditions (wavefunction = 0 on
    the box surface), consistent with the original script and with
    Batabyal & Dev, Physica E 64 (2014) 224-233, who show this is the
    appropriate choice for small/finite systems (periodic boundary
    conditions are only appropriate in the bulk limit).
    """

    lx_nm: float
    ly_nm: float
    lz_nm: float
    effective_mass: float

    def __post_init__(self) -> None:
        for name in ("lx_nm", "ly_nm", "lz_nm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"QuantumBox.{name} must be a finite, positive "
                    f"number (nm); got {value!r}."
                )
        if not np.isfinite(self.effective_mass) or self.effective_mass <= 0:
            raise ValueError(
                "QuantumBox.effective_mass must be a finite, positive "
                f"number (in units of m_e); got {self.effective_mass!r}."
            )

    @property
    def dimensions_nm(self) -> tuple[float, float, float]:
        """Return (lx_nm, ly_nm, lz_nm) as a tuple."""
        return (self.lx_nm, self.ly_nm, self.lz_nm)

    @property
    def volume_nm3(self) -> float:
        """Box volume in nm^3."""
        return self.lx_nm * self.ly_nm * self.lz_nm

    @property
    def volume_m3(self) -> float:
        """Box volume in m^3."""
        return self.volume_nm3 * 1e-27


@dataclass(frozen=True)
class ConfinementThresholds:
    """
    Length thresholds used to classify a box axis (or a whole box) as
    quantum-confined, in a dimensional crossover regime, or bulk-like.

    Parameters
    ----------
    confined_nm : float
        An axis shorter than this is considered "confined"
        (quantum-dot-like along that direction). Default 2.0 nm.
    bulk_nm : float
        An axis longer than this is considered "bulk-like"
        (behaves as an extended/continuum direction). Default 8.0 nm.

    Notes
    -----
    These values are **not** universal physical constants. They are a
    practical, configurable convention (following the qualitative
    discussion in Batabyal & Dev, Physica E 64 (2014) 224-233, who
    show the confined -> extended crossover length depends on the
    material, effective mass and energy/broadening resolution used).
    Change them explicitly if your system requires different cutoffs.

    This single threshold set is shared by every part of the package
    that needs to decide "is this axis/box confined, crossover, or
    bulk-like" (`analysis.analytical_dos`, `analysis.classify_regime`,
    `analysis.classify_axis`). The original script used three
    inconsistent, hardcoded threshold sets (2.0 nm in one function,
    5.0 nm in another, 2.0/8.0 nm in a third); unifying them is a
    deliberate, documented behavior change (see CHANGELOG.md) that
    affects boxes with axes between 2-5 nm and 5-8 nm.
    """

    confined_nm: float = 2.0
    bulk_nm: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.confined_nm) or self.confined_nm <= 0:
            raise ValueError(
                "ConfinementThresholds.confined_nm must be finite and "
                f"positive; got {self.confined_nm!r}."
            )
        if not np.isfinite(self.bulk_nm) or self.bulk_nm <= 0:
            raise ValueError(
                "ConfinementThresholds.bulk_nm must be finite and "
                f"positive; got {self.bulk_nm!r}."
            )
        if self.confined_nm >= self.bulk_nm:
            raise ValueError(
                "ConfinementThresholds requires confined_nm < bulk_nm; "
                f"got confined_nm={self.confined_nm!r}, "
                f"bulk_nm={self.bulk_nm!r}."
            )


@dataclass(frozen=True)
class DOSResult:
    """
    Result of a density-of-states calculation.

    Attributes
    ----------
    energy_eV : np.ndarray, shape (n_points,)
        Energy grid, in eV.
    dos : np.ndarray, shape (n_points,)
        Gaussian-broadened numerical DOS evaluated on ``energy_eV``.
        This is an un-normalized sum of Gaussians (one per enumerated
        discrete state), i.e. it is in arbitrary units, not
        states/eV/volume. See physics.calculate_dos for details.
    fermi_energy_eV : float
        Estimated Fermi energy, in eV.
    n_states_enumerated : int
        Number of discrete quantum-box states enumerated below the
        energy cutoff used for this calculation. Useful for judging
        whether the cutoff / grid was adequate.
    """

    energy_eV: np.ndarray
    dos: np.ndarray
    fermi_energy_eV: float
    n_states_enumerated: int
