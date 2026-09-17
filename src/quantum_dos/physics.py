# -*- coding: utf-8 -*-
"""
Scientific core of Quantum DOS Explorer: quantum-box energy levels,
Gaussian-broadened density of states, Fermi energy and Fermi-Dirac
occupation.

This module has no dependency on matplotlib or any other plotting
library, so it can be imported and tested (or used from a script or
notebook) independently of the GUI::

    from quantum_dos.physics import calculate_dos
    from quantum_dos.models import QuantumBox

    box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
    result = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
    print(result.fermi_energy_eV)

Physical model
--------------
Electrons are treated as free (parabolic-band) particles confined to
a rectangular box of sides Lx, Ly, Lz with Dirichlet (infinite-well)
boundary conditions. The energy eigenvalues are

    E(nx, ny, nz) = (hbar^2 * pi^2) / (2 * m*) *
                    (nx^2/Lx^2 + ny^2/Ly^2 + nz^2/Lz^2)

with nx, ny, nz = 1, 2, 3, ... This matches Eq. (2) of
Batabyal & Dev, Physica E 64 (2014) 224-233, which is the reference
model this software implements.

The numerical DOS is built by explicitly enumerating discrete states
below an energy cutoff and replacing each discrete level with a
Gaussian of standard deviation ``sigma_eV`` (Eq. (9) of the same
reference, in the sigma -> 0+ *finite* limit, i.e. sigma is a finite
broadening parameter representing e.g. experimental energy
resolution, not a mathematical limit taken to zero).
"""

from __future__ import annotations

import numpy as np

from .constants import BOLTZMANN, ELECTRON_MASS, EV_TO_J, HBAR
from .models import DOSResult, QuantumBox

__all__ = [
    "energy_levels",
    "broadened_dos",
    "fermi_energy",
    "fermi_dirac",
    "calculate_dos",
]


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number; got {value!r}.")


def energy_levels(box: QuantumBox, e_max_eV: float, sigma_eV: float) -> np.ndarray:
    """
    Enumerate discrete quantum-box energy levels up to a cutoff.

    Parameters
    ----------
    box : QuantumBox
        Box geometry and effective mass (effective_mass in units of m_e).
    e_max_eV : float
        Nominal upper energy bound of interest, in eV. States are
        enumerated up to ``e_max_eV + 4 * sigma_eV`` so that Gaussian
        tails from just-above-cutoff states are still captured when
        later broadening onto an energy grid that goes up to
        ``e_max_eV``. Must be > 0.
    sigma_eV : float
        Intended Gaussian broadening width, in eV. Used only to set
        the enumeration margin above ``e_max_eV`` (see above). Must
        be > 0.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of energy eigenvalues (eV), one entry per
        enumerated (nx, ny, nz) combination with energy at or below
        ``e_max_eV + 4 * sigma_eV``. May be empty if no states fall
        in range (this can only happen for pathological inputs).

    Notes
    -----
    Degenerate states are *not* merged: each (nx, ny, nz) combination
    contributes its own entry, matching the original script's
    behaviour and the Gaussian-sum construction used downstream.

    The number of states enumerated grows roughly as the product of
    the three per-axis state counts, i.e. it can grow quickly for
    large boxes and/or small effective mass. See README "Numerical
    method" / "Limitations" for a discussion of this scaling.
    """
    _validate_positive("e_max_eV", e_max_eV)
    _validate_positive("sigma_eV", sigma_eV)

    mass_kg = box.effective_mass * ELECTRON_MASS
    l_m = np.array(box.dimensions_nm, dtype=float) * 1e-9

    # Per-axis energy coefficient A_i such that E = sum_i A_i * n_i^2 [eV]
    coeff_eV = (np.pi**2 * HBAR**2) / (2.0 * mass_kg * l_m**2) / EV_TO_J

    cutoff_eV = e_max_eV + 4.0 * sigma_eV

    n_max = np.ceil(np.sqrt(np.maximum(cutoff_eV / coeff_eV, 0.0))).astype(int) + 2

    nx = np.arange(1, n_max[0] + 1)
    ny = np.arange(1, n_max[1] + 1)
    nz = np.arange(1, n_max[2] + 1)
    grid_x, grid_y, grid_z = np.meshgrid(nx, ny, nz, indexing="ij")

    energies_eV = (
        coeff_eV[0] * grid_x**2 + coeff_eV[1] * grid_y**2 + coeff_eV[2] * grid_z**2
    ).ravel()
    del grid_x, grid_y, grid_z

    energies_eV = energies_eV[energies_eV <= cutoff_eV]
    return np.sort(energies_eV)


def broadened_dos(
    energies_eV: np.ndarray,
    egrid_eV: np.ndarray,
    sigma_eV: float,
    truncation_sigma: float = 8.0,
) -> np.ndarray:
    """
    Replace each discrete energy level with a Gaussian and sum.

    Parameters
    ----------
    energies_eV : np.ndarray
        Discrete state energies, in eV (as returned by
        :func:`energy_levels`). **Must be sorted ascending**, which is
        what :func:`energy_levels` guarantees; the truncation
        optimization below relies on this ordering. May be empty.
    egrid_eV : np.ndarray
        Energy grid on which to evaluate the broadened DOS, in eV.
        Assumed to be sorted ascending (as produced by
        ``numpy.linspace``).
    sigma_eV : float
        Gaussian standard deviation, in eV. Must be > 0.
    truncation_sigma : float, optional
        Each state's Gaussian is only evaluated for grid points within
        ``truncation_sigma * sigma_eV`` of that state's energy; beyond
        that the Gaussian is treated as zero. Must be > 0. The default
        of 8.0 makes the discarded tail ``exp(-8^2/2) = exp(-32) ~ 1.3e-14``
        of the peak, so the result is identical to the full untruncated
        sum to ~13 significant figures (see the regression test
        ``tests/test_physics.py::TestBroadenedDOS::
        test_truncated_matches_full_untruncated_sum``). This is a pure
        performance optimization: it does not change the DOS to any
        physically or visually meaningful precision. Lower it only if
        you knowingly want a coarser/faster approximation.

    Returns
    -------
    np.ndarray
        Broadened DOS evaluated on ``egrid_eV``, same shape as
        ``egrid_eV``. This is an **un-normalized** sum of unit-height
        Gaussians (arbitrary units) — it is not divided by
        ``sigma_eV * sqrt(2*pi)`` and is not a states/eV/volume
        density. Each Gaussian individually integrates to
        ``sigma_eV * sqrt(2*pi)``, but the array returned here is the
        sum of Gaussian *values*, not areas.

    Notes
    -----
    Only the states within ``truncation_sigma * sigma_eV`` of each grid
    point contribute measurably to that point, so for each grid point
    we use ``numpy.searchsorted`` on the sorted ``energies_eV`` to
    select just that window and evaluate Gaussians only there. This
    avoids the dominant cost of the naive approach (evaluating every
    state's Gaussian at every grid point, including negligible tails),
    which profiling showed to be ~99% of a full recompute for large
    boxes. Memory stays bounded because only the in-window states are
    materialized per grid point.
    """
    _validate_positive("sigma_eV", sigma_eV)
    _validate_positive("truncation_sigma", truncation_sigma)

    dos = np.zeros_like(egrid_eV, dtype=float)
    if energies_eV.size == 0:
        return dos

    two_sigma_sq = 2.0 * sigma_eV**2
    half_window_eV = truncation_sigma * sigma_eV

    # For each grid point, find the contiguous span of sorted states
    # whose energy lies within +/- half_window_eV of it. searchsorted is
    # vectorized over the whole grid at once.
    lo_idx = np.searchsorted(energies_eV, egrid_eV - half_window_eV, side="left")
    hi_idx = np.searchsorted(energies_eV, egrid_eV + half_window_eV, side="right")

    for i in range(egrid_eV.size):
        lo, hi = lo_idx[i], hi_idx[i]
        if hi > lo:
            delta = egrid_eV[i] - energies_eV[lo:hi]
            dos[i] = np.sum(np.exp(-(delta**2) / two_sigma_sq))
    return dos


def fermi_energy(sorted_energies_eV: np.ndarray, n_electrons: float) -> float:
    """
    Estimate the Fermi energy by filling discrete states from the
    bottom, two electrons per state (spin degeneracy), following the
    filling procedure of Batabyal & Dev, Physica E 64 (2014) 224-233
    (Section 4).

    Parameters
    ----------
    sorted_energies_eV : np.ndarray
        Discrete state energies in eV, sorted ascending (as returned
        by :func:`energy_levels`).
    n_electrons : float
        Total number of electrons to place in the box. Typically
        ``electron_density_m3 * box.volume_m3``. Must be > 0. Non-integer
        values are rounded up to the nearest whole electron count.

    Returns
    -------
    float
        Energy of the highest filled state, in eV.

    Raises
    ------
    ValueError
        If ``n_electrons`` is not finite/positive, or if there are
        not enough enumerated states below the cutoff used to
        generate ``sorted_energies_eV`` to accommodate the requested
        electron count. In the latter case, the caller must increase
        the energy cutoff (``e_max_eV`` in :func:`calculate_dos`) —
        the original script silently clipped to the last available
        state instead of raising, which could under-report the true
        Fermi energy for small/light boxes; this is a deliberate,
        documented behavior change (see CHANGELOG.md).
    """
    _validate_positive("n_electrons", n_electrons)

    n_states_available = sorted_energies_eV.size
    n_states_needed = int(np.ceil(n_electrons / 2.0))

    if n_states_needed > n_states_available:
        raise ValueError(
            "Not enough enumerated states to fill the requested electron "
            f"count: need {n_states_needed} states for {n_electrons:g} "
            f"electrons, but only {n_states_available} states were "
            "enumerated below the energy cutoff. Increase e_max_eV in "
            "calculate_dos() (or reduce electron_density_m3 / box volume)."
        )

    return float(sorted_energies_eV[n_states_needed - 1])


def fermi_dirac(
    energy_eV: np.ndarray, fermi_energy_eV: float, temperature_K: float
) -> np.ndarray:
    """
    Fermi-Dirac occupation function.

        f(E) = 1 / (exp((E - Ef) / (kB * T)) + 1)

    Parameters
    ----------
    energy_eV : np.ndarray or float
        Energy (energies), in eV.
    fermi_energy_eV : float
        Fermi energy, in eV.
    temperature_K : float
        Temperature, in Kelvin. Must be >= 0.

    Returns
    -------
    np.ndarray
        Occupation number(s) in [0, 1], same shape as ``energy_eV``.

    Notes
    -----
    At exactly ``temperature_K = 0`` this function returns the sharp
    step function (1 for E < Ef, 0.5 at E == Ef, 0 for E > Ef) rather
    than raising an error, by treating T=0 as a literal mathematical
    limit rather than dividing by zero.

    For E far above/below Ef relative to kB*T, the exponent is
    clipped to +/-500 before exponentiating to avoid floating-point
    overflow; this has no effect on the returned occupation, since
    exp(500) already saturates f(E) to 0.0 (and exp(-500) to 1.0) at
    double precision.

    The GUI and CLI in this package clamp the *displayed/used*
    temperature to a minimum of 1 K as a practical smoothing choice
    (see gui.py); this function itself supports the physical T=0
    limit directly and does not perform that clamping.
    """
    if not np.isfinite(temperature_K) or temperature_K < 0:
        raise ValueError(
            f"temperature_K must be finite and >= 0; got {temperature_K!r}."
        )

    energy_eV = np.asarray(energy_eV, dtype=float)

    if temperature_K == 0.0:
        occupation = np.where(
            energy_eV < fermi_energy_eV,
            1.0,
            np.where(energy_eV > fermi_energy_eV, 0.0, 0.5),
        )
        return occupation

    exponent = (energy_eV - fermi_energy_eV) * EV_TO_J / (BOLTZMANN * temperature_K)
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (np.exp(exponent) + 1.0)


def calculate_dos(
    box: QuantumBox,
    sigma_eV: float,
    electron_density_m3: float,
    e_max_eV: float = 12.0,
    n_points: int = 500,
) -> DOSResult:
    """
    Compute the Gaussian-broadened DOS and Fermi energy for a
    rectangular quantum box. This is the main public entry point of
    the scientific API.

    Parameters
    ----------
    box : QuantumBox
        Box geometry and effective mass.
    sigma_eV : float
        Gaussian broadening width, in eV. Must be > 0.
    electron_density_m3 : float
        Electron number density to fill the box with, in m^-3. Must
        be > 0. This parameter is required explicitly — the original
        script silently assumed bulk silver's electron density
        (5.86e28 m^-3) for every box; see
        constants.SILVER_ELECTRON_DENSITY_M3 for that value if you
        want to reproduce the original script's numbers.
    e_max_eV : float, optional
        Upper bound of the energy grid and state enumeration, in eV.
        Default 12.0 eV (matches the original script).
    n_points : int, optional
        Number of points in the energy grid. Default 500 (matches
        the original script).

    Returns
    -------
    DOSResult
        energy_eV, dos, fermi_energy_eV, n_states_enumerated.

    Raises
    ------
    ValueError
        For non-positive/non-finite sigma_eV, electron_density_m3,
        e_max_eV, or n_points, or if the requested electron count
        cannot be filled within the enumerated states (see
        :func:`fermi_energy`).
    """
    _validate_positive("sigma_eV", sigma_eV)
    _validate_positive("electron_density_m3", electron_density_m3)
    _validate_positive("e_max_eV", e_max_eV)
    if not isinstance(n_points, (int, np.integer)) or n_points <= 1:
        raise ValueError(f"n_points must be an integer > 1; got {n_points!r}.")

    egrid_eV = np.linspace(0.01, e_max_eV, n_points)

    energies_eV = energy_levels(box, e_max_eV=e_max_eV, sigma_eV=sigma_eV)
    dos = broadened_dos(energies_eV, egrid_eV, sigma_eV)

    n_electrons = electron_density_m3 * box.volume_m3
    ef_eV = fermi_energy(energies_eV, n_electrons)

    return DOSResult(
        energy_eV=egrid_eV,
        dos=dos,
        fermi_energy_eV=ef_eV,
        n_states_enumerated=int(energies_eV.size),
    )
