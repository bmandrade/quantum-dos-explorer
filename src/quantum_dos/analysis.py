# -*- coding: utf-8 -*-
"""
Analytical (idealized) DOS shapes and dimensional-regime
classification for a rectangular quantum box.

This module has no dependency on matplotlib. It returns *semantic*
classifications only (e.g. the string "2-D") — presentation choices
such as colors belong in the GUI layer (see gui.py).

Important scientific distinction
---------------------------------
The functions here compute an **idealized limiting DOS shape** for
the effective dimensionality of a box (e.g. the free-electron 3-D
sqrt(E) law). This is conceptually different from the **numerical
discrete-state DOS** in :mod:`quantum_dos.physics`, which is built by
explicitly enumerating a finite box's discrete energy levels and
Gaussian-broadening them. The two curves are not expected to be
identical even in the same units: the analytical curve represents
the E -> bulk-of-that-dimension limiting behaviour, while the
numerical curve retains finite-size oscillations/discreteness that
only wash out as the box grows large in the relevant directions (see
Batabyal & Dev, Physica E 64 (2014) 224-233, Figs. 1-3 and 6-8, for a
detailed discussion of this crossover).
"""

from __future__ import annotations

import numpy as np

from .models import ConfinementThresholds, QuantumBox

__all__ = ["analytical_dos", "classify_axis", "classify_regime"]


def _n_bulk_axes(box: QuantumBox, thresholds: ConfinementThresholds) -> int:
    return sum(L > thresholds.bulk_nm for L in box.dimensions_nm)


def analytical_dos(
    energy_eV: np.ndarray,
    box: QuantumBox,
    thresholds: ConfinementThresholds = ConfinementThresholds(),
) -> np.ndarray:
    """
    Idealized analytical DOS shape for the free-electron model,
    selected by how many box axes are classified as "bulk-like".

    The returned array is a *shape*, not a normalized DOS — it must
    be rescaled by the caller (e.g. to match a numerical DOS's peak)
    before being compared visually; see GUI code for an example.

    Classification (using ``thresholds.bulk_nm``, default 8.0 nm):

    - 3 bulk-like axes -> 3-D shape: g(E) ~ sqrt(E)      (Eq. 5 of
      Batabyal & Dev 2014)
    - 2 bulk-like axes -> 2-D shape: g(E) ~ constant (step-like)
      (Eq. 6)
    - 1 bulk-like axis  -> 1-D shape: g(E) ~ 1/sqrt(E)    (Eq. 7)
    - 0 bulk-like axes  -> 0-D: returned as all-zero, since the true
      0-D DOS is a sum of delta functions (Eq. 8), which has no
      smooth analytical "shape" to plot on a continuous grid; use
      the numerical DOS (physics.calculate_dos) to see the discrete
      peaks directly.

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy grid, in eV.
    box : QuantumBox
        Box geometry (only the dimensions are used).
    thresholds : ConfinementThresholds, optional
        Shared confinement/bulk length thresholds. Defaults to
        ``ConfinementThresholds()`` (confined_nm=2.0, bulk_nm=8.0).

    Returns
    -------
    np.ndarray
        Same shape as ``energy_eV``.
    """
    energy_eV = np.asarray(energy_eV, dtype=float)
    n_bulk = _n_bulk_axes(box, thresholds)

    if n_bulk == 3:
        return np.sqrt(np.clip(energy_eV, 0.0, None))
    if n_bulk == 2:
        return np.ones_like(energy_eV)
    if n_bulk == 1:
        # Floor to avoid a divide-by-zero singularity at E -> 0.
        floor_eV = 0.05
        return 1.0 / np.sqrt(np.maximum(energy_eV, floor_eV))
    return np.zeros_like(energy_eV)


def classify_axis(
    length_nm: float, thresholds: ConfinementThresholds = ConfinementThresholds()
) -> str:
    """
    Classify a single box axis as confined, crossover, or bulk-like.

    Returns
    -------
    str
        One of ``"confined"``, ``"crossover"``, ``"bulk-like"``.
    """
    if length_nm < thresholds.confined_nm:
        return "confined"
    if length_nm < thresholds.bulk_nm:
        return "crossover"
    return "bulk-like"


def classify_regime(
    box: QuantumBox, thresholds: ConfinementThresholds = ConfinementThresholds()
) -> str:
    """
    Classify the overall dimensional regime of a box.

    Returns one of: ``"0-D"``, ``"1-D"``, ``"2-D"``, ``"3-D"``,
    ``"crossover"``.

    Logic
    -----
    - If every axis is shorter than ``thresholds.confined_nm``: "0-D".
    - Otherwise, count axes longer than ``thresholds.bulk_nm``
      ("bulk-like" axes) -> that count directly gives "1-D"/"2-D"/"3-D".
    - Any other combination (e.g. one confined + one crossover +
      one bulk-like axis) is reported as "crossover", since it does
      not cleanly match a single lower-dimensional limit.

    Notes
    -----
    This mirrors the qualitative crossover-regime discussion in
    Batabyal & Dev, Physica E 64 (2014) 224-233, but with thresholds
    that are explicit, shared, and user-configurable via
    :class:`~quantum_dos.models.ConfinementThresholds` rather than
    hardcoded per-function magic numbers (see that class's docstring
    for the behavior-change note relative to the original script).
    """
    dims = box.dimensions_nm

    if all(L < thresholds.confined_nm for L in dims):
        return "0-D"

    n_bulk = _n_bulk_axes(box, thresholds)
    if n_bulk == 3:
        return "3-D"
    if n_bulk == 2:
        return "2-D"
    if n_bulk == 1:
        return "1-D"
    return "crossover"
