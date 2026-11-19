# API reference

This page is a concise reference for the public API of Quantum DOS
Explorer. It complements the in-source docstrings (which carry the full
detail) and the physics write-up in
[`scientific_background.md`](scientific_background.md).

The scientific API (`models`, `physics`, `analysis`, `constants`) has **no
matplotlib dependency** and can be imported, scripted, and tested without a
display. The convenience import surface is:

```python
from quantum_dos import (
    QuantumBox, ConfinementThresholds, DOSResult,   # data structures
    energy_levels, broadened_dos, fermi_energy,      # physics
    fermi_dirac, calculate_dos,
    analytical_dos, classify_axis, classify_regime,  # analysis
)
```

## Conventions

- **Units are in the names.** Every argument and field carries its unit as
  a suffix (`_eV`, `_nm`, `_m3`, `_K`) or is documented explicitly
  (`effective_mass` is in units of the free-electron mass `m_e`).
- **Invalid input raises `ValueError`.** Non-finite or non-physical values
  (zero/negative lengths, masses, broadening, densities, temperatures) are
  rejected at the boundary with a message naming the offending parameter.
  `assert` is reserved for internal invariants, never input validation.
- **The DOS is un-normalized** (arbitrary units): a sum of unit-height
  Gaussians, not a states·eV⁻¹·volume⁻¹ density. See `DOSResult.dos`.
- **Calculations are deterministic.** The same inputs always produce the
  same outputs; no random seed is used anywhere.

---

## `quantum_dos.models` — data structures

### `QuantumBox(lx_nm, ly_nm, lz_nm, effective_mass)`

Immutable (`frozen`) dataclass holding the geometry and effective mass of a
rectangular quantum box. Validates on construction: every side length and
the effective mass must be finite and `> 0`, else `ValueError`.

| Field | Unit | Meaning |
| --- | --- | --- |
| `lx_nm`, `ly_nm`, `lz_nm` | nm | box side lengths (Dirichlet boundary conditions) |
| `effective_mass` | `m_e` | electron effective mass (1.0 = free-electron mass) |

Properties: `dimensions_nm` → `(lx, ly, lz)` tuple; `volume_nm3` → volume in
nm³; `volume_m3` → volume in m³.

### `ConfinementThresholds(confined_nm=2.0, bulk_nm=8.0)`

Immutable dataclass of the two length thresholds used to classify an axis
(or box) as confined, in dimensional crossover, or bulk-like. Requires
`0 < confined_nm < bulk_nm`, else `ValueError`. These are a configurable
convention, **not** universal constants (see the class docstring and
CHANGELOG for the behaviour-change note versus the original script).

### `DOSResult(energy_eV, dos, fermi_energy_eV, n_states_enumerated)`

Immutable result object returned by `calculate_dos`.

| Field | Type / unit | Meaning |
| --- | --- | --- |
| `energy_eV` | ndarray (eV) | energy grid |
| `dos` | ndarray (arb.) | Gaussian-broadened DOS on that grid (un-normalized) |
| `fermi_energy_eV` | float (eV) | estimated Fermi energy |
| `n_states_enumerated` | int | number of discrete states enumerated below the cutoff |

---

## `quantum_dos.physics` — scientific core

### `calculate_dos(box, sigma_eV, electron_density_m3, e_max_eV=12.0, n_points=500) → DOSResult`

Main entry point. Enumerates the box's discrete levels, Gaussian-broadens
them onto an energy grid, and estimates the Fermi energy by filling states
(two electrons each) to the requested density.

| Argument | Unit | Notes |
| --- | --- | --- |
| `box` | — | `QuantumBox` |
| `sigma_eV` | eV | Gaussian broadening width; `> 0` |
| `electron_density_m3` | m⁻³ | filling density; `> 0`, required explicitly (no default) |
| `e_max_eV` | eV | grid/enumeration upper bound; default 12.0 |
| `n_points` | — | grid points; integer `> 1`, default 500 |

Raises `ValueError` for invalid parameters, or if the enumerated states
cannot accommodate the requested electron count (increase `e_max_eV` or
lower the density/volume).

### `energy_levels(box, e_max_eV, sigma_eV) → ndarray`

Enumerate discrete quantum-box energy eigenvalues (eV), sorted ascending,
up to `e_max_eV + 4·sigma_eV` (the margin keeps near-cutoff Gaussian tails
correct when broadening onto a grid ending at `e_max_eV`). Degenerate states
are not merged.

### `broadened_dos(energies_eV, egrid_eV, sigma_eV, truncation_sigma=8.0) → ndarray`

Replace each (sorted) discrete level with a Gaussian and sum onto
`egrid_eV`. `truncation_sigma` bounds each Gaussian to ±`truncation_sigma·σ`
of its centre — a pure performance optimization that matches the full
untruncated sum to ~13 significant figures at the default of 8.0. Returns an
un-normalized sum of Gaussian values (arbitrary units).

### `fermi_energy(sorted_energies_eV, n_electrons) → float`

Energy (eV) of the highest filled state when `n_electrons` are placed two
per state from the bottom. `n_electrons` is typically
`electron_density_m3 · box.volume_m3`. Raises `ValueError` if there are too
few enumerated states to hold the electrons (a deliberate change from the
original script, which silently clipped — see CHANGELOG).

### `fermi_dirac(energy_eV, fermi_energy_eV, temperature_K) → ndarray`

Fermi–Dirac occupation `f(E) = 1 / (exp((E−E_f)/k_B T) + 1)`, in `[0, 1]`.
`temperature_K` must be `≥ 0`; exactly `0` returns the sharp step (1 below
`E_f`, 0.5 at `E_f`, 0 above) as a literal limit rather than dividing by
zero.

---

## `quantum_dos.analysis` — idealized shapes & classification

No matplotlib dependency; returns *semantic* results only (strings,
arrays) — colour and other presentation choices live in the GUI layer.

### `analytical_dos(energy_eV, box, thresholds=ConfinementThresholds()) → ndarray`

Idealized limiting DOS **shape** selected by the number of bulk-like axes:
3 → `√E` (3-D), 2 → constant (2-D), 1 → `1/√E` (1-D), 0 → all zeros (0-D has
no smooth shape; use the numerical DOS). The result is an unnormalized shape
to be rescaled by the caller; it is the `E → bulk` limit, not the
finite-size numerical DOS.

### `classify_axis(length_nm, thresholds=ConfinementThresholds()) → str`

Classify one axis as `"confined"` (`< confined_nm`), `"crossover"`
(between), or `"bulk-like"` (`≥ bulk_nm`).

### `classify_regime(box, thresholds=ConfinementThresholds()) → str`

Classify the whole box as `"0-D"`, `"1-D"`, `"2-D"`, `"3-D"`, or
`"crossover"`. `"0-D"` if every axis is confined; otherwise the label is set
by the count of bulk-like axes; `"crossover"` only when no axis is bulk-like
and not all are confined.

---

## `quantum_dos.constants`

CODATA-2018 values in SI (truncated to the original script's precision so
results are unchanged by the refactor): `HBAR` (J·s), `ELECTRON_MASS` (kg),
`EV_TO_J` (J/eV), `BOLTZMANN` (J/K). Also `SILVER_ELECTRON_DENSITY_M3`
(5.86×10²⁸ m⁻³) — an illustrative example value, **not** a default used by
the engine; every caller supplies `electron_density_m3` explicitly.

---

## `quantum_dos.cli` — command-line interface

Installed as the `quantum-dos` console script (entry point
`quantum_dos.cli:main`).

- `build_parser() → argparse.ArgumentParser` — construct the parser
  (box geometry, physics parameters, numerical grid, thresholds, output);
  separated from `main` so it can be inspected and tested on its own.
- `main(argv=None) → int` — parse arguments and run one calculation;
  returns the process exit code (0 success, 1 on a handled calculation
  error, 2 on an argparse usage error).

Run `quantum-dos --help` for the full option list. With `--output PATH` it
writes a CSV with columns `energy_eV, dos, occupied_dos`.

---

## Plotting layer (not part of the scientific API)

`quantum_dos.gui` and `quantum_dos.gui_fast` provide the interactive
Matplotlib explorer and its cached fast-path. They depend on Matplotlib and
are intentionally kept out of the scientific API surface above. Their
numerical behaviour is pinned to the scientific core by the equivalence
tests (see [`testing_strategy.md`](testing_strategy.md)); their visual
mechanics are not unit-tested.
