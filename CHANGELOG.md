# Changelog

All notable changes to this project are documented here. This
project has not yet made a numbered public release; entries below
describe the transformation from the original single-file
`dos_explorer.py` prototype into the `quantum_dos` package.

## [Unreleased]

### Performance (optimized GUI, identical results)
- **New `quantum-dos-gui-fast` interactive GUI** (`gui_fast.py`): a
  drop-in alternative to `quantum-dos-gui` with identical appearance,
  controls, and numerical results, but much smoother interaction. The
  original `gui.py` is preserved unchanged. A cross-check test
  (`tests/test_gui_equivalence.py`) asserts the two GUIs produce
  byte-for-byte identical DOS curves, Fermi levels, regime labels, and
  state-summary text across all regimes.
- **`broadened_dos` optimized with truncated Gaussians** (science core).
  Each state's Gaussian is now evaluated only within `truncation_sigma *
  sigma` (default 8 sigma) of each grid point, using `searchsorted` on
  the sorted energies. This is output-preserving: the discarded tail is
  `exp(-32) ~ 1.3e-14` of the peak, so the DOS is identical to the full
  sum to ~14 significant figures (regression-tested to `rtol=1e-12`).
  Measured speedups: 16x (5 nm box), 55x (20 nm box, 759k states).
- **Result caching in the fast GUI** (`DosCache`): temperature changes
  trigger no physics recompute (only occupation re-evaluates); density
  changes reuse the cached DOS curve and re-fill only the Fermi energy;
  geometry/mass/sigma changes do a full recompute. A density-only change
  on the 20 nm box drops from ~158 ms to ~0.006 ms.
- **Debounced slider events**: a drag's burst of events collapses to a
  single recompute at its end.
- The whole optimization changes only speed, not the physics; the
  scientific core's results are unchanged to floating-point precision.
  (An earlier iteration also blitted the plot, but that relied on
  ``animated`` artists which some interactive backends fail to render on
  initial show, producing a blank plot; it was removed in favor of a
  plain, robust redraw. The caching and truncated-Gaussian broadening
  already provide the bulk of the speedup.)

### Changed
- The GUI temperature slider now **defaults to 0 K** (true
  zero-temperature Fermi-Dirac step) instead of 300 K, in both GUIs.

### Added
- Installable Python package (`quantum_dos`) with a pure-NumPy
  scientific core (`constants.py`, `models.py`, `physics.py`,
  `analysis.py`), independent of matplotlib.
- `pytest` test suite (75 tests, 100% line coverage of the scientific
  core) including regression tests for bugs found during refactoring.
- `pyproject.toml` packaging (`pip install .` / `pip install -e ".[dev]"`),
  MIT `LICENSE`, `.gitignore`.

### Changed (deliberate behavior differences from the original script)
- **Electron density is now a required, explicit parameter**
  (`electron_density_m3`) of `physics.calculate_dos`, instead of a
  hidden constant (bulk silver's density) used unconditionally for
  every box. The silver value is still available, explicitly labeled,
  as `constants.SILVER_ELECTRON_DENSITY_M3`, for anyone who wants to
  reproduce the original script's numbers.
- **Confinement/bulk length thresholds are unified** into one shared,
  configurable `models.ConfinementThresholds` (default
  `confined_nm=2.0`, `bulk_nm=8.0`), used consistently by
  `analysis.analytical_dos`, `analysis.classify_regime`, and
  `analysis.classify_axis`. The original script used three different,
  inconsistent hardcoded threshold sets (2.0 nm in `analytical_dos`,
  5.0 nm in `classify_regime`, 2.0/8.0 nm in `dim_color`/`dim_label`).
  **This changes GUI-visible output for boxes with one or more axes
  between 2-5 nm or 5-8 nm** compared to the original script.
- **`physics.fermi_energy` now raises `ValueError`** if the requested
  electron count needs more states than were enumerated below the
  energy cutoff, instead of silently clipping the fill index to the
  last available state. The original script's silent clipping could
  under-report the Fermi energy for small-box / low-effective-mass
  combinations without any warning (see `tests/test_physics.py::TestFermiEnergy::test_raises_when_requested_electrons_exceed_available_states`
  for a reproduction case).

### Notes
- The underlying physical model (particle-in-a-rectangular-box energy
  levels, Gaussian-broadened discrete-state DOS, Fermi-Dirac
  occupation) is unchanged; the DOS array produced by
  `physics.calculate_dos` is numerically identical (`np.allclose`) to
  the original script's `calculate_physics` for inputs that do not
  trigger the Fermi-energy clipping bug above.
