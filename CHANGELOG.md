# Changelog

All notable changes to this project are documented here. This
project has not yet made a numbered public release; entries below
describe the transformation from the original single-file
`dos_explorer.py` prototype into the `quantum_dos` package.

## [Unreleased]

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
