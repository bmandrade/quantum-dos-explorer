# Quantum DOS Explorer

Explore how quantum confinement changes the electronic **density of
states (DOS)** of a rectangular quantum box, using a simple free-electron
model with Gaussian-broadened discrete energy levels.

The package provides a clean, tested scientific API (pure NumPy, no
plotting dependency), a command-line interface for non-interactive
calculations, and an interactive Matplotlib GUI for exploration.

## Overview

Given a box geometry (`Lx, Ly, Lz`), an electron effective mass, a
broadening width, and an electron density, the tool:

- enumerates the discrete particle-in-a-box energy levels,
- builds a Gaussian-broadened numerical DOS,
- estimates the Fermi energy by filling those levels,
- computes the Fermi-Dirac occupied DOS,
- classifies the box's dimensional regime (0-D / 1-D / 2-D / 3-D /
  crossover), and
- compares the numerical DOS against the idealized analytical DOS shape
  for that dimensionality.

The physical model follows R. Batabyal and B. N. Dev, *Electronic
structure in the crossover regimes in lower dimensional structures*,
Physica E **64** (2014) 224-233,
[doi:10.1016/j.physe.2014.07.017](https://doi.org/10.1016/j.physe.2014.07.017).

## Scientific background

A brief summary (full detail in
[`docs/scientific_background.md`](docs/scientific_background.md)):

- **Quantum confinement.** Confining electrons to a small box discretizes
  their allowed energies. The energy levels of a rectangular box with
  Dirichlet boundary conditions are
  `E(nx,ny,nz) = (hbar^2 pi^2 / 2m*)(nx^2/Lx^2 + ny^2/Ly^2 + nz^2/Lz^2)`.
- **Density of states.** The exact confined DOS is a set of sharp peaks
  (delta functions), one per level. Replacing each with a Gaussian of
  width `sigma` (representing finite experimental resolution) gives a
  smooth, plottable numerical DOS.
- **Fermi level and occupation.** Filling the levels from the bottom with
  a known number of electrons (two per state) gives the Fermi energy;
  the Fermi-Dirac function then gives the occupation at temperature `T`.
- **Dimensional limits.** As a box grows large in one or more directions,
  its DOS approaches an idealized shape: `sqrt(E)` (3-D), constant (2-D),
  or `1/sqrt(E)` (1-D). The numerical (broadened, finite-size) DOS and
  these analytical limits are **different quantities** and are treated
  as such throughout.

## Installation

Requires Python >= 3.10.

```bash
# Core scientific API + CLI (NumPy only):
pip install .

# Also install the interactive GUI (adds matplotlib):
pip install ".[gui]"

# For development (tests + coverage):
pip install -e ".[dev,gui]"
```

## Quick start

### API

```python
from quantum_dos import QuantumBox, calculate_dos
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3

box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
result = calculate_dos(
    box,
    sigma_eV=0.05,
    electron_density_m3=SILVER_ELECTRON_DENSITY_M3,
)
print(result.fermi_energy_eV)   # -> 5.9413 (eV)
```

### CLI

```bash
quantum-dos --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 --density 5.86e28
```

## Interactive application

Launch the GUI with:

```bash
quantum-dos-gui
```

Sliders control `Lx, Ly, Lz`, effective mass `m*`, temperature `T`,
broadening `sigma`, and electron density `n`. The plot shows the total
DOS, the analytical limit, the Fermi-Dirac-occupied DOS, and the Fermi
level, with a dimensional-regime badge and a state summary. A **Reset**
button restores defaults. (Requires the `[gui]` extra.)

For large boxes the naive GUI can lag on every slider move, so an
optimized version is also provided:

```bash
quantum-dos-gui-fast
```

It looks and behaves identically and produces the same numbers (verified
by a cross-check test), but stays smooth via three presentation-layer
optimizations, none of which change the physics — see **Performance**
below.

### Performance

The interactive experience was profiled and optimized without altering
any scientific result (a regression test asserts the DOS is identical to
~14 significant figures, and a cross-check test asserts the two GUIs
agree byte for byte):

- **Truncated-Gaussian broadening** (in the science core): the dominant
  cost, `broadened_dos`, now evaluates each state's Gaussian only near
  its peak (within `8 sigma`), giving ~16x–55x faster DOS construction
  with output identical to the full sum to floating-point precision.
- **Result caching** (`gui_fast`): temperature changes need no recompute
  at all; density changes reuse the DOS curve and only re-derive the
  Fermi energy; only geometry/mass/`sigma` changes recompute the curve.
- **Debouncing**: a slider drag triggers one recompute at its end
  rather than dozens.

On a 20×20×20 nm box (≈760k states) this turns a ~4 s-per-move naive
update into an interactive one (temperature/density moves are near
instant because they skip the recompute entirely).

## Parameters

| Parameter    | Meaning                                   | Unit           | Default (GUI) |
| ------------ | ----------------------------------------- | -------------- | ------------- |
| `Lx, Ly, Lz` | Box side lengths                          | nm             | 5, 5, 5       |
| `m*`         | Electron effective mass                   | units of `m_e` | 1.0           |
| `sigma`      | Gaussian broadening width                 | eV             | 0.05          |
| `T`          | Temperature (for occupation)              | K              | 300           |
| `n`          | Electron number density (fills the box)   | m^-3           | 5.86e28 (Ag)  |
| `e_max`      | Energy-grid upper bound / state cutoff    | eV             | 12.0          |
| `n_points`   | Number of energy-grid points              | -              | 500           |
| `confined_nm`| Axis shorter than this is "confined"      | nm             | 2.0           |
| `bulk_nm`    | Axis longer than this is "bulk-like"      | nm             | 8.0           |

All units are explicit in the API (field/argument names carry them).

## Scientific model

- **Model:** free-electron (parabolic band) particle in a rectangular
  box with Dirichlet boundary conditions.
- **Energy levels:** `E(nx,ny,nz) = (hbar^2 pi^2 / 2m*) * sum_i n_i^2/L_i^2`.
- **Numerical DOS:** sum of unit-height Gaussians (width `sigma`), one
  per enumerated discrete state -- arbitrary units (see Limitations).
- **Fermi energy:** fill sorted levels, two electrons per state, up to
  `n * volume` electrons.
- **Occupation:** Fermi-Dirac `f(E) = 1/(exp((E-Ef)/kB T)+1)`.

See [`docs/scientific_background.md`](docs/scientific_background.md) for
the equations, assumptions, and their relation to the reference paper.

## Numerical method

States are generated by enumerating quantum numbers on a 3D grid
(`numpy.meshgrid`) up to a per-axis cutoff, computing each combination's
energy, discarding those above `e_max + 4*sigma`, and sorting. Each
surviving level is broadened by a Gaussian and summed onto the energy
grid (in memory-bounded chunks). Degenerate states are counted with
their correct multiplicity. Full detail in the scientific-background
document, section 2.

## Analytical limits

`analytical_dos` returns the idealized free-electron DOS *shape* for the
effective dimensionality (`sqrt(E)`, constant, or `1/sqrt(E)`), selected
by how many axes exceed `bulk_nm`. This limiting shape is **not** the
same object as the numerical broadened DOS: the numerical curve is a
finite box's actual (oscillating) spectrum, while the analytical curve
is the large-size limit of that dimensionality. Any overlay of the two
rescales the analytical shape for visual comparison only.

## Examples

Executable scripts in [`examples/`](examples/):

```bash
python examples/basic_calculation.py       # API summary, no GUI
python examples/dimensional_crossover.py   # size sweep, 0-D -> 2-D
python examples/publication_figure.py      # saves a static DOS figure (needs [gui])
```

None depend on local paths; the figure example takes an optional output
path argument.

## Testing

```bash
pytest                       # run all tests
pytest --cov=quantum_dos     # with coverage
```

The suite tests scientific behavior (energies vs. the analytical
formula, DOS properties, Fermi filling, Fermi-Dirac limits, dimensional
classification, input validation) rather than just execution, and
includes regression tests for bugs found while refactoring, plus a
headless GUI smoke test (skipped automatically if matplotlib is absent).

## Development

```bash
git clone <your-fork-url>
cd quantum-dos-explorer
pip install -e ".[dev,gui]"
pytest
```

The package is laid out under `src/quantum_dos/`:
`constants.py`, `models.py`, `physics.py`, `analysis.py` (the
matplotlib-free scientific core), plus `cli.py` and `gui.py`
(presentation layers). The GUI contains no physics; it only calls the
API and maps semantic classifications to colors.

## Limitations

This is a deliberately simple exploratory model. Be aware that:

- **Free-electron / parabolic-band assumption.** Real band structure
  (non-parabolicity, multiple bands, band gaps) is ignored.
- **Effective-mass approximation.** A single scalar, isotropic effective
  mass is used; real materials can have anisotropic/energy-dependent
  masses.
- **Rectangular-box, hard-wall geometry.** Dirichlet (infinite-well)
  boundary conditions; no finite barrier height, surface states, or
  non-rectangular shapes.
- **Finite energy cutoff.** Only states below `e_max + 4*sigma` are
  enumerated. Fillings that would exceed this raise an error rather than
  silently truncating -- increase `e_max` if needed.
- **Gaussian broadening.** `sigma` is a phenomenological resolution
  parameter, not derived from a physical lifetime; it sets both the
  visual smoothness and the enumeration margin.
- **DOS is in arbitrary units.** The numerical DOS is an un-normalized
  sum of Gaussians, not states/eV/volume.
- **Dimensional-classification thresholds are conventional.** The
  `confined_nm` / `bulk_nm` values are configurable defaults, not
  universal physical constants; the true crossover length is material-,
  mass-, and resolution-dependent.
- **Finite-size effects.** The numerical DOS retains discreteness/
  oscillations that only approach the analytical limit for large boxes;
  the two curves are compared by shape, not equated.
- **Performance.** State enumeration scales roughly with the product of
  the per-axis state counts, i.e. it grows quickly for large boxes and/or
  small effective mass. Very large boxes can be slow or memory-heavy.

## License

MIT -- see [`LICENSE`](LICENSE).
