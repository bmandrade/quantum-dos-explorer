# Scientific background and numerical method

This document explains the physics implemented by Quantum DOS Explorer
and how the numerical density of states (DOS) is constructed. It is
intended to give a researcher enough context to use, trust, and modify
the software. It is not a full textbook treatment.

The model follows the free-electron approach of

> R. Batabyal and B. N. Dev,
> *Electronic structure in the crossover regimes in lower dimensional
> structures*, Physica E **64** (2014) 224-233,
> <https://doi.org/10.1016/j.physe.2014.07.017>

which is the direct reference for the equations below. Equation numbers
in parentheses refer to that paper.

## 1. Physical model

Electrons are treated as **free (parabolic-band) particles** confined
to a rectangular box of sides `Lx, Ly, Lz` with **Dirichlet boundary
conditions** (the wavefunction vanishes on the box surface). This is
the particle-in-a-3D-box problem. Its energy eigenvalues are (Eq. 2)

```
E(nx, ny, nz) = (hbar^2 * pi^2) / (2 * m*) *
                ( nx^2 / Lx^2 + ny^2 / Ly^2 + nz^2 / Lz^2 )
```

with quantum numbers `nx, ny, nz = 1, 2, 3, ...` and `m*` the electron
effective mass. In this package the per-axis energy coefficient
`A_i = (hbar^2 * pi^2) / (2 * m* * L_i^2)` is computed once and the
energy of each state is `sum_i A_i * n_i^2` (see
`physics.energy_levels`).

Dirichlet (rather than periodic) boundary conditions are used because,
as the reference paper notes, periodic boundary conditions are only
appropriate in the bulk limit; for small/finite systems Dirichlet
conditions are the correct choice.

## 2. Density of states: numerical construction

The exact 0-D DOS is a sum of Dirac delta functions, one per discrete
state (Eq. 8). To obtain something plottable on a continuous energy
grid, each delta is replaced by a **normalized-width Gaussian** of
standard deviation `sigma` (Eq. 9, used here at finite `sigma` rather
than in the mathematical `sigma -> 0` limit):

```
g(E) = sum over states  exp( -(E - E_state)^2 / (2 * sigma^2) )
```

`sigma` represents a physical broadening -- e.g. finite experimental
energy resolution. The reference paper uses `sigma = 0.1 eV`
(corresponding to ~0.235 eV FWHM, a typical STS resolution); this
package defaults to `sigma = 0.05 eV` in the GUI but the value is fully
user-controlled.

**Units.** The DOS returned by this package is an **un-normalized sum
of unit-height Gaussians** (arbitrary units). It is *not* divided by
`sigma * sqrt(2*pi)` and is *not* expressed as states/eV/volume. This
is sufficient for exploring the *shape* and relative features of the
DOS, which is the purpose of the tool. If you need an absolutely
normalized DOS in physical units, you must rescale accordingly.

### How states are generated

`physics.energy_levels` enumerates quantum numbers on a 3D grid. The
maximum quantum number per axis is chosen so that the single-axis
energy reaches the cutoff:

```
n_max_i = ceil( sqrt( (E_max + 4*sigma) / A_i ) ) + 2
```

All `(nx, ny, nz)` combinations are formed with `numpy.meshgrid`, their
energies computed, and those above `E_max + 4*sigma` discarded (the
`4*sigma` margin ensures Gaussian tails from states just above `E_max`
still contribute correctly on a grid that ends at `E_max`). The
surviving energies are sorted ascending.

Degenerate states are **not** merged: each `(nx, ny, nz)` contributes
its own entry, so a `k`-fold degenerate level contributes `k` Gaussians
(equivalently, a Gaussian of `k` times the height), which is the
physically correct weighting.

## 3. Analytical limiting DOS

Separately from the numerical curve, `analysis.analytical_dos` returns
the **idealized free-electron DOS shape** for the effective
dimensionality of the box (Eqs. 5-8):

| Effective dimension | Shape        |
| ------------------- | ------------ |
| 3-D                 | `g ~ sqrt(E)` |
| 2-D                 | `g ~ const`   |
| 1-D                 | `g ~ 1/sqrt(E)` |
| 0-D                 | (delta functions; returned as zero on a continuous grid) |

**This is a different quantity from the numerical DOS.** The analytical
curve is the limiting shape a system of that dimensionality approaches
as it becomes large in the extended directions; the numerical curve is
the actual broadened discrete spectrum of a *finite* box, which retains
finite-size oscillations that only wash out as the box grows. The GUI
and the `publication_figure.py` example rescale the analytical shape to
the numerical peak purely so the two shapes can be compared on one axis
-- this is a visual aid, not an assertion that they are equal.

The effective dimensionality is decided by counting how many axes
exceed the "bulk-like" length threshold (see next section).

## 4. Dimensional classification

Whether an axis is "confined", "crossover", or "bulk-like" is decided
by two length thresholds held in `models.ConfinementThresholds`:

- axis length `< confined_nm` (default 2.0 nm): **confined**
- `confined_nm <= length < bulk_nm` (default 2.0-8.0 nm): **crossover**
- length `>= bulk_nm` (default 8.0 nm): **bulk-like**

The overall regime (`0-D`/`1-D`/`2-D`/`3-D`/`crossover`) is then:

- all three axes confined -> `0-D`
- otherwise, the number of bulk-like axes gives `1-D`/`2-D`/`3-D`
- anything else (no bulk-like axis, but not all confined) -> `crossover`

**These thresholds are a practical convention, not universal
constants.** The reference paper shows the confined-to-extended
crossover length depends on the material, effective mass, and energy
resolution; e.g. for their Ag system a lateral size of ~15 nm is needed
before a slab behaves cleanly 2-D. Change the thresholds explicitly if
your system requires it. (The original prototype script hardcoded three
*different* thresholds in different functions; this package unifies them
into one configurable object -- see `CHANGELOG.md`.)

## 5. Fermi energy

Given a total electron count `N = electron_density * box_volume`, the
Fermi energy is found by sorting the discrete levels ascending and
filling them from the bottom, **two electrons per state** (spin
degeneracy), until all `N` electrons are placed. The energy of the
highest filled state is the Fermi energy (Section 4 of the reference
paper; `physics.fermi_energy`).

The electron density is a **required, explicit parameter** of the API.
The original prototype silently assumed bulk silver's value
(`5.86e28 m^-3`) for every box; that value is still available as
`constants.SILVER_ELECTRON_DENSITY_M3` for anyone who wants to
reproduce the prototype's numbers, but it is never applied implicitly.

If the requested electron count needs more states than were enumerated
below `E_max`, `fermi_energy` raises `ValueError` (rather than silently
clipping, as the prototype did). Increase `e_max_eV` in that case.

## 6. Fermi-Dirac occupation

Occupation at temperature `T` is the standard Fermi-Dirac function
(`physics.fermi_dirac`):

```
f(E) = 1 / ( exp( (E - Ef) / (kB * T) ) + 1 )
```

- At `T = 0` the function returns the exact step (1 below `Ef`, 0.5 at
  `Ef`, 0 above), handled as a literal limit rather than a division by
  zero.
- For energies far from `Ef`, the exponent is clipped to `+/-500`
  before exponentiation to avoid floating-point overflow; this does not
  change the returned values (they have already saturated to 0 or 1 at
  double precision).
- The GUI clamps the *temperature it passes in* to a minimum of 1 K
  (a presentation smoothing choice, so an infinitely sharp step does not
  render as a vertical line). The physics function itself supports the
  true `T = 0` limit directly.

## 7. Validation against the analytical formula

The test suite checks the numerical implementation against the closed-
form particle-in-a-box result: the ground-state energy `E(1,1,1)`
returned by `energy_levels` matches the direct evaluation of Eq. (2) to
better than 1 part in 10^8
(`tests/test_physics.py::TestEnergyLevels::test_ground_state_energy_matches_particle_in_a_box_formula`).

Qualitative limiting behavior is also asserted as automated tests:

- larger box -> smaller level spacing (denser levels);
- heavier effective mass -> smaller kinetic-energy spacing (lower
  ground state);
- larger `sigma` -> smoother DOS (fewer resolved peaks);
- Fermi-Dirac limits (`E << Ef -> 1`, `E >> Ef -> 0`, `T -> 0` step).

## 8. Reproducibility

The entire pipeline is deterministic. There is no random number
generation anywhere; identical inputs always produce identical outputs
(asserted in
`tests/test_physics.py::TestCalculateDOS::test_same_inputs_give_identical_results_every_call`).
