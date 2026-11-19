# Testing strategy

This page describes how Quantum DOS Explorer is tested and why. The suite
has **112 tests** (`pytest`), all passing, and runs headless in seconds.

```bash
pytest                 # run everything
pytest -q tests/test_physics.py::TestCalculateDOS   # one class
```

## Principles

- **Behaviour-driven docstrings.** Every test opens with *why* it exists and
  a GIVEN / WHEN / THEN scenario. The docstring states the physical or
  software property being protected, so a failure is read as "this property
  broke", not "line N changed".
- **One test fails for one reason.** Each test checks a single concept.
  Multi-assert checks were split so that a red test points at one cause;
  shared setup lives in small documented helpers rather than being repeated.
- **Prefer oracles over golden values.** Where an exact answer is known from
  physics, the test compares against that closed form rather than a
  recorded number, so the test encodes understanding and does not drift.
- **Exceptions, not asserts, for input.** Invalid input is expected to raise
  `ValueError`; tests assert that it does. `assert` in the suite checks
  computed results, never user-input validation.
- **Determinism.** The pipeline uses no randomness, so tests assert that
  identical inputs give byte-identical results across repeated calls.

## What the tests cover

**Oracle tests — against exact physics.** The ground-state energy matches
the analytic particle-in-a-box formula; filling places exactly two electrons
per state; a single level's broadened DOS peaks at that level's energy; the
Fermi–Dirac function hits its known limits (→1 below `E_f`, exactly 0.5 at
`E_f`, →0 above, and a sharp step at `T=0`); the idealized `analytical_dos`
shapes reproduce `√E` (3-D), constant (2-D) and `1/√E` (1-D); and the
physical constants match their CODATA-2018 values.

**Invariant / property tests.** The DOS is always finite and non-negative on
the whole grid; result arrays have the grid shape; occupied DOS never
exceeds total DOS; and qualitative trends hold — a heavier effective mass
gives smaller level spacing, a larger box gives smaller spacing than a
smaller one, a larger broadening `σ` yields a smoother curve with fewer
local maxima, and a higher temperature smears the occupation more.

**Regression / performance-safety test.** The truncated Gaussian sum (the
`truncation_sigma` optimization) is asserted to match the full untruncated
sum to high precision, so the speed optimization can never silently change
the numbers.

**Validation tests.** Non-positive or non-finite geometry, mass, broadening,
density, cutoff, grid size and temperature each raise `ValueError`; an
electron count exceeding the enumerated states raises with an actionable
message; the dimensional-classification thresholds reject
`confined_nm ≥ bulk_nm`.

**CLI tests.** `main(argv)` is exercised in-process (no subprocess, no
display): exit codes (0 success, 1 handled error, 2 usage error), the
`--output` CSV header and row count, the finiteness of every output column,
`occupied_dos ≤ dos`, `--quiet`, and that the installed `quantum-dos` script
actually runs.

## The GUI: what is and isn't tested

Per the project convention, **visualization mechanics are not unit-tested** —
there are no tests that a slider moved a line or that an axis has a given
colour. What *is* tested is the GUI's **numerical** behaviour, which is where
a bug would corrupt results:

- `test_gui_fast.py` pins the fast-path DOS **cache**: its first result
  equals a direct `calculate_dos` call; geometry / mass / `σ` changes trigger
  a real recompute; a density-only change reuses the cached curve but updates
  the Fermi energy; and it propagates errors rather than hiding them.
- `test_gui_equivalence.py` pins the fast GUI's numbers to the reference
  implementation, including after the occupied-state shading step.

This keeps the tests focused on correctness that matters, while leaving
pixel-level presentation to manual inspection.

## Organization

| File | Tests | Focus |
| --- | --- | --- |
| `test_models.py` | 17 | dataclass validation, properties, constants |
| `test_physics.py` | 37 | levels, broadening, filling, Fermi–Dirac, `calculate_dos` |
| `test_analysis.py` | 16 | analytical shapes, axis/box classification |
| `test_cli.py` | 13 | command-line behaviour and CSV output |
| `test_gui_fast.py` | 8 | fast-path DOS cache (numerical) |
| `test_gui_equivalence.py` | 2 | fast GUI vs reference equivalence |

Coverage of the scientific core (`models`, `physics`, `analysis`) is
effectively complete; the plotting modules are covered only at the numerical
boundary described above. Tests force the headless Matplotlib backend
(`Agg`) where needed and require no network access.
