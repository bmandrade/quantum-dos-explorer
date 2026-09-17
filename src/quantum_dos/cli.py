# -*- coding: utf-8 -*-
"""
Command-line interface for Quantum DOS Explorer.

Runs a single density-of-states calculation without opening the GUI,
prints a human-readable summary, and optionally saves the numerical
results to a CSV file.

Examples
--------
Basic calculation, summary printed to stdout::

    quantum-dos --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 --density 5.86e28

Save the full DOS curve to a file::

    quantum-dos --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 \\
        --density 5.86e28 --output results.csv

See ``quantum-dos --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import csv
import sys

from .analysis import classify_regime
from .constants import SILVER_ELECTRON_DENSITY_M3
from .models import ConfinementThresholds, QuantumBox
from .physics import calculate_dos, fermi_dirac

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="quantum-dos",
        description=(
            "Compute the Gaussian-broadened density of states (DOS) and "
            "Fermi energy of a rectangular quantum box (free-electron "
            "model), without opening the interactive GUI."
        ),
        epilog=(
            "All calculations are deterministic: the same inputs always "
            "produce the same numerical results (no random seed is used "
            "or needed anywhere in this pipeline).\n\n"
            f"Example: quantum-dos --lx 5 --ly 5 --lz 5 --mass 1.0 "
            f"--sigma 0.05 --density {SILVER_ELECTRON_DENSITY_M3:.3g} "
            f"  (the last value is bulk silver's conduction-electron "
            f"density, provided here only as a realistic example)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    geometry = parser.add_argument_group("box geometry")
    geometry.add_argument(
        "--lx", type=float, required=True, metavar="NM",
        help="Box side length along x, in nanometres. Must be > 0.",
    )
    geometry.add_argument(
        "--ly", type=float, required=True, metavar="NM",
        help="Box side length along y, in nanometres. Must be > 0.",
    )
    geometry.add_argument(
        "--lz", type=float, required=True, metavar="NM",
        help="Box side length along z, in nanometres. Must be > 0.",
    )

    physics = parser.add_argument_group("physics parameters")
    physics.add_argument(
        "--mass", type=float, required=True, metavar="M_E",
        help="Electron effective mass, in units of the free-electron "
             "mass m_e (e.g. 1.0 = free-electron mass). Must be > 0.",
    )
    physics.add_argument(
        "--sigma", type=float, required=True, metavar="EV",
        help="Gaussian broadening width of each discrete energy level, "
             "in eV. Must be > 0. Typical experimental resolution: "
             "0.05-0.2 eV.",
    )
    physics.add_argument(
        "--density", type=float, required=True, metavar="M^-3",
        help="Electron number density used to fill the box and "
             "determine the Fermi energy, in electrons per cubic "
             "metre. Must be > 0. There is no default: this value is "
             "material-dependent and must be supplied explicitly "
             f"(e.g. bulk silver is approximately "
             f"{SILVER_ELECTRON_DENSITY_M3:.3g} m^-3).",
    )
    physics.add_argument(
        "--temperature", type=float, default=300.0, metavar="K",
        help="Temperature used for the Fermi-Dirac occupied-DOS "
             "column in --output, in Kelvin. Must be >= 0. "
             "Default: 300 K.",
    )

    numerics = parser.add_argument_group("numerical grid")
    numerics.add_argument(
        "--e-max", type=float, default=12.0, metavar="EV", dest="e_max",
        help="Upper bound of the energy grid and state-enumeration "
             "cutoff, in eV. Must be > 0. Default: 12.0 eV. Increase "
             "this if you get a 'not enough enumerated states' error.",
    )
    numerics.add_argument(
        "--n-points", type=int, default=500, metavar="N", dest="n_points",
        help="Number of points in the energy grid. Default: 500.",
    )

    classification = parser.add_argument_group("dimensional classification")
    classification.add_argument(
        "--confined-nm", type=float, default=2.0, metavar="NM",
        dest="confined_nm",
        help="An axis shorter than this is classified as 'confined'. "
             "Default: 2.0 nm.",
    )
    classification.add_argument(
        "--bulk-nm", type=float, default=8.0, metavar="NM", dest="bulk_nm",
        help="An axis longer than this is classified as 'bulk-like'. "
             "Default: 8.0 nm.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output", type=str, default=None, metavar="PATH",
        help="If given, save the full numerical DOS curve to this CSV "
             "file, with columns: energy_eV, dos, occupied_dos "
             "(dos * Fermi-Dirac occupation at --temperature). If not "
             "given, only the summary is printed to stdout.",
    )
    output.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary on stdout (useful "
             "when only --output is wanted, e.g. in scripts).",
    )

    return parser


def _run(args: argparse.Namespace) -> int:
    try:
        box = QuantumBox(
            lx_nm=args.lx, ly_nm=args.ly, lz_nm=args.lz,
            effective_mass=args.mass,
        )
        thresholds = ConfinementThresholds(
            confined_nm=args.confined_nm, bulk_nm=args.bulk_nm,
        )
        result = calculate_dos(
            box,
            sigma_eV=args.sigma,
            electron_density_m3=args.density,
            e_max_eV=args.e_max,
            n_points=args.n_points,
        )
    except ValueError as exc:
        print(f"quantum-dos: error: {exc}", file=sys.stderr)
        return 1

    regime = classify_regime(box, thresholds)
    occupation = fermi_dirac(result.energy_eV, result.fermi_energy_eV, args.temperature)
    occupied_dos = result.dos * occupation

    idx = (abs(result.energy_eV - result.fermi_energy_eV)).argmin()
    dos_at_ef = float(result.dos[idx])

    if not args.quiet:
        print(f"Box            : {box.lx_nm:g} x {box.ly_nm:g} x {box.lz_nm:g} nm "
              f"(volume {box.volume_nm3:g} nm^3)")
        print(f"Effective mass : {box.effective_mass:g} m_e")
        print(f"Sigma          : {args.sigma:g} eV")
        print(f"Density        : {args.density:.3g} m^-3")
        print(f"Regime         : {regime}")
        print(f"States found   : {result.n_states_enumerated}")
        print(f"Fermi energy   : {result.fermi_energy_eV:.4f} eV")
        print(f"DOS @ Ef       : {dos_at_ef:.4f} (arb. units)")
        print(f"Temperature    : {args.temperature:g} K "
              f"(used for occupied-DOS output only)")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["energy_eV", "dos", "occupied_dos"])
            for e, d, od in zip(result.energy_eV, result.dos, occupied_dos):
                writer.writerow([f"{e:.6f}", f"{d:.6f}", f"{od:.6f}"])
        if not args.quiet:
            print(f"Saved DOS curve to: {args.output}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
