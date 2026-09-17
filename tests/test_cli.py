# -*- coding: utf-8 -*-
"""Tests for quantum_dos.cli.

These call `main()` directly (in-process) rather than spawning a
subprocess, so they exercise the same code path as the installed
`quantum-dos` console script without depending on it being on PATH.
"""

import csv

import numpy as np
import pytest

from quantum_dos.cli import build_parser, main


BASE_ARGS = [
    "--lx", "5.0", "--ly", "5.0", "--lz", "5.0",
    "--mass", "1.0", "--sigma", "0.05", "--density", "5.86e28",
]


class TestArgumentParsing:
    def test_help_exits_with_code_zero(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "quantum-dos" in captured.out

    def test_missing_required_argument_exits_with_code_two(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--lx", "5.0"])  # missing ly, lz, mass, sigma, density
        assert exc_info.value.code == 2

    def test_default_temperature_and_grid_values(self):
        parser = build_parser()
        args = parser.parse_args(BASE_ARGS)
        assert args.temperature == 300.0
        assert args.e_max == 12.0
        assert args.n_points == 500
        assert args.confined_nm == 2.0
        assert args.bulk_nm == 8.0


class TestMainSummaryOutput:
    def test_successful_run_returns_exit_code_zero(self, capsys):
        exit_code = main(BASE_ARGS)
        assert exit_code == 0

    def test_summary_reports_fermi_energy_and_regime(self, capsys):
        main(BASE_ARGS)
        captured = capsys.readouterr()
        assert "Fermi energy" in captured.out
        assert "Regime" in captured.out
        assert "States found" in captured.out

    def test_quiet_flag_suppresses_summary(self, capsys):
        main(BASE_ARGS + ["--quiet"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_invalid_geometry_prints_error_and_returns_exit_code_one(self, capsys):
        exit_code = main(["--lx", "-5.0", "--ly", "5.0", "--lz", "5.0",
                           "--mass", "1.0", "--sigma", "0.05", "--density", "5.86e28"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_electron_count_exceeding_enumerated_states_returns_exit_code_one(self, capsys):
        # Same reproduction case as the physics-layer regression test.
        exit_code = main(["--lx", "20", "--ly", "20", "--lz", "20",
                           "--mass", "0.1", "--sigma", "0.05", "--density", "5.86e28"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Not enough enumerated states" in captured.err


class TestCSVOutput:
    def test_output_file_contains_expected_header_and_row_count(self, tmp_path):
        out_file = tmp_path / "dos.csv"
        exit_code = main(BASE_ARGS + ["--output", str(out_file), "--n-points", "50"])
        assert exit_code == 0
        assert out_file.exists()

        with open(out_file, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert header == ["energy_eV", "dos", "occupied_dos"]
        assert len(rows) == 50

    def test_output_values_are_numeric_and_finite(self, tmp_path):
        out_file = tmp_path / "dos.csv"
        main(BASE_ARGS + ["--output", str(out_file), "--n-points", "20"])

        with open(out_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        energies = np.array([float(r["energy_eV"]) for r in rows])
        dos = np.array([float(r["dos"]) for r in rows])
        occupied = np.array([float(r["occupied_dos"]) for r in rows])

        assert np.all(np.isfinite(energies))
        assert np.all(np.isfinite(dos))
        assert np.all(np.isfinite(occupied))
        assert np.all(dos >= 0)
        # Occupied DOS can never exceed the total DOS (occupation in [0, 1]).
        assert np.all(occupied <= dos + 1e-9)

    def test_custom_e_max_is_reflected_in_energy_grid_range(self, tmp_path):
        out_file = tmp_path / "dos.csv"
        main(BASE_ARGS + ["--output", str(out_file), "--e-max", "6.0", "--n-points", "10"])

        with open(out_file, newline="") as f:
            reader = csv.DictReader(f)
            energies = [float(r["energy_eV"]) for r in reader]

        assert max(energies) <= 6.0 + 1e-9


class TestConsoleScriptSmoke:
    """End-to-end check that the installed `quantum-dos` entry point runs.

    Skipped automatically if the console script is not on PATH (e.g. the
    package was imported from source without being pip-installed), so the
    suite never hard-fails purely because of the install method.
    """

    def test_installed_console_script_runs_and_reports_fermi_energy(self):
        import shutil
        import subprocess

        executable = shutil.which("quantum-dos")
        if executable is None:
            pytest.skip("quantum-dos console script not on PATH (package not installed)")

        completed = subprocess.run(
            [executable, "--lx", "5", "--ly", "5", "--lz", "5",
             "--mass", "1.0", "--sigma", "0.05", "--density", "5.86e28"],
            capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0
        assert "Fermi energy" in completed.stdout

