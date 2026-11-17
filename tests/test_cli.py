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
        """--help must print usage and exit successfully.

        WHY: a command-line tool must offer discoverable help, and the
        convention is that --help prints usage to stdout and exits with
        code 0 (success). This checks both that the help text is produced
        and that the exit code signals success, which is the single
        concept "help works as a user expects".

        GIVEN the parser and the --help flag,
        WHEN the arguments are parsed,
        THEN it exits with code 0 and the help text mentions the program
        name.
        """
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "quantum-dos" in captured.out

    def test_missing_required_argument_exits_with_code_two(self, capsys):
        """Missing required arguments must exit with the usage-error code.

        WHY: if the user omits required parameters the tool must fail
        clearly rather than run with undefined inputs. argparse's
        convention for a usage error is exit code 2, so this pins that
        the required arguments are actually enforced.

        GIVEN only --lx (the other required arguments omitted),
        WHEN the arguments are parsed,
        THEN it exits with code 2.
        """
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--lx", "5.0"])  # missing ly, lz, mass, sigma, density
        assert exc_info.value.code == 2

    def test_default_temperature_and_grid_values(self):
        """Optional parameters must fall back to their documented defaults.

        WHY: the CLI documents default values for the parameters a user
        may omit (temperature, energy range, grid size, and the
        confinement thresholds). This checks that those defaults are
        actually applied when the options are not given -- one concept,
        "the documented defaults hold", verified across the optional
        parameters.

        GIVEN only the required arguments,
        WHEN they are parsed,
        THEN temperature, e_max, n_points, confined_nm, and bulk_nm take
        their documented default values.
        """
        parser = build_parser()
        args = parser.parse_args(BASE_ARGS)
        assert args.temperature == 300.0
        assert args.e_max == 12.0
        assert args.n_points == 500
        assert args.confined_nm == 2.0
        assert args.bulk_nm == 8.0


class TestMainSummaryOutput:
    def test_successful_run_returns_exit_code_zero(self, capsys):
        """A valid run must return the success exit code.

        WHY: the process exit code is how scripts and pipelines know
        whether the tool succeeded, so a valid invocation must return 0.
        This checks the success path's exit code on its own.

        GIVEN a complete, valid set of arguments,
        WHEN main is run,
        THEN it returns exit code 0.
        """
        exit_code = main(BASE_ARGS)
        assert exit_code == 0

    def test_summary_reports_fermi_energy_and_regime(self, capsys):
        """A normal run must print the key results in its summary.

        WHY: the point of the CLI is to report the computed physics, so
        the printed summary must contain the headline results a user
        looks for -- the Fermi energy, the dimensional regime, and the
        number of states found. This checks the summary actually carries
        that information (one concept: the summary is informative).

        GIVEN a valid run,
        WHEN main prints its summary,
        THEN the output mentions the Fermi energy, the regime, and the
        states found.
        """
        main(BASE_ARGS)
        captured = capsys.readouterr()
        assert "Fermi energy" in captured.out
        assert "Regime" in captured.out
        assert "States found" in captured.out

    def test_quiet_flag_suppresses_summary(self, capsys):
        """The --quiet flag must suppress all summary output.

        WHY: --quiet exists so the tool can be used in scripts without
        cluttering stdout, so when it is set the summary must not be
        printed at all. This checks the flag genuinely silences the
        output.

        GIVEN a valid run with --quiet,
        WHEN main is executed,
        THEN nothing is written to stdout.
        """
        main(BASE_ARGS + ["--quiet"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_invalid_geometry_prints_error_and_returns_exit_code_one(self, capsys):
        """Invalid physics input must report an error and fail cleanly.

        WHY: when the user supplies an invalid parameter (here a negative
        box dimension) the tool must fail with a non-zero exit code and
        an error message, rather than crash with a traceback or exit 0.
        Exit code 1 signals a runtime (not usage) error, and a message on
        stderr tells the user what went wrong.

        GIVEN a negative Lx,
        WHEN main is run,
        THEN it returns exit code 1 and writes an error to stderr.
        """
        exit_code = main(["--lx", "-5.0", "--ly", "5.0", "--lz", "5.0",
                           "--mass", "1.0", "--sigma", "0.05", "--density", "5.86e28"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_electron_count_exceeding_enumerated_states_returns_exit_code_one(self, capsys):
        """The 'not enough states' error must surface through the CLI.

        WHY: the physics layer raises when a filling needs more states
        than were enumerated; the CLI must catch that and report it as a
        clean failure (exit code 1 with the explanatory message on
        stderr) rather than let the exception escape. This is the
        CLI-level counterpart of the physics regression test.

        GIVEN a large, light-mass box whose filling exceeds the
        enumerated states,
        WHEN main is run,
        THEN it returns exit code 1 and stderr explains that not enough
        states were enumerated.
        """
        exit_code = main(["--lx", "20", "--ly", "20", "--lz", "20",
                           "--mass", "0.1", "--sigma", "0.05", "--density", "5.86e28"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Not enough enumerated states" in captured.err


class TestCSVOutput:
    def test_output_file_has_expected_header_and_row_count(self, tmp_path):
        """The --output CSV must have the documented header and row count.

        WHY: the CSV is the machine-readable result users will load
        elsewhere, so its structure is a contract: a fixed header row and
        exactly one data row per energy-grid point. This checks that
        structure (one concept: the file layout is as documented) for a
        50-point grid.

        GIVEN a valid run writing a CSV with n_points = 50,
        WHEN the file is read back,
        THEN its header is the documented column names and it contains 50
        data rows.
        """
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

    def test_output_columns_are_all_finite(self, tmp_path):
        """Every numeric value written to the CSV must be finite.

        WHY: a CSV containing NaN or inf would be unusable downstream, so
        all three columns (energy, DOS, occupied DOS) must be finite
        numbers. This is checked on its own so a failure means a
        non-finite value and nothing else.

        GIVEN a valid run writing a CSV,
        WHEN the three numeric columns are read back,
        THEN every value in each column is finite.
        """
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

    def test_occupied_dos_never_exceeds_total_dos(self, tmp_path):
        """The occupied DOS must never exceed the total DOS.

        WHY: the occupied DOS is the total DOS multiplied by the
        Fermi-Dirac occupation, which lies in [0, 1], so by construction
        the occupied column can never be larger than the total DOS
        column. This physical invariant is checked separately from the
        finiteness check so a failure isolates exactly this relationship.
        A small tolerance absorbs floating-point round-off at points
        where occupation is essentially 1.

        GIVEN a valid run writing a CSV,
        WHEN the DOS and occupied-DOS columns are read back,
        THEN every occupied value is <= the corresponding total value
        (within floating-point tolerance).
        """
        out_file = tmp_path / "dos.csv"
        main(BASE_ARGS + ["--output", str(out_file), "--n-points", "20"])

        with open(out_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        dos = np.array([float(r["dos"]) for r in rows])
        occupied = np.array([float(r["occupied_dos"]) for r in rows])

        assert np.all(occupied <= dos + 1e-9)

    def test_custom_e_max_is_reflected_in_energy_grid_range(self, tmp_path):
        """A custom --e-max must bound the energy column in the CSV.

        WHY: --e-max controls the top of the energy range, so the energy
        column written to the CSV must not exceed it. This confirms the
        option is applied to the grid the output is built from, rather
        than ignored in favour of the default.

        GIVEN a valid run with --e-max 6.0,
        WHEN the energy column is read back,
        THEN its maximum does not exceed 6.0 eV (within tolerance).
        """
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
        """The installed console script must run end to end.

        WHY: the in-process tests exercise main() directly, but they do
        not prove the packaged `quantum-dos` entry point is wired up
        correctly. This runs the actual installed executable as a
        subprocess and checks it succeeds and prints a result, confirming
        the console-script installation works as a real user would invoke
        it. It is skipped when the script is not on PATH so the suite does
        not fail merely because of how the package was installed.

        GIVEN the installed quantum-dos executable (or a skip if absent),
        WHEN it is run with a valid set of arguments,
        THEN it exits successfully and prints the Fermi energy.
        """
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
