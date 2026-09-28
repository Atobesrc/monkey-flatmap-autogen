import subprocess
import sys

import pytest

from mfa.cli import build_parser

SUBCOMMANDS = ["flatten", "apply", "import-subject", "tune-joint", "reference", "compare", "overlay-parcels",
               "atlas-annot", "qc"]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help(cmd):
    with pytest.raises(SystemExit) as ex:
        build_parser().parse_args([cmd, "--help"])
    assert ex.value.code == 0


def test_module_entry_point_help():
    out = subprocess.run([sys.executable, "-m", "mfa.cli", "--help"], capture_output=True, text=True, check=True)
    for cmd in SUBCOMMANDS:
        assert cmd in out.stdout


def test_print_recipe(tmp_path, synthetic_subject):
    subjects_dir, subject = synthetic_subject
    out = subprocess.run([sys.executable, "-m", "mfa.cli", "flatten", "--subjects-dir", subjects_dir, "--subject",
                          subject, "--name", "t", "--slit-width", "lh=auto,rh=2", "--print-recipe"],
                         capture_output=True, text=True, check=True)
    assert "temporal_src=nearest_" in out.stdout
    assert "slit_width auto" in out.stdout
    assert "rh: slit_width=2" in out.stdout


def test_flatten_refuses_reserved_name(synthetic_subject):
    subjects_dir, subject = synthetic_subject
    r = subprocess.run([sys.executable, "-m", "mfa.cli", "flatten", "--subjects-dir", subjects_dir, "--subject",
                        subject, "--name", "flatten", "--patch-only"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "reserved" in (r.stdout + r.stderr)
