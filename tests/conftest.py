import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from synthetic_subject import make_subject  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_subject(tmp_path_factory) -> tuple[str, str]:
    """``(subjects_dir, subject)`` of a small synthetic FreeSurfer-style subject."""
    root = tmp_path_factory.mktemp("subjects")
    make_subject(str(root), "sub-synth", subdivisions=4)
    return str(root), "sub-synth"


def has_igl() -> bool:
    try:
        import igl  # noqa: F401
    except ImportError:
        return False
    return True


needs_igl = pytest.mark.skipif(not has_igl(), reason="libigl (pip install libigl) is required")
