import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "test_ini_copy.c"


def test_ini_copy_is_independent(tmp_path):
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        pytest.skip("No C compiler available")

    executable = tmp_path / ("test_ini_copy.exe" if os.name == "nt" else "test_ini_copy")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-I",
            str(ROOT / "src"),
            str(SOURCE),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
