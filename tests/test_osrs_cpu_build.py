import os
import shutil
import stat
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_osrs_cpu_build_selects_visual_entrypoint(tmp_path: Path) -> None:
    shutil.copy(REPOSITORY_ROOT / "build.sh", tmp_path / "build.sh")

    source = tmp_path / "ocean/osrs_inferno/osrs_inferno.c"
    source.parent.mkdir(parents=True)
    source.write_text("")

    scripts = tmp_path / "ocean/osrs/scripts"
    scripts.mkdir(parents=True)
    (scripts / "osrs_asset_manifest.py").write_text("")
    write_executable(scripts / "setup-data.sh", "#!/bin/bash\n")
    (tmp_path / "ocean/osrs/asset_manifest.json").write_text("{}")

    for raylib in ("raylib-5.5_linux_amd64", "raylib-5.5_macos"):
        (tmp_path / raylib).mkdir()

    tools = tmp_path / "tools"
    tools.mkdir()
    write_executable(tools / "brew", "#!/bin/bash\nprintf '%s\\n' \"$PWD/libomp\"\n")
    compiler_arguments = tmp_path / "compiler-arguments"
    write_executable(
        tools / "capture-compiler",
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$COMPILER_ARGUMENTS\"\n",
    )

    environment = os.environ | {
        "CC": str(tools / "capture-compiler"),
        "COMPILER_ARGUMENTS": str(compiler_arguments),
        "PATH": f"{tools}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", "build.sh", "osrs_inferno", "--cpu"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    arguments = compiler_arguments.read_text().splitlines()
    # build.sh resolves sources from its own directory (symlinks resolved) and
    # writes the binary to the directory it was run from.
    root = tmp_path.resolve()
    assert str(root / "ocean/osrs_inferno/osrs_inferno.c") in arguments
    assert "src/puffercpu.c" not in arguments
    assert arguments[arguments.index("-o") + 1] == str(root / "osrs_inferno")
