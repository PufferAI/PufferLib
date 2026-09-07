#!/usr/bin/env python3
"""Fail-closed dependency and asset checks for Fight Caves builds."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "ocean" / "fight_caves"
SETUP_DATA_PATH = ENV_ROOT / "scripts" / "setup_data.py"


def command_name(value: str | None, default: str) -> str:
    words = shlex.split(value or default)
    return words[0] if words else default


def command_words(value: str | None, default: str) -> list[str]:
    words = shlex.split(value or default)
    return words or [default]


def require_command(errors: list[str], name: str, purpose: str) -> None:
    if shutil.which(name) is None:
        errors.append(f"required command '{name}' is unavailable ({purpose})")


def require_python_module(errors: list[str], name: str, purpose: str) -> None:
    if importlib.util.find_spec(name) is None:
        errors.append(
            f"required Python module '{name}' is unavailable ({purpose}); "
            f"install the repository dependencies first"
        )


def load_setup_data():
    spec = importlib.util.spec_from_file_location("fight_caves_setup_data", SETUP_DATA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import asset verifier: {SETUP_DATA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_assets(errors: list[str], names: tuple[str, ...]) -> None:
    try:
        setup_data = load_setup_data()
        manifest = setup_data.load_manifest(setup_data.DEFAULT_MANIFEST)
        for name in names:
            bundle = manifest["bundles"].get(name)
            if not isinstance(bundle, dict):
                errors.append(f"asset manifest has no {name} bundle")
                continue
            failures = setup_data.verify_tree(
                setup_data.RESOURCE_ROOT, bundle, exact=False
            )
            if failures:
                errors.append(f"{name} asset bundle is invalid: {failures[0]}")
    except Exception as exc:
        errors.append(f"could not verify Fight Caves assets: {exc}")


def check_linux_viewer_link(
    errors: list[str], compiler_value: str | None
) -> None:
    if sys.platform != "linux":
        return
    x11_header = Path("/usr/include/X11/Xlib.h")
    if not x11_header.is_file():
        errors.append(
            "X11 development headers are unavailable; on Ubuntu install "
            "libx11-dev libxrandr-dev libxi-dev libxcursor-dev libxinerama-dev"
        )
        return
    compiler = command_words(compiler_value, "clang")
    source = "int main(void) { return 0; }\n"
    try:
        with tempfile.TemporaryDirectory(prefix="fight-caves-preflight-") as value:
            root = Path(value)
            result = subprocess.run(
                [*compiler, "-x", "c", "-", "-o", str(root / "link-test"),
                 "-lGL", "-lX11"],
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
    except OSError as exc:
        errors.append(
            f"could not run viewer link check with {' '.join(compiler)}: {exc}"
        )
        return
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        errors.append(
            "OpenGL/X11 development libraries cannot be linked; on Ubuntu "
            "install libgl1-mesa-dev and the X11 development packages"
            f"{suffix}"
        )


def check_openmp(errors: list[str], compiler_value: str | None, language: str) -> None:
    compiler = command_words(compiler_value, "clang" if language == "c" else "g++")
    suffix = ".c" if language == "c" else ".cpp"
    source = "#include <omp.h>\nint main(void) { return omp_get_max_threads() < 1; }\n"
    try:
        with tempfile.TemporaryDirectory(prefix="fight-caves-openmp-") as value:
            root = Path(value)
            source_path = root / f"test{suffix}"
            source_path.write_text(source, encoding="utf-8")
            arguments = [
                *compiler, str(source_path), "-fopenmp", "-o", str(root / "test")
            ]
            if language == "c++":
                omp_library = os.environ.get(
                    "PUFFER_OMP_LIB", "-lomp5" if sys.platform == "linux" else "-lomp"
                )
                arguments.extend(shlex.split(omp_library))
            result = subprocess.run(
                arguments,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
    except OSError as exc:
        errors.append(f"could not run OpenMP {language.upper()} check: {exc}")
        return
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        last_line = f": {detail[-1]}" if detail else ""
        errors.append(
            f"{language.upper()} compiler cannot build and link OpenMP; on Ubuntu "
            f"install libomp-dev or select an OpenMP-capable compiler with "
            f"{'CC' if language == 'c' else 'CXX'}{last_line}"
        )


def check_graphical_display(errors: list[str]) -> None:
    if sys.platform != "linux":
        return
    display = os.environ.get("DISPLAY")
    if not display:
        errors.append(
            "no graphical DISPLAY is configured; run under xvfb-run for "
            "headless validation or launch from a graphical session"
        )
        return
    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        errors.append(
            "'xdpyinfo' is required to validate the X11 display; on Ubuntu "
            "install x11-utils"
        )
        return
    result = subprocess.run(
        [xdpyinfo, "-display", display],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        errors.append(
            f"X11 display {display!r} is not accessible; run under xvfb-run "
            "for headless validation or fix the display authorization"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Fight Caves build dependencies and installed assets."
    )
    parser.add_argument(
        "--mode",
        choices=("core", "native", "cpu", "cuda", "viewer", "viewer-runtime", "web"),
        required=True,
        help="build path to validate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10 or newer is required; found {sys.version.split()[0]}"
        )

    compiler = command_name(os.environ.get("CC"), "clang")
    if args.mode != "viewer-runtime":
        require_command(errors, compiler, "C compilation")

    if args.mode in ("core", "native", "cpu", "cuda", "web"):
        verify_assets(errors, ("core",))
    elif args.mode in ("viewer", "viewer-runtime"):
        verify_assets(errors, ("core", "viewer"))

    if args.mode in ("native", "cpu", "cuda"):
        require_command(errors, "ar", "static library creation")
    if args.mode in ("cpu", "cuda"):
        cxx = command_name(os.environ.get("CXX"), "g++")
        require_command(errors, cxx, "C++ extension compilation")
        for module, purpose in (
            ("numpy", "Puffer observation buffers"),
            ("pybind11", "Puffer Python extension bindings"),
            ("torch", "Puffer policy execution"),
        ):
            require_python_module(errors, module, purpose)
        check_openmp(errors, os.environ.get("CXX"), "c++")
    if args.mode in ("native", "cpu", "cuda"):
        check_openmp(errors, os.environ.get("CC"), "c")
    if args.mode == "native" and sys.platform == "linux":
        check_linux_viewer_link(errors, os.environ.get("CC"))
    if args.mode == "cuda":
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        nvcc = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc"
        require_command(errors, nvcc, "CUDA backend compilation")
        require_command(errors, "nvidia-smi", "CUDA device validation")
    if args.mode == "viewer":
        require_command(errors, "cmake", "viewer configuration")
        check_linux_viewer_link(errors, os.environ.get("CC"))
    if args.mode == "viewer-runtime":
        check_graphical_display(errors)
    if args.mode == "web":
        require_command(errors, "emcc", "WebAssembly compilation")

    if errors:
        print("Fight Caves preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if any("asset bundle" in error or "verify Fight Caves assets" in error
               for error in errors):
            print(
                "Install assets with: bash ocean/fight_caves/scripts/setup-data.sh --all",
                file=sys.stderr,
            )
        return 1

    print(f"Fight Caves {args.mode} preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
