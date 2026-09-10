#!/usr/bin/env python3
"""Fight Caves asset installation, build checks, playable launch and checkpoint replay."""

from __future__ import annotations

import argparse
import configparser
import ctypes
from dataclasses import dataclass
import glob
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Asset installation

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = REPO_ROOT / "resources" / "fight_caves"
DEFAULT_MANIFEST = RESOURCE_ROOT / "asset_manifest.json"
BUFFER_SIZE = 1024 * 1024


class AssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetError(f"asset manifest not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"could not read asset manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise AssetError("unsupported Fight Caves asset manifest schema")
    if not isinstance(manifest.get("bundles"), dict):
        raise AssetError("asset manifest has no bundle definitions")
    return manifest


def checked_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise AssetError(f"unsafe path in asset manifest or archive: {value!r}")
    return path


def expected_files(bundle: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for entry in bundle.get("files", []):
        relative = checked_relative_path(entry.get("path", "")).as_posix()
        if relative in result:
            raise AssetError(f"duplicate file in asset manifest: {relative}")
        if not isinstance(entry.get("size_bytes"), int) or not entry.get("sha256"):
            raise AssetError(f"incomplete asset metadata for {relative}")
        result[relative] = entry
    if not result:
        raise AssetError("asset bundle contains no files")
    return result


def verify_tree(root: Path, bundle: dict, *, exact: bool) -> list[str]:
    expected = expected_files(bundle)
    errors: list[str] = []
    for relative, entry in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            errors.append(
                f"wrong size for {relative}: expected {entry['size_bytes']}, "
                f"got {actual_size}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            errors.append(f"checksum mismatch for {relative}")

    if exact:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        for relative in sorted(actual - set(expected)):
            errors.append(f"unexpected file in bundle: {relative}")
    return errors


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "PufferLib-Fight-Caves-assets/1"})
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, BUFFER_SIZE)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AssetError(f"download failed for {url}: {exc}") from exc


def extract_checked(archive: Path, destination: Path, bundle: dict) -> None:
    expected = set(expected_files(bundle))
    archived_files: set[str] = set()

    try:
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                relative = checked_relative_path(member.name)
                relative_name = relative.as_posix()
                if member.isdir():
                    continue
                if not member.isfile():
                    raise AssetError(
                        f"unsupported non-file entry in asset archive: {relative_name}"
                    )
                if relative_name not in expected:
                    raise AssetError(
                        f"unexpected file in asset archive: {relative_name}"
                    )
                if relative_name in archived_files:
                    raise AssetError(f"duplicate file in asset archive: {relative_name}")

                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = source.extractfile(member)
                if extracted is None:
                    raise AssetError(f"could not extract {relative_name}")
                with extracted, target.open("wb") as out:
                    shutil.copyfileobj(extracted, out, BUFFER_SIZE)
                target.chmod(0o644)
                archived_files.add(relative_name)
    except (OSError, tarfile.TarError) as exc:
        raise AssetError(f"could not extract {archive}: {exc}") from exc

    missing = expected - archived_files
    if missing:
        raise AssetError(f"asset archive is missing {sorted(missing)[0]}")


def replace_bundle(staged_root: Path, bundle: dict) -> None:
    prefix = checked_relative_path(bundle.get("install_prefix", ""))
    if len(prefix.parts) != 1:
        raise AssetError("bundle install prefix must be one directory name")

    staged = staged_root.joinpath(*prefix.parts)
    destination = RESOURCE_ROOT.joinpath(*prefix.parts)
    incoming = RESOURCE_ROOT / f".{prefix.name}.installing"
    backup = RESOURCE_ROOT / f".{prefix.name}.backup"

    if not staged.is_dir():
        raise AssetError(f"archive did not contain expected {prefix}/ directory")
    if incoming.exists() or backup.exists():
        raise AssetError(
            f"stale installer directory found under {RESOURCE_ROOT}; "
            "remove it and retry"
        )

    shutil.copytree(staged, incoming)
    try:
        if destination.exists():
            destination.rename(backup)
        incoming.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if incoming.exists():
            shutil.rmtree(incoming)
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def install_bundle(name: str, bundle: dict, *, force: bool) -> None:
    if not force:
        current_errors = verify_tree(RESOURCE_ROOT, bundle, exact=False)
        if not current_errors:
            print(f"Fight Caves {name} assets are already installed and verified.")
            return

    archive_name = bundle.get("archive", "")
    archive_hash = bundle.get("sha256", "")
    archive_size = bundle.get("size_bytes")
    url = bundle.get("url", "")
    if not archive_name or not archive_hash or not url or not isinstance(archive_size, int):
        raise AssetError(f"incomplete archive metadata for {name}")

    RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fight-caves-assets-") as temp_value:
        temp = Path(temp_value)
        archive = temp / archive_name
        extracted = temp / "extracted"
        extracted.mkdir()
        download(url, archive)

        if archive.stat().st_size != archive_size:
            raise AssetError(
                f"wrong archive size for {archive_name}: expected {archive_size}, "
                f"got {archive.stat().st_size}"
            )
        if sha256_file(archive) != archive_hash:
            raise AssetError(f"checksum mismatch for downloaded {archive_name}")

        extract_checked(archive, extracted, bundle)
        staged_errors = verify_tree(extracted, bundle, exact=True)
        if staged_errors:
            raise AssetError(staged_errors[0])
        replace_bundle(extracted, bundle)

    installed_errors = verify_tree(RESOURCE_ROOT, bundle, exact=False)
    if installed_errors:
        raise AssetError(installed_errors[0])
    print(f"Installed and verified Fight Caves {name} assets.")


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install pinned Fight Caves runtime and viewer assets."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--core", action="store_true", help="select runtime maps")
    selection.add_argument("--viewer", action="store_true", help="select viewer assets")
    selection.add_argument("--all", action="store_true", help="select both bundles")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify installed files without downloading or changing them",
    )
    parser.add_argument(
        "--force", action="store_true", help="download and reinstall selected bundles"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def setup_main() -> int:
    args = setup_args()
    try:
        manifest = load_manifest(args.manifest)
        names = (
            ["core", "viewer"]
            if args.all or not (args.core or args.viewer)
            else []
        )
        if args.core:
            names = ["core"]
        elif args.viewer:
            names = ["viewer"]

        for name in names:
            bundle = manifest["bundles"].get(name)
            if not isinstance(bundle, dict):
                raise AssetError(f"asset manifest has no {name} bundle")
            if args.verify_only:
                errors = verify_tree(RESOURCE_ROOT, bundle, exact=False)
                if errors:
                    raise AssetError(f"{name}: {errors[0]}")
                print(f"Fight Caves {name} assets are installed and verified.")
            else:
                install_bundle(name, bundle, force=args.force)
    except (AssetError, KeyError, OSError) as exc:
        print(f"Fight Caves asset setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


# Release bundles

RUNTIME_FILES = (
    "fightcaves.collision",
    "fightcaves.movement",
    "fightcaves.los",
)


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str


def collect_runtime(source: Path) -> list[BundleFile]:
    result = []
    for name in RUNTIME_FILES:
        path = source / name
        if not path.is_file():
            raise SystemExit(f"missing runtime asset: {path}")
        result.append(BundleFile(path, f"runtime/{name}"))
    return result


def collect_viewer(source: Path) -> list[BundleFile]:
    result = [
        BundleFile(path, f"viewer/{path.relative_to(source).as_posix()}")
        for path in source.rglob("*")
        if path.is_file()
    ]
    if not result:
        raise SystemExit(f"no viewer assets found under {source}")
    return sorted(result, key=lambda entry: entry.archive_path)


def write_archive(destination: Path, files: list[BundleFile]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
            ) as archive:
                for entry in sorted(files, key=lambda value: value.archive_path):
                    info = archive.gettarinfo(str(entry.source), entry.archive_path)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    info.mtime = 0
                    with entry.source.open("rb") as source:
                        archive.addfile(info, source)
    temporary.replace(destination)


def bundle_manifest(
    archive: Path,
    files: list[BundleFile],
    repository: str,
    release_tag: str,
    install_prefix: str,
) -> dict:
    return {
        "archive": archive.name,
        "url": (
            f"https://github.com/{repository}/releases/download/"
            f"{release_tag}/{archive.name}"
        ),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "install_prefix": install_prefix,
        "files": [
            {
                "path": entry.archive_path,
                "size_bytes": entry.source.stat().st_size,
                "sha256": sha256_file(entry.source),
            }
            for entry in sorted(files, key=lambda value: value.archive_path)
        ],
    }


def bundle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--viewer-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-repository", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bundle-version", default="v2")
    return parser.parse_args()


def bundle_main() -> int:
    args = bundle_args()
    runtime_files = collect_runtime(args.runtime_source)
    viewer_files = collect_viewer(args.viewer_source)
    runtime_archive = (
        args.output_dir / f"fight-caves-runtime-assets-{args.bundle_version}.tar.gz"
    )
    viewer_archive = (
        args.output_dir / f"fight-caves-viewer-assets-{args.bundle_version}.tar.gz"
    )

    write_archive(runtime_archive, runtime_files)
    write_archive(viewer_archive, viewer_files)
    manifest = {
        "schema_version": 1,
        "release_tag": args.release_tag,
        "source": {
            "repository": f"https://github.com/{args.release_repository}",
            "revision": args.source_revision,
        },
        "bundles": {
            "core": bundle_manifest(
                runtime_archive,
                runtime_files,
                args.release_repository,
                args.release_tag,
                "runtime",
            ),
            "viewer": bundle_manifest(
                viewer_archive,
                viewer_files,
                args.release_repository,
                args.release_tag,
                "viewer",
            ),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(runtime_archive)
    print(viewer_archive)
    print(args.manifest)
    return 0


# Dependency preflight

ENV_ROOT = REPO_ROOT / "ocean" / "fight_caves"


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


def verify_assets(errors: list[str], names: tuple[str, ...]) -> None:
    try:
        manifest = load_manifest(DEFAULT_MANIFEST)
        for name in names:
            bundle = manifest["bundles"].get(name)
            if not isinstance(bundle, dict):
                errors.append(f"asset manifest has no {name} bundle")
                continue
            failures = verify_tree(
                RESOURCE_ROOT, bundle, exact=False
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
                # Match the unmodified Puffer 4.0 build.sh link flags.
                arguments.append("-lomp5" if sys.platform == "linux" else "-lomp")
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


def preflight_args() -> argparse.Namespace:
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


def preflight_main() -> int:
    return run_preflight(preflight_args().mode)


def run_preflight(mode: str) -> int:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10 or newer is required; found {sys.version.split()[0]}"
        )

    compiler = command_name(os.environ.get("CC"), "clang")
    if mode != "viewer-runtime":
        require_command(errors, compiler, "C compilation")

    if mode in ("core", "native", "cpu", "cuda", "web"):
        verify_assets(errors, ("core",))
    elif mode in ("viewer", "viewer-runtime"):
        verify_assets(errors, ("core", "viewer"))

    if mode in ("native", "cpu", "cuda"):
        require_command(errors, "ar", "static library creation")
    if mode in ("cpu", "cuda"):
        require_command(errors, "python", "Puffer build.sh; activate your Python environment")
        cxx = command_name(os.environ.get("CXX"), "g++")
        require_command(errors, cxx, "C++ extension compilation")
        for module, purpose in (
            ("numpy", "Puffer observation buffers"),
            ("pybind11", "Puffer Python extension bindings"),
            ("torch", "Puffer policy execution"),
        ):
            require_python_module(errors, module, purpose)
        check_openmp(errors, os.environ.get("CXX"), "c++")
    if mode in ("native", "cpu", "cuda"):
        check_openmp(errors, os.environ.get("CC"), "c")
    if mode == "native" and sys.platform == "linux":
        check_linux_viewer_link(errors, os.environ.get("CC"))
    if mode == "cuda":
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        nvcc = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc"
        require_command(errors, nvcc, "CUDA backend compilation")
        require_command(errors, "nvidia-smi", "CUDA device validation")
    if mode == "viewer":
        require_command(errors, "cmake", "viewer configuration")
        check_linux_viewer_link(errors, os.environ.get("CC"))
    if mode == "viewer-runtime":
        check_graphical_display(errors)
    if mode == "web":
        require_command(errors, "emcc", "WebAssembly compilation")

    if errors:
        print("Fight Caves preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if any("asset bundle" in error or "verify Fight Caves assets" in error
               for error in errors):
            print(
                "Install assets with: python3 ocean/fight_caves/tools.py setup --all",
                file=sys.stderr,
            )
        return 1

    print(f"Fight Caves {mode} preflight passed.")
    return 0


# Optional viewer build; the shared Puffer build.sh remains unmodified.

RAYLIB_FILES = ("include/raylib.h", "include/raymath.h", "include/rlgl.h",
                "lib/libraylib.a")


def require_raylib(root: Path) -> Path:
    missing = [name for name in RAYLIB_FILES if not (root / name).is_file()]
    if missing:
        raise AssetError(f"Raylib is incomplete at {root}: missing {', '.join(missing)}. "
                         "Supply a complete installation with --raylib-root.")
    return root


def viewer_raylib(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return require_raylib(explicit_root.expanduser().resolve())
    if sys.platform == "linux" and os.uname().machine in ("x86_64", "amd64"):
        name = "raylib-5.5_linux_amd64"
    elif sys.platform == "darwin":
        name = "raylib-5.5_macos"
    else:
        raise AssetError("No bundled Raylib 5.5 for this platform. "
                         "Supply a compatible build with --raylib-root.")

    # Reuse Puffer's download when available; otherwise keep this optional
    # dependency under build/, without creating a partial shared installation.
    shared = REPO_ROOT / name
    root = REPO_ROOT / "build" / name
    for existing in (shared, root):
        if existing.exists():
            return require_raylib(existing)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fight-caves-raylib-", dir=root.parent) as value:
        staging = Path(value)
        archive_path = staging / f"{name}.tar.gz"
        download(f"https://github.com/raysan5/raylib/releases/download/5.5/{name}.tar.gz",
                 archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            # Copy only the exact headers/static library used by this viewer.
            # No archive paths or links are ever extracted to the filesystem.
            for relative in RAYLIB_FILES:
                member = archive.getmember(f"{name}/{relative}")
                if not member.isfile():
                    raise AssetError(f"Raylib archive contains a non-file: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise AssetError(f"Raylib archive cannot read {member.name}")
                target = staging / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        require_raylib(staging / name).rename(root)
    return root


def build_viewer_main() -> int:
    parser = argparse.ArgumentParser(description="Build the optional Fight Caves viewer.")
    parser.add_argument("--raylib-root", type=Path,
                        help="use an existing Raylib installation instead of downloading 5.5")
    args = parser.parse_args()
    if run_preflight("viewer") != 0:
        return 1
    try:
        raylib = viewer_raylib(args.raylib_root)
        build = REPO_ROOT / "build" / "fight_caves-viewer"
        subprocess.run(["cmake", "-S", str(ENV_ROOT), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release", f"-DRAYLIB_ROOT={raylib}"],
                       cwd=REPO_ROOT, check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel"],
                       cwd=REPO_ROOT, check=True)
    except (AssetError, OSError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print(f"Fight Caves viewer build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Built: {build / 'fc_viewer'}")
    return 0


# Checkpoint contract

class ContractError(RuntimeError):
    pass


REQUIRED_FIELDS = (
    "contract_dump_schema_version",
    "policy_obs_size",
    "puffer_obs_size",
    "puffer_action_dims",
    "puffer_mask_size",
    "observation_version",
    "action_version",
    "reward_version",
    "prayer_timing_version",
    "state_hash_version",
    "active_loadout",
)


def contract_identity(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_compiled_contract(backend_path: str | Path) -> dict[str, Any]:
    backend = Path(backend_path).resolve()
    if not backend.is_file():
        raise ContractError(f"compiled backend is unavailable: {backend}")
    try:
        library = ctypes.CDLL(str(backend))
        symbol = library.fc_training_contract_json
    except (OSError, AttributeError) as exc:
        raise ContractError(
            f"compiled backend does not export the Fight Caves contract: {backend}"
        ) from exc
    symbol.argtypes = []
    symbol.restype = ctypes.c_char_p
    raw = symbol()
    if raw is None:
        raise ContractError("compiled Fight Caves contract returned null")
    try:
        contract = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("compiled Fight Caves contract is invalid") from exc
    if not isinstance(contract, dict):
        raise ContractError("compiled Fight Caves contract is not an object")
    return contract


def validate_compiled_contract(
    contract: dict[str, Any], expected_active_loadout: str | None = None
) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in contract]
    if missing:
        raise ContractError(f"compiled contract omits {missing[0]}")
    if contract["contract_dump_schema_version"] != 1:
        raise ContractError("unsupported compiled contract schema")

    policy_obs_size = contract["policy_obs_size"]
    puffer_obs_size = contract["puffer_obs_size"]
    mask_size = contract["puffer_mask_size"]
    action_dims = contract["puffer_action_dims"]
    if not all(isinstance(value, int) and value > 0 for value in (
        policy_obs_size, puffer_obs_size, mask_size
    )):
        raise ContractError("compiled contract contains invalid observation sizes")
    if (
        not isinstance(action_dims, list)
        or not action_dims
        or not all(isinstance(value, int) and value > 0 for value in action_dims)
    ):
        raise ContractError("compiled contract contains invalid action dimensions")
    if sum(action_dims) != mask_size:
        raise ContractError("compiled action dimensions do not match mask size")
    if policy_obs_size + mask_size != puffer_obs_size:
        raise ContractError("compiled policy observation and mask sizes do not add up")
    if expected_active_loadout is not None:
        actual = contract["active_loadout"]
        if actual != expected_active_loadout:
            raise ContractError(
                "compiled active loadout mismatch: "
                f"expected={expected_active_loadout!r}, actual={actual!r}"
            )


def merged_config(default_path: Path, selected_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    loaded = parser.read([default_path, selected_path], encoding="utf-8")
    if str(default_path) not in loaded or str(selected_path) not in loaded:
        raise ContractError(
            f"Puffer configuration is unavailable: {default_path}, {selected_path}"
        )
    return parser


def validate_config_contract(
    contract: dict[str, Any], default_path: Path, selected_path: Path
) -> None:
    parser = merged_config(default_path, selected_path)
    if not parser.has_section("run"):
        raise ContractError(f"Fight Caves config has no [run] section: {selected_path}")
    for field in ("observation_version", "action_version", "reward_version"):
        if not parser.has_option("run", field):
            raise ContractError(f"Fight Caves config omits [run].{field}")
        configured = parser.get("run", field).strip().strip("'\"")
        if configured != contract[field]:
            raise ContractError(
                f"Fight Caves config/compiled contract mismatch for {field}: "
                f"configured={configured!r}, compiled={contract[field]!r}"
            )


def build_verified_preflight(
    backend_path: str | Path,
    selected_config: str | Path,
    default_config: str | Path,
    active_loadout: str,
) -> dict[str, Any]:
    contract = load_compiled_contract(backend_path)
    validate_compiled_contract(contract, active_loadout)
    validate_config_contract(
        contract, Path(default_config).resolve(), Path(selected_config).resolve()
    )
    return {
        "contract": contract,
        "contract_identity": contract_identity(contract),
        "backend_path": str(Path(backend_path).resolve()),
        "config_path": str(Path(selected_config).resolve()),
    }


def expected_checkpoint_parameter_bytes(
    contract: dict[str, Any],
    selected_config: str | Path,
    default_config: str | Path,
) -> int:
    parser = merged_config(
        Path(default_config).resolve(), Path(selected_config).resolve()
    )
    try:
        hidden_size = parser.getint("policy", "hidden_size")
        num_layers = parser.getint("policy", "num_layers")
        network_name = parser.get("torch", "network").strip().strip("'\"")
    except (configparser.Error, ValueError) as exc:
        raise ContractError(f"cannot read checkpoint policy topology: {exc}") from exc
    if network_name != "MinGRU":
        raise ContractError(
            f"unsupported raw checkpoint network for replay: {network_name!r}"
        )
    if hidden_size <= 0 or num_layers <= 0:
        raise ContractError("checkpoint policy topology must be positive")

    parameter_floats = (
        contract["puffer_obs_size"] * hidden_size
        + (sum(contract["puffer_action_dims"]) + 1) * hidden_size
        + num_layers * 3 * hidden_size * hidden_size
    )
    return parameter_floats * 4


def find_checkpoint_marker(checkpoint: Path, checkpoint_root: Path) -> Path | None:
    adjacent = checkpoint.with_name(f"{checkpoint.name}.contract.json")
    if adjacent.is_file():
        return adjacent
    if checkpoint_root not in checkpoint.parents:
        external_root = next(
            (parent for parent in checkpoint.parents if parent.name == "checkpoints"),
            None,
        )
        if external_root is None:
            return None
        checkpoint_root = external_root
    current = checkpoint.parent
    while current == checkpoint_root or checkpoint_root in current.parents:
        marker = current / "contract.json"
        if marker.is_file():
            return marker
        if current == checkpoint_root:
            break
        current = current.parent
    return None


def validate_checkpoint_marker(marker: Path, preflight: dict[str, Any]) -> None:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid checkpoint contract sidecar: {marker}") from exc
    actual = payload.get("contract")
    expected = preflight["contract"]
    # v5 only adds manually controlled inventory/equipment to the state hash.
    # Policy weights do not serialize that state; allow only v4 -> v5 when
    # every other contract field is identical, just as in the v38 evaluator.
    if isinstance(actual, dict):
        actual = dict(actual)
        if actual.get("state_hash_version") == 4 and expected.get("state_hash_version") == 5:
            actual["state_hash_version"] = 5
    if actual != expected:
        raise ContractError(
            f"checkpoint contract does not match compiled Fight Caves: {marker}"
        )


def checkpoint_format(checkpoint: Path, expected_raw_bytes: int) -> str | None:
    if not checkpoint.is_file():
        return None
    if checkpoint.stat().st_size == expected_raw_bytes:
        return "raw"
    try:
        with checkpoint.open("rb") as handle:
            if handle.read(4) == b"PK\x03\x04":
                return "pytorch"
    except OSError:
        return None
    return None


def compatible_checkpoint(
    checkpoint: Path,
    checkpoint_root: Path,
    preflight: dict[str, Any],
    expected_bytes: int,
) -> bool:
    checkpoint_kind = checkpoint_format(checkpoint, expected_bytes)
    if checkpoint_kind is None:
        return False
    marker = find_checkpoint_marker(checkpoint, checkpoint_root)
    if marker is None:
        return True
    try:
        validate_checkpoint_marker(marker, preflight)
    except ContractError:
        return False
    return True


def resolve_checkpoint(
    request_mode: str,
    checkpoint_root: str | Path,
    preflight: dict[str, Any],
    expected_bytes: int,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    if request_mode == "explicit":
        if checkpoint_path is None:
            raise ContractError("explicit checkpoint replay requires a path")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ContractError(f"checkpoint is unavailable: {checkpoint}")
        checkpoint_kind = checkpoint_format(checkpoint, expected_bytes)
        if checkpoint_kind is None:
            raise ContractError(
                "checkpoint is neither a compatible raw-weight file nor a "
                "PyTorch state dictionary: "
                f"expected_raw_bytes={expected_bytes}, "
                f"actual_bytes={checkpoint.stat().st_size}"
            )
        marker = find_checkpoint_marker(checkpoint, root)
        if marker is not None:
            validate_checkpoint_marker(marker, preflight)
        return {
            "resolved_path": str(checkpoint),
            "sidecar_path": marker,
            "format": checkpoint_kind,
        }

    if request_mode != "latest":
        raise ContractError(f"unsupported checkpoint request: {request_mode!r}")
    if not root.is_dir():
        raise ContractError(f"checkpoint root is unavailable: {root}")
    candidates = [
        checkpoint
        for checkpoint in root.rglob("*.bin")
        if compatible_checkpoint(checkpoint, root, preflight, expected_bytes)
    ]
    if not candidates:
        raise ContractError(
            f"no compatible checkpoint found under {root} for Fight Caves"
        )
    checkpoint = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    marker = find_checkpoint_marker(checkpoint, root)
    return {
        "resolved_path": str(checkpoint),
        "sidecar_path": marker,
        "format": checkpoint_format(checkpoint, expected_bytes),
    }


# Policy replay

def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def repo_root():
    return os.path.abspath(os.path.join(script_dir(), "..", ".."))


def ensure_local_pufferlib_on_path():
    default_puffer_dir = repo_root()
    puffer_dir = os.environ.get("PUFFERLIB_DIR", default_puffer_dir)
    if os.path.isdir(puffer_dir) and puffer_dir not in sys.path:
        sys.path.insert(0, puffer_dir)
    return puffer_dir


def find_compiled_backend(puffer_dir=None):
    override = os.environ.get("FC_COMPILED_BACKEND_PATH")
    if override:
        if os.path.isfile(override):
            return override
        raise RuntimeError(f"FC_COMPILED_BACKEND_PATH is not a file: {override}")

    puffer_dir = puffer_dir or ensure_local_pufferlib_on_path()
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    preferred = os.path.join(puffer_dir, "pufferlib", f"_C{extension_suffix}")
    if os.path.isfile(preferred):
        return preferred
    candidates = []
    for pattern in ("_C*.so", "_C*.dylib", "_C*.pyd"):
        candidates.extend(glob.glob(os.path.join(puffer_dir, "pufferlib", pattern)))
    if not candidates:
        raise RuntimeError(
            f"compiled Puffer backend not found under {puffer_dir}/pufferlib"
        )
    return max(candidates, key=os.path.getmtime)


def load_evaluator_preflight(backend_path):

    source_config = os.environ.get(
        "CONFIG_PATH", os.path.join(repo_root(), "config", "fight_caves.ini")
    )
    default_config = os.path.join(repo_root(), "config", "default.ini")
    active_loadout = os.environ.get("FC_ACTIVE_LOADOUT", "FC_LOADOUT_SOTA_TBOW")
    return build_verified_preflight(
        backend_path, source_config, default_config, active_loadout
    )


def expected_parameter_bytes(contract):
    source_config = os.environ.get(
        "CONFIG_PATH", os.path.join(repo_root(), "config", "fight_caves.ini")
    )
    default_config = os.path.join(repo_root(), "config", "default.ini")

    return expected_checkpoint_parameter_bytes(
        contract, source_config, default_config
    )


def verify_runtime_assets():
    preflight = os.path.join(
        repo_root(), "ocean", "fight_caves", "tools.py"
    )
    result = subprocess.run(
        [sys.executable, preflight, "preflight", "--mode", "viewer-runtime"],
        cwd=repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Fight Caves runtime/viewer asset preflight failed")


def checkpoint_diagnostic(reason, checkpoint_path, expected_bytes, contract):
    actual_bytes = (
        os.path.getsize(checkpoint_path)
        if checkpoint_path and os.path.isfile(checkpoint_path)
        else "missing"
    )
    return (
        f"checkpoint rejected: {reason}\n"
        f"expected_policy_obs={contract['policy_obs_size']} "
        f"actual_policy_obs={contract['policy_obs_size']}\n"
        f"expected_puffer_obs={contract['puffer_obs_size']} "
        f"actual_puffer_obs={contract['puffer_obs_size']}\n"
        f"expected_action_dims={contract['puffer_action_dims']} "
        f"actual_action_dims={contract['puffer_action_dims']}\n"
        f"expected_parameter_bytes={expected_bytes} "
        f"actual_parameter_bytes={actual_bytes}\n"
        f"observation_version={contract['observation_version']}\n"
        f"action_version={contract['action_version']}\n"
        f"reward_version={contract['reward_version']}\n"
        f"prayer_timing_version={contract['prayer_timing_version']}\n"
        f"state_hash_version={contract['state_hash_version']}"
    )


def latest_source_mtime():
    repo = repo_root()
    patterns = [
        os.path.join(repo, "ocean", "fight_caves", "*.h"),
        os.path.join(repo, "ocean", "fight_caves", "*.c"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return max((os.path.getmtime(path) for path in files), default=0.0)


def find_viewer():
    """Find the fc_viewer binary."""
    override = os.environ.get("FC_VIEWER_PATH")
    if override:
        if os.path.isfile(override):
            return override
        raise RuntimeError(f"FC_VIEWER_PATH does not point to a file: {override}")

    repo = repo_root()
    source_mtime = latest_source_mtime()
    preferred = [
        os.path.join(repo, "build", "fight_caves-viewer", "fc_viewer"),
    ]
    candidates = [path for path in preferred if os.path.isfile(path)]

    patterns = [
        os.path.join(repo, "build*", "fight_caves-viewer", "fc_viewer"),
    ]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None

    seen = set()
    unique = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    for path in unique:
        if os.path.getmtime(path) >= source_mtime:
            return path
    return max(unique, key=os.path.getmtime)


def read_obs_line(proc, total_line_floats):
    import numpy as np
    """Read one line of space-separated floats from viewer stdout."""
    line = proc.stdout.readline()
    if not line:
        return None
    values = line.strip().split()
    if len(values) != total_line_floats:
        print(f"[eval] Warning: expected {total_line_floats} floats, got {len(values)}",
              file=sys.stderr)
        return None
    return np.array([float(v) for v in values], dtype=np.float32)


def send_actions(proc, actions):
    """Write one action per Puffer action head to viewer stdin."""
    line = " ".join(str(int(a)) for a in actions) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def sample_masked(logits_list, mask, act_dims, deterministic=False):
    import numpy as np
    """Sample actions from logits with mask applied."""
    actions = []
    mask_offset = 0
    for head_idx, (logits, dim) in enumerate(zip(logits_list, act_dims)):
        head_mask = mask[mask_offset:mask_offset + dim]
        mask_offset += dim

        # Apply mask: set invalid actions to -inf
        masked_logits = logits.copy()
        for i in range(dim):
            if head_mask[i] < 0.5:
                masked_logits[i] = -1e9

        if deterministic:
            action = np.argmax(masked_logits)
        else:
            # Softmax + sample
            logits_shifted = masked_logits - np.max(masked_logits)
            probs = np.exp(logits_shifted)
            probs = probs / (probs.sum() + 1e-8)
            action = np.random.choice(dim, p=probs)

        actions.append(action)
    return actions


def load_policy_weights(policy, checkpoint_path, checkpoint_kind, parameter_bytes):
    import numpy as np
    import torch

    if checkpoint_kind == "pytorch":
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("PyTorch checkpoint does not contain a state dictionary")
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        expected = policy.state_dict()
        if set(state_dict) != set(expected):
            missing = sorted(set(expected) - set(state_dict))
            extra = sorted(set(state_dict) - set(expected))
            raise RuntimeError(
                "PyTorch checkpoint keys do not match the configured policy: "
                f"missing={missing[:1]}, extra={extra[:1]}"
            )
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"PyTorch checkpoint value is not a tensor: {key}")
            if tensor.shape != expected[key].shape:
                raise RuntimeError(
                    f"PyTorch checkpoint shape mismatch for {key}: "
                    f"expected={list(expected[key].shape)}, actual={list(tensor.shape)}"
                )
        policy.load_state_dict(state_dict, strict=True)
        print(
            f"[eval] Loaded PyTorch state dictionary ({len(state_dict)} tensors)",
            file=sys.stderr,
        )
        return

    if checkpoint_kind != "raw":
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_kind!r}")

    # The CUDA trainer saves a flat float32 buffer in this order:
    # encoder.weight, fused decoder/action+value weight, and recurrent layers.
    weights = np.fromfile(checkpoint_path, dtype=np.float32)
    print(f"[eval] Checkpoint: {len(weights)} floats", file=sys.stderr)
    state_dict = policy.state_dict()
    for key in state_dict:
        if "bias" in key:
            state_dict[key] = torch.zeros_like(state_dict[key])

    offset = 0

    def load_tensor(key):
        nonlocal offset
        if key not in state_dict:
            raise KeyError(f"{key} not in model state_dict")
        numel = state_dict[key].numel()
        if offset + numel > len(weights):
            raise RuntimeError(f"weights exhausted at {key}")
        state_dict[key] = torch.from_numpy(
            weights[offset:offset + numel].reshape(state_dict[key].shape).copy()
        )
        offset += numel
        print(
            f"  loaded {key}: {list(state_dict[key].shape)} ({numel})",
            file=sys.stderr,
        )

    load_tensor("encoder.encoder.weight")
    decoder_key = "decoder.decoder.weight"
    value_key = "decoder.value_function.weight"
    if decoder_key not in state_dict or value_key not in state_dict:
        raise KeyError("decoder weights missing from model state_dict")

    decoder_rows = state_dict[decoder_key].shape[0]
    hidden_size = state_dict[decoder_key].shape[1]
    value_rows = state_dict[value_key].shape[0]
    fused_rows = decoder_rows + value_rows
    fused_numel = fused_rows * hidden_size
    if offset + fused_numel > len(weights):
        raise RuntimeError("weights exhausted at fused decoder")
    fused_decoder = weights[offset:offset + fused_numel].reshape(
        fused_rows, hidden_size
    ).copy()
    state_dict[decoder_key] = torch.from_numpy(fused_decoder[:decoder_rows])
    state_dict[value_key] = torch.from_numpy(fused_decoder[decoder_rows:])
    offset += fused_numel
    print(
        f"  loaded fused decoder: {list(fused_decoder.shape)} "
        f"-> {list(state_dict[decoder_key].shape)} + "
        f"{list(state_dict[value_key].shape)}",
        file=sys.stderr,
    )

    network_keys = sorted(
        [
            key for key in state_dict
            if key.startswith("network.layers.") and key.endswith(".weight")
        ],
        key=lambda key: int(key.split(".")[2]),
    )
    model_parameter_floats = (
        state_dict["encoder.encoder.weight"].numel()
        + state_dict[decoder_key].numel()
        + state_dict[value_key].numel()
        + sum(state_dict[key].numel() for key in network_keys)
    )
    model_parameter_bytes = model_parameter_floats * np.dtype(np.float32).itemsize
    if model_parameter_bytes != parameter_bytes:
        raise RuntimeError(
            "constructed model/raw layout mismatch: "
            f"expected_parameter_bytes={parameter_bytes}, "
            f"actual_parameter_bytes={model_parameter_bytes}"
        )
    for key in network_keys:
        load_tensor(key)

    policy.load_state_dict(state_dict)
    if offset != len(weights):
        raise RuntimeError(
            f"unused supplied weights: loaded={offset}, actual={len(weights)}"
        )
    print(f"[eval] Loaded {offset}/{len(weights)} raw weights", file=sys.stderr)


def eval_main():
    import numpy as np
    # Parse our args FIRST, then clear sys.argv so PufferLib's
    # load_config() doesn't choke on our flags.
    parser = argparse.ArgumentParser(description="Watch trained policy in debug viewer")
    parser.add_argument("--ckpt", type=str, default="latest",
                        help="Path to .bin checkpoint or 'latest'")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax instead of sampling")
    parser.add_argument("--random", action="store_true",
                        help="Use random valid actions (no checkpoint needed)")
    parser.add_argument("--start-wave", type=int, default=0,
                        help="Start at this wave (0 = wave 1)")
    parser.add_argument("--speed", type=int, choices=[1, 2, 4, 10], default=1,
                        help="Initial replay speed multiplier (buttons can switch to TPS presets)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Stop after this many replay episodes (0 = unlimited)")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Clear sys.argv so PufferLib doesn't see our flags
    sys.argv = [sys.argv[0]]
    puffer_dir = ensure_local_pufferlib_on_path()

    try:
        verify_runtime_assets()
        backend_path = find_compiled_backend(puffer_dir)
        verified_preflight = load_evaluator_preflight(backend_path)
    except Exception as exc:
        print(f"Error: evaluator compiled-contract preflight failed: {exc}", file=sys.stderr)
        return 1
    contract = verified_preflight["contract"]
    policy_obs_size = contract["policy_obs_size"]
    act_dims = contract["puffer_action_dims"]
    mask_size = contract["puffer_mask_size"]
    total_line_floats = contract["puffer_obs_size"]

    # Find viewer binary
    viewer_path = find_viewer()
    if not viewer_path:
        print(
            "Error: fc_viewer binary not found. Build with: "
            "python3 ocean/fight_caves/tools.py build-viewer",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.path.getmtime(viewer_path) < latest_source_mtime():
        print(
            f"Error: selected fc_viewer is older than current core/viewer sources: {viewer_path}",
            file=sys.stderr,
        )
        print(
            "Rebuild it first with: python3 ocean/fight_caves/tools.py build-viewer",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[eval] Viewer: {viewer_path}", file=sys.stderr)
    print(f"[eval] Replay speed: {args.speed}x", file=sys.stderr)
    if args.episodes > 0:
        print(f"[eval] Episode limit: {args.episodes}", file=sys.stderr)
    print(
        f"[eval] Contract: policy_obs={policy_obs_size} mask={mask_size} "
        f"heads={len(act_dims)} total={total_line_floats}",
        file=sys.stderr,
    )

    # Load checkpoint (unless --random)
    policy = None
    if not args.random:
        try:
            parameter_bytes = expected_parameter_bytes(contract)
        except Exception as exc:
            print(f"Error: cannot derive expected checkpoint size: {exc}", file=sys.stderr)
            return 1


        request_mode = "latest" if args.ckpt == "latest" else "explicit"
        checkpoint_root = os.environ.get(
            "FC_CHECKPOINT_ROOT",
            os.path.join(repo_root(), "checkpoints"),
        )
        try:
            resolution = resolve_checkpoint(
                request_mode,
                checkpoint_root,
                verified_preflight,
                parameter_bytes,
                checkpoint_path=None if request_mode == "latest" else args.ckpt,
            )
        except ContractError as exc:
            print(
                checkpoint_diagnostic(
                    exc, None if args.ckpt == "latest" else args.ckpt,
                    parameter_bytes, contract,
                ),
                file=sys.stderr,
            )
            return 1

        checkpoint_path = resolution["resolved_path"]
        checkpoint_kind = resolution["format"]
        print(
            f"[eval] Checkpoint: {checkpoint_path} ({checkpoint_kind})",
            file=sys.stderr,
        )

        try:
            import torch
            import pufferlib.models
            from pufferlib.pufferl import load_config

            eval_args = load_config("fight_caves")
            policy_kwargs = eval_args["policy"]
            network_cls = getattr(pufferlib.models, eval_args["torch"]["network"])
            encoder_cls = getattr(pufferlib.models, eval_args["torch"]["encoder"])
            decoder_cls = getattr(pufferlib.models, eval_args["torch"]["decoder"])

            network = network_cls(**policy_kwargs)
            encoder = encoder_cls(total_line_floats, policy_kwargs["hidden_size"])
            decoder = decoder_cls(act_dims, policy_kwargs["hidden_size"])
            policy = pufferlib.models.Policy(encoder, decoder, network)
            policy = policy.cpu()
            load_policy_weights(
                policy, checkpoint_path, checkpoint_kind, parameter_bytes
            )

            policy = policy.cpu()
            policy.eval()
            print("[eval] Policy ready (CPU)", file=sys.stderr)

        except Exception as exc:
            print(
                checkpoint_diagnostic(
                    exc, checkpoint_path, parameter_bytes, contract
                ),
                file=sys.stderr,
            )
            return 1

    # Launch viewer subprocess from repo root so sprite paths resolve
    print("[eval] Launching viewer...", file=sys.stderr)
    viewer_env = os.environ.copy()
    viewer_env.setdefault(
        "FC_ASSET_ROOT",
        os.path.join(repo_root(), "resources", "fight_caves", "viewer"),
    )
    viewer_env.setdefault("FC_REPO_ROOT", repo_root())
    proc = subprocess.Popen(
        [viewer_path, "--policy-pipe", "--speed", str(args.speed)] +
            (["--episodes", str(args.episodes)] if args.episodes > 0 else []) +
            (["--start-wave", str(args.start_wave)] if args.start_wave > 0 else []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # let viewer stderr pass through to terminal
        text=True,
        bufsize=1,
        cwd=repo_root(),
        env=viewer_env,
    )

    # Hidden state for recurrent policy (MinGRU)
    hidden = None
    if policy is not None:
        import torch
        hidden = policy.initial_state(1, 'cpu')

    try:
        tick = 0
        while True:
            # Read observation from viewer
            obs_data = read_obs_line(proc, total_line_floats)
            if obs_data is None:
                print("[eval] Viewer closed or read error", file=sys.stderr)
                break

            obs = obs_data[:policy_obs_size]
            mask = obs_data[policy_obs_size:]

            if args.random or policy is None:
                # Random valid actions
                actions = sample_masked(
                    [np.zeros(d) for d in act_dims], mask, act_dims, deterministic=False)
            else:
                # Policy inference: feed the same Puffer observation used in training.
                import torch
                with torch.no_grad():
                    full_input = torch.from_numpy(obs_data).unsqueeze(0)
                    output = policy.forward_eval(full_input, hidden)
                    # forward_eval returns (logits, values, state)
                    logits_raw, _values, hidden = output

                    # Extract per-head logits
                    if isinstance(logits_raw, (list, tuple)):
                        logits_list = [l.squeeze(0).numpy() for l in logits_raw]
                    else:
                        # Single tensor — split by action dims
                        lr = logits_raw.squeeze(0).numpy()
                        logits_list = []
                        off = 0
                        for d in act_dims:
                            logits_list.append(lr[off:off+d])
                            off += d

                actions = sample_masked(logits_list, mask, act_dims, args.deterministic)

            # Send actions to viewer
            send_actions(proc, actions)
            tick += 1

            if tick % 100 == 0:
                print(f"[eval] Tick {tick}", file=sys.stderr)
            if args.max_ticks > 0 and tick >= args.max_ticks:
                print(f"[eval] Smoke limit reached at tick {tick}", file=sys.stderr)
                break

    except (BrokenPipeError, KeyboardInterrupt):
        print("[eval] Stopped", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def play_main() -> int:
    result = subprocess.run(
        [sys.executable, __file__, "preflight", "--mode", "viewer-runtime"],
        cwd=REPO_ROOT,
    )
    if result.returncode:
        return result.returncode
    viewer = REPO_ROOT / "build/fight_caves-viewer/fc_viewer"
    if not viewer.is_file() or not os.access(viewer, os.X_OK):
        print(f"Fight Caves viewer is not built: {viewer}\n"
              "Build it with: python3 ocean/fight_caves/tools.py build-viewer", file=sys.stderr)
        return 1
    os.chdir(REPO_ROOT)
    os.execv(str(viewer), [str(viewer), *sys.argv[1:]])


def main() -> int:
    commands = {"setup": setup_main, "bundle": bundle_main,
                "preflight": preflight_main, "build-viewer": build_viewer_main,
                "play": play_main, "eval": eval_main}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python3 ocean/fight_caves/tools.py "
              "{setup,bundle,preflight,build-viewer,play,eval} [options]\n"
              "Use COMMAND --help for command options (play forwards viewer options).")
        return 0 if len(sys.argv) > 1 else 2
    command = sys.argv.pop(1)
    if command not in commands:
        print(f"Unknown Fight Caves command: {command}", file=sys.stderr)
        return 2
    return commands[command]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
