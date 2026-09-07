#!/usr/bin/env python3
"""Install and verify Fight Caves runtime and viewer asset bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[3]
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


def parse_args() -> argparse.Namespace:
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


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
