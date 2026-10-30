#!/usr/bin/env python3
"""Install the Fight Caves asset bundles pinned in resources/fight_caves/asset_manifest.json.

Usage: python3 ocean/fight_caves/tools.py setup [--force | --verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = REPO_ROOT / "resources" / "fight_caves"
MANIFEST = RESOURCE_ROOT / "asset_manifest.json"
# Written into each installed bundle directory: the SHA-256 of its archive.
STAMP = ".archive-sha256"


class AssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(bundle: dict) -> bool:
    stamp = RESOURCE_ROOT / bundle["install_prefix"] / STAMP
    return stamp.is_file() and stamp.read_text().strip() == bundle["sha256"]


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "PufferLib-Fight-Caves-assets/1"})
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (OSError, HTTPException) as exc:
        raise AssetError(f"download failed for {url}: {exc}") from exc


def install_bundle(bundle: dict) -> None:
    """Download and verify the archive, extract it, then swap the stamped
    directory in. A failure before the swap leaves the current install alone."""
    archive_name, prefix = bundle["archive"], bundle["install_prefix"]
    # Stage beside the install so the swap is a rename on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=RESOURCE_ROOT) as temp_name:
        temp = Path(temp_name)
        archive = temp / archive_name
        download(bundle["url"], archive)
        if sha256_file(archive) != bundle["sha256"]:
            raise AssetError(f"checksum mismatch for downloaded {archive_name}")
        try:
            with tarfile.open(archive, "r:gz") as source:
                # The data filter rejects absolute paths, paths that leave the
                # destination, links out of it and special files.
                source.extractall(temp / "extracted", filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise AssetError(f"could not extract {archive_name}: {exc}") from exc
        staged = temp / "extracted" / prefix
        if not staged.is_dir():
            raise AssetError(f"{archive_name} has no {prefix}/ directory")
        (staged / STAMP).write_text(bundle["sha256"] + "\n")
        destination = RESOURCE_ROOT / prefix
        if destination.exists():
            destination.rename(temp / "previous")
        staged.rename(destination)
    print(f"Installed Fight Caves {prefix} assets from {archive_name}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the pinned Fight Caves runtime and viewer asset bundles."
    )
    parser.add_argument("command", choices=["setup"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true",
                      help="download and reinstall every bundle")
    mode.add_argument("--verify-only", action="store_true",
                      help="check the installed stamps without downloading")
    args = parser.parse_args(argv)
    try:
        for bundle in json.loads(MANIFEST.read_text(encoding="utf-8"))["bundles"]:
            prefix, archive_name = bundle["install_prefix"], bundle["archive"]
            if is_installed(bundle) and not args.force:
                print(f"Fight Caves {prefix} assets are installed from {archive_name}.")
            elif args.verify_only:
                raise AssetError(f"{prefix}/ is missing or was not installed from "
                                 f"{archive_name}; run "
                                 "python3 ocean/fight_caves/tools.py setup")
            else:
                install_bundle(bundle)
    except (AssetError, OSError) as exc:
        print(f"Fight Caves asset setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
