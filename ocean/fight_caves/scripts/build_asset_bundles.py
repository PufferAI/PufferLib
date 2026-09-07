#!/usr/bin/env python3
"""Build deterministic Fight Caves release bundles and their manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


BUFFER_SIZE = 1024 * 1024
RUNTIME_FILES = (
    "fightcaves.collision",
    "fightcaves.movement",
    "fightcaves.los",
)


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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


def parse_args() -> argparse.Namespace:
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


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
