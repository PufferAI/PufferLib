#!/usr/bin/env python3
"""Validate, package, and generate runtime metadata for OSRS assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile


EXPECTED_FORMAT = "puffer-osrs-asset-manifest-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IGNORED_ARCHIVE_NAMES = {".DS_Store"}
IGNORED_ARCHIVE_PREFIXES = ("._",)
IGNORED_ARCHIVE_DIRS = {"__MACOSX"}


@dataclass(frozen=True)
class AssetArchive:
    name: str
    url: str
    sha256: str
    strip_components: int


@dataclass(frozen=True)
class RequiredGroup:
    name: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class AssetManifest:
    format: str
    asset_version: str
    archive: AssetArchive
    required_groups: tuple[RequiredGroup, ...]


def path_is_safe(path: str | None) -> bool:
    if not path:
        return False
    if path.startswith("/") or "\\" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def expect_str(raw: object, key: str, context: str) -> str:
    if not isinstance(raw, dict):
        raise SystemExit(f"osrs-assets: {context} must be an object")
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"osrs-assets: {context}.{key} must be a string")
    return value


def expect_int(raw: object, key: str, context: str) -> int:
    if not isinstance(raw, dict):
        raise SystemExit(f"osrs-assets: {context} must be an object")
    value = raw.get(key)
    if not isinstance(value, int):
        raise SystemExit(f"osrs-assets: {context}.{key} must be an integer")
    return value


def load_manifest(path: Path) -> AssetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("osrs-assets: manifest must be an object")

    manifest_format = expect_str(raw, "format", "manifest")
    if manifest_format != EXPECTED_FORMAT:
        raise SystemExit(f"osrs-assets: unsupported manifest format: {manifest_format}")

    asset_version = expect_str(raw, "asset_version", "manifest")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", asset_version):
        raise SystemExit(f"osrs-assets: unsafe asset version: {asset_version}")

    archive_raw = raw.get("archive")
    archive = AssetArchive(
        name=expect_str(archive_raw, "name", "archive"),
        url=expect_str(archive_raw, "url", "archive"),
        sha256=expect_str(archive_raw, "sha256", "archive"),
        strip_components=expect_int(archive_raw, "strip_components", "archive"),
    )
    if archive.name != f"{asset_version}.tar.gz":
        raise SystemExit(
            "osrs-assets: archive.name must match asset_version plus .tar.gz"
        )
    if not archive.url.startswith(("https://", "http://", "file://")):
        raise SystemExit(f"osrs-assets: unsupported archive URL: {archive.url}")
    if not SHA256_PATTERN.fullmatch(archive.sha256):
        raise SystemExit(f"osrs-assets: bad archive SHA256: {archive.sha256}")
    if archive.strip_components < 0:
        raise SystemExit("osrs-assets: archive strip_components must be nonnegative")

    groups_raw = raw.get("required_groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise SystemExit("osrs-assets: required_groups must be a nonempty list")

    groups: list[RequiredGroup] = []
    seen_group_names: set[str] = set()
    for group_raw in groups_raw:
        name = expect_str(group_raw, "name", "required_group")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise SystemExit(f"osrs-assets: unsafe required group name: {name}")
        if name in seen_group_names:
            raise SystemExit(f"osrs-assets: repeated required group: {name}")
        seen_group_names.add(name)

        files_raw = group_raw.get("files") if isinstance(group_raw, dict) else None
        if not isinstance(files_raw, list) or not files_raw:
            raise SystemExit(f"osrs-assets: group {name} has no files")
        files: list[str] = []
        seen_files: set[str] = set()
        for file_raw in files_raw:
            if not isinstance(file_raw, str) or not path_is_safe(file_raw):
                raise SystemExit(f"osrs-assets: unsafe asset path in group {name}: {file_raw}")
            if file_raw in seen_files:
                raise SystemExit(f"osrs-assets: repeated asset path in group {name}: {file_raw}")
            seen_files.add(file_raw)
            files.append(file_raw)
        groups.append(RequiredGroup(name=name, files=tuple(files)))

    return AssetManifest(
        format=manifest_format,
        asset_version=asset_version,
        archive=archive,
        required_groups=tuple(groups),
    )


def missing_required_files(manifest: AssetManifest, data_dir: Path) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for group in manifest.required_groups:
        for asset_path in group.files:
            if not (data_dir / asset_path).is_file():
                missing.append((group.name, asset_path))
    return missing


def c_identifier_group_name(name: str) -> str:
    return "OSRS_ASSET_GROUP_" + name.upper()


def c_path_array_name(name: str) -> str:
    return "OSRS_ASSET_" + name.upper() + "_PATHS"


def c_string(value: str) -> str:
    return json.dumps(value)


def generated_header_text(manifest: AssetManifest) -> str:
    lines: list[str] = [
        "/* Generated by ocean/osrs/scripts/osrs_asset_manifest.py. */",
        "#ifndef OSRS_ASSETS_GENERATED_H",
        "#define OSRS_ASSETS_GENERATED_H",
        "",
        "#include <stddef.h>",
        "",
        "typedef enum {",
    ]
    for idx, group in enumerate(manifest.required_groups):
        lines.append(f"    {c_identifier_group_name(group.name)} = {idx},")
    lines.extend([
        "    OSRS_ASSET_GROUP_COUNT,",
        "} OsrsAssetGroupKind;",
        "",
        "typedef struct {",
        "    const char* name;",
        "    const char* const* paths;",
        "    size_t path_count;",
        "} OsrsAssetGroup;",
        "",
    ])

    for group in manifest.required_groups:
        lines.append(f"static const char* const {c_path_array_name(group.name)}[] = {{")
        for asset_path in group.files:
            lines.append(f"    {c_string(asset_path)},")
        lines.extend(["};", ""])

    lines.extend([
        "static const OsrsAssetGroup OSRS_ASSET_GROUPS[OSRS_ASSET_GROUP_COUNT] = {",
    ])
    for group in manifest.required_groups:
        ident = c_identifier_group_name(group.name)
        array_name = c_path_array_name(group.name)
        lines.extend([
            f"    [{ident}] = {{",
            f"        .name = {c_string(group.name)},",
            f"        .paths = {array_name},",
            f"        .path_count = sizeof({array_name}) / sizeof({array_name}[0]),",
            "    },",
        ])
    lines.extend([
        "};",
        "",
        "#endif",
        "",
    ])
    return "\n".join(lines)


def write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def archive_member_allowed(path: Path) -> bool:
    if path.name in IGNORED_ARCHIVE_NAMES:
        return False
    if any(part in IGNORED_ARCHIVE_DIRS for part in path.parts):
        return False
    return not any(part.startswith(IGNORED_ARCHIVE_PREFIXES) for part in path.parts)


def archive_relpaths(data_dir: Path) -> list[Path]:
    relpaths: list[Path] = []
    for path in data_dir.rglob("*"):
        relpath = path.relative_to(data_dir)
        if ".download" in relpath.parts:
            continue
        if not archive_member_allowed(relpath):
            continue
        if path.is_file():
            relpaths.append(relpath)
    return sorted(relpaths, key=lambda p: p.as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_archive(manifest: AssetManifest, data_dir: Path, dist_dir: Path) -> Path:
    missing = missing_required_files(manifest, data_dir)
    if missing:
        for group_name, asset_path in missing:
            print(f"{group_name}\t{asset_path}", file=sys.stderr)
        raise SystemExit("osrs-assets: cannot package archive with missing required files")

    dist_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dist_dir / manifest.archive.name
    top_level = manifest.asset_version
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gzip_file:
            with tarfile.open(
                fileobj=gzip_file,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                for relpath in archive_relpaths(data_dir):
                    tar.add(
                        data_dir / relpath,
                        arcname=f"{top_level}/{relpath.as_posix()}",
                        filter=normalize_tarinfo,
                        recursive=False,
                    )
    return archive_path


def normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def print_group_files(manifest: AssetManifest, group_names: list[str]) -> None:
    requested = set(group_names)
    manifest_names = {group.name for group in manifest.required_groups}
    unknown = requested - manifest_names
    if unknown:
        raise SystemExit(f"osrs-assets: unknown groups: {', '.join(sorted(unknown))}")
    for group in manifest.required_groups:
        if requested and group.name not in requested:
            continue
        for asset_path in group.files:
            print(f"{group.name}\t{asset_path}")


def emcc_preload_args(manifest: AssetManifest, group_names: list[str]) -> list[str]:
    requested = set(group_names)
    manifest_names = {group.name for group in manifest.required_groups}
    unknown = requested - manifest_names
    if unknown:
        raise SystemExit(f"osrs-assets: unknown groups: {', '.join(sorted(unknown))}")
    seen: set[str] = set()
    lines: list[str] = []
    for group in manifest.required_groups:
        if requested and group.name not in requested:
            continue
        for asset_path in group.files:
            if asset_path in seen:
                continue
            seen.add(asset_path)
            runtime_path = f"ocean/osrs/data/{asset_path}"
            lines.append(f"--preload-file {runtime_path}@{runtime_path}")
    return lines


def print_emcc_preload_args(manifest: AssetManifest, group_names: list[str]) -> None:
    for line in emcc_preload_args(manifest, group_names):
        print(line)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="manage OSRS asset manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive_parser = subparsers.add_parser("archive-tsv")
    archive_parser.add_argument("manifest", type=Path)

    missing_parser = subparsers.add_parser("missing-required")
    missing_parser.add_argument("manifest", type=Path)
    missing_parser.add_argument("data_dir", type=Path)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("manifest", type=Path)

    generate_parser = subparsers.add_parser("generate-c-header")
    generate_parser.add_argument("manifest", type=Path)
    generate_parser.add_argument("--output", required=True, type=Path)

    check_parser = subparsers.add_parser("check-c-header")
    check_parser.add_argument("manifest", type=Path)
    check_parser.add_argument("--output", required=True, type=Path)

    group_files_parser = subparsers.add_parser("group-files")
    group_files_parser.add_argument("manifest", type=Path)
    group_files_parser.add_argument("--group", action="append", default=[])

    emcc_parser = subparsers.add_parser("emcc-preload-args")
    emcc_parser.add_argument("manifest", type=Path)
    emcc_parser.add_argument("--group", action="append", default=[])

    package_parser = subparsers.add_parser("package-archive")
    package_parser.add_argument("manifest", type=Path)
    package_parser.add_argument("--data-dir", required=True, type=Path)
    package_parser.add_argument("--dist-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)

    if args.command == "archive-tsv":
        archive = manifest.archive
        print(f"{archive.name}\t{archive.url}\t{archive.sha256}\t{archive.strip_components}")
        return 0

    if args.command == "missing-required":
        missing = missing_required_files(manifest, args.data_dir)
        for group_name, asset_path in missing:
            print(f"{group_name}\t{asset_path}")
        return 1 if missing else 0

    if args.command == "validate":
        return 0

    if args.command == "generate-c-header":
        write_if_changed(args.output, generated_header_text(manifest))
        return 0

    if args.command == "check-c-header":
        expected = generated_header_text(manifest)
        if not args.output.exists():
            raise SystemExit(f"osrs-assets: generated header is missing: {args.output}")
        actual = args.output.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(f"osrs-assets: generated header is stale: {args.output}")
        return 0

    if args.command == "group-files":
        print_group_files(manifest, args.group)
        return 0

    if args.command == "emcc-preload-args":
        print_emcc_preload_args(manifest, args.group)
        return 0

    if args.command == "package-archive":
        archive_path = package_archive(manifest, args.data_dir, args.dist_dir)
        print(f"{sha256_file(archive_path)}\t{archive_path}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
