import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


BUNDLE_FORMAT = "puffer-osrs-web-v2"
VERSION_TOKEN = b"__OSRS_BUNDLE_VERSION__"
GAME_SCRIPT_REFERENCE = b"src=game.js"
GAME_FILE_NAMES = (
    "game.html",
    "game.js",
    "game.wasm",
    "game.wasm.map",
    "game.data",
)
MODEL_OUTPUT_PATH = "models/policy.bin"
OSRS_ENVIRONMENTS = ("inferno", "colosseum", "zulrah", "nh_pvp")
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PackageMetadata:
    environment: str
    source_commit: str
    hidden_size: int
    num_layers: int
    entity_encoder: int


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_record(content: bytes) -> dict[str, int | str]:
    return {"sha256": sha256_bytes(content), "size": len(content)}


def derive_bundle_version(
    ordered_hashes: Sequence[tuple[str, str]], metadata: PackageMetadata
) -> str:
    version_input = {
        "files": ordered_hashes,
        "environment": metadata.environment,
        "source_commit": metadata.source_commit,
        "hidden_size": metadata.hidden_size,
        "num_layers": metadata.num_layers,
        "entity_encoder": metadata.entity_encoder,
    }
    encoded = json.dumps(version_input, separators=(",", ":")).encode()
    return sha256_bytes(encoded)[:16]


def build_manifest(
    version: str,
    metadata: PackageMetadata,
    files: dict[str, dict[str, int | str]],
) -> dict[str, object]:
    return {
        "format": BUNDLE_FORMAT,
        "version": version,
        "environment": metadata.environment,
        "source_commit": metadata.source_commit,
        "files": files,
        "model": {
            "path": MODEL_OUTPUT_PATH,
            "hidden_size": metadata.hidden_size,
            "num_layers": metadata.num_layers,
            "entity_encoder": metadata.entity_encoder,
        },
    }


def paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def package_bundle(
    build_dir: Path,
    model_path: Path,
    output_dir: Path,
    metadata: PackageMetadata,
) -> None:
    build_dir = build_dir.resolve()
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    if paths_overlap(build_dir, output_dir):
        raise ValueError("output directory must not overlap the build directory")
    if paths_overlap(model_path, output_dir):
        raise ValueError("output directory must not overlap the model path")
    source_contents = {
        name: (build_dir / name).read_bytes() for name in GAME_FILE_NAMES
    }
    model_content = model_path.read_bytes()
    source_html = source_contents["game.html"]
    if source_html.count(VERSION_TOKEN) != 1:
        raise ValueError("game.html must contain exactly one __OSRS_BUNDLE_VERSION__ token")
    if source_html.count(GAME_SCRIPT_REFERENCE) != 1:
        raise ValueError("game.html must contain exactly one game.js script reference")

    source_records = {
        name: file_record(source_contents[name]) for name in GAME_FILE_NAMES
    }
    model_record = file_record(model_content)
    ordered_hashes = tuple(
        (name, str(source_records[name]["sha256"])) for name in GAME_FILE_NAMES
    ) + ((MODEL_OUTPUT_PATH, str(model_record["sha256"])),)
    version = derive_bundle_version(ordered_hashes, metadata)

    packaged_contents = dict(source_contents)
    version_bytes = version.encode()
    packaged_contents["game.html"] = source_html.replace(
        VERSION_TOKEN, version_bytes
    ).replace(
        GAME_SCRIPT_REFERENCE,
        GAME_SCRIPT_REFERENCE + b"?v=" + version_bytes,
    )
    packaged_contents[MODEL_OUTPUT_PATH] = model_content
    final_records = dict(source_records)
    final_records["game.html"] = file_record(packaged_contents["game.html"])
    final_records[MODEL_OUTPUT_PATH] = model_record
    manifest = build_manifest(version, metadata, final_records)
    manifest_content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    if output_dir.exists() and (not output_dir.is_dir() or output_dir.is_symlink()):
        raise NotADirectoryError(f"output path is not a dedicated directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    backup_dir = staged_dir.with_name(f"{staged_dir.name}.previous")
    try:
        (staged_dir / "models").mkdir()
        for relative_path, content in packaged_contents.items():
            (staged_dir / relative_path).write_bytes(content)
        (staged_dir / "bundle.json").write_bytes(manifest_content)

        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            staged_dir.replace(output_dir)
        except BaseException:
            if backup_dir.exists():
                backup_dir.replace(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def source_commit(value: str) -> str:
    if SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "source commit must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def positive_integer(label: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{label} must be a positive integer"
            ) from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{label} must be a positive integer")
        return parsed

    return parse


def entity_encoder(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("entity encoder must be 0 or 1") from error
    if parsed not in (0, 1):
        raise argparse.ArgumentTypeError("entity encoder must be 0 or 1")
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--environment",
        choices=OSRS_ENVIRONMENTS,
        required=True,
    )
    parser.add_argument("--source-commit", type=source_commit, required=True)
    parser.add_argument(
        "--hidden-size", type=positive_integer("hidden size"), required=True
    )
    parser.add_argument(
        "--num-layers", type=positive_integer("num layers"), required=True
    )
    parser.add_argument("--entity-encoder", type=entity_encoder, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    metadata = PackageMetadata(
        environment=arguments.environment,
        source_commit=arguments.source_commit,
        hidden_size=arguments.hidden_size,
        num_layers=arguments.num_layers,
        entity_encoder=arguments.entity_encoder,
    )
    package_bundle(arguments.build_dir, arguments.model, arguments.out, metadata)


if __name__ == "__main__":
    main()
