import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = REPOSITORY_ROOT / "tools/web/package_osrs.py"
SOURCE_COMMIT = "a" * 40
ENVIRONMENT = "inferno"
GAME_FILES = (
    "game.html",
    "game.js",
    "game.wasm",
    "game.wasm.map",
    "game.data",
)
OUTPUT_FILES = {
    "game.html",
    "game.js",
    "game.wasm",
    "game.wasm.map",
    "game.data",
    "bundle.json",
    "models/policy.bin",
}


def write_build(
    build_dir: Path,
    html: bytes = b"before __OSRS_BUNDLE_VERSION__ <script src=game.js></script> after\n",
) -> None:
    build_dir.mkdir()
    contents = {
        "game.html": html,
        "game.js": b"javascript\n",
        "game.wasm": b"wasm\x00bytes",
        "game.wasm.map": b'{"version": 3}\n',
        "game.data": b"data\x00bytes",
    }
    for name, content in contents.items():
        (build_dir / name).write_bytes(content)


def run_package(
    build_dir: Path,
    model: Path,
    output_dir: Path,
    *,
    source_commit: str = SOURCE_COMMIT,
    environment: str = ENVIRONMENT,
    hidden_size: str = "512",
    num_layers: str = "2",
    entity_encoder: str = "0",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--build-dir",
            str(build_dir),
            "--model",
            str(model),
            "--out",
            str(output_dir),
            "--environment",
            environment,
            "--source-commit",
            source_commit,
            "--hidden-size",
            hidden_size,
            "--num-layers",
            num_layers,
            "--entity-encoder",
            entity_encoder,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )


def output_file_paths(output_dir: Path) -> set[str]:
    return {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_required_build_file_does_not_replace_existing_output(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    (build_dir / "game.data").unlink()
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep")

    result = run_package(build_dir, model, output_dir)

    assert result.returncode != 0
    assert "game.data" in result.stderr
    assert sentinel.read_text() == "keep"
    assert output_file_paths(output_dir) == {"keep.txt"}


def test_missing_model_does_not_replace_existing_output(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    output_dir = tmp_path / "site"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep")

    result = run_package(build_dir, tmp_path / "missing.bin", output_dir)

    assert result.returncode != 0
    assert "missing.bin" in result.stderr
    assert sentinel.read_text() == "keep"
    assert output_file_paths(output_dir) == {"keep.txt"}


@pytest.mark.parametrize(
    ("overrides", "invalid_value"),
    [
        ({"source_commit": "A" * 40}, "source commit"),
        ({"source_commit": "a" * 39}, "source commit"),
        ({"hidden_size": "0"}, "hidden size"),
        ({"hidden_size": "not-an-int"}, "hidden size"),
        ({"num_layers": "0"}, "num layers"),
        ({"entity_encoder": "2"}, "entity encoder"),
    ],
)
def test_invalid_metadata_fails(
    tmp_path: Path,
    overrides: dict[str, str],
    invalid_value: str,
) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"

    result = run_package(build_dir, model, output_dir, **overrides)

    assert result.returncode != 0
    assert invalid_value in result.stderr.lower()
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "html",
    [
        b"no token\n",
        b"__OSRS_BUNDLE_VERSION__ and __OSRS_BUNDLE_VERSION__\n",
    ],
)
def test_html_requires_exactly_one_version_token(tmp_path: Path, html: bytes) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir, html)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"

    result = run_package(build_dir, model, output_dir)

    assert result.returncode != 0
    assert "exactly one" in result.stderr.lower()
    assert not output_dir.exists()

@pytest.mark.parametrize(
    "html",
    [
        b"__OSRS_BUNDLE_VERSION__ without script\n",
        b"__OSRS_BUNDLE_VERSION__ <script src=game.js></script> <script src=game.js></script>\n",
    ],
)
def test_html_requires_exactly_one_game_script_reference(
    tmp_path: Path, html: bytes
) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir, html)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"

    result = run_package(build_dir, model, output_dir)

    assert result.returncode != 0
    assert "game.js" in result.stderr
    assert not output_dir.exists()


def test_output_cannot_replace_build_input(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    original_build = {name: (build_dir / name).read_bytes() for name in GAME_FILES}
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")

    result = run_package(build_dir, model, build_dir)

    assert result.returncode != 0
    assert {name: (build_dir / name).read_bytes() for name in GAME_FILES} == original_build


def test_output_cannot_replace_model_input(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    output_dir = tmp_path / "site"
    output_dir.mkdir()
    model = output_dir / "policy.bin"
    model.write_bytes(b"model")

    result = run_package(build_dir, model, output_dir)

    assert result.returncode != 0
    assert model.read_bytes() == b"model"


def test_success_replaces_output_and_writes_complete_verified_bundle(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    original_build = {name: (build_dir / name).read_bytes() for name in GAME_FILES}
    model = tmp_path / "xbeeukiz-policy.bin"
    model.write_bytes(b"policy\x00weights")
    original_model = model.read_bytes()
    output_dir = tmp_path / "site"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale")

    result = run_package(build_dir, model, output_dir)

    assert result.returncode == 0, result.stderr
    assert output_file_paths(output_dir) == OUTPUT_FILES
    assert (output_dir / "models/policy.bin").read_bytes() == original_model
    final_html = (output_dir / "game.html").read_text()
    assert "__OSRS_BUNDLE_VERSION__" not in final_html
    manifest_bytes = (output_dir / "bundle.json").read_bytes()
    assert manifest_bytes.endswith(b"\n")
    assert manifest_bytes == json.dumps(
        json.loads(manifest_bytes), indent=2, sort_keys=True
    ).encode() + b"\n"
    manifest = json.loads(manifest_bytes)
    assert manifest["format"] == "puffer-osrs-web-v2"
    assert re.fullmatch(r"[0-9a-f]{16}", manifest["version"])
    assert manifest["version"] in final_html
    assert f"src=game.js?v={manifest['version']}" in final_html
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["environment"] == ENVIRONMENT
    assert manifest["model"] == {
        "path": "models/policy.bin",
        "hidden_size": 512,
        "num_layers": 2,
        "entity_encoder": 0,
    }
    for relative_path in OUTPUT_FILES - {"bundle.json"}:
        packaged_file = output_dir / relative_path
        assert manifest["files"][relative_path] == {
            "sha256": sha256(packaged_file),
            "size": packaged_file.stat().st_size,
        }
    assert set(manifest["files"]) == OUTPUT_FILES - {"bundle.json"}
    assert {name: (build_dir / name).read_bytes() for name in GAME_FILES} == original_build
    assert model.read_bytes() == original_model


def test_repeated_package_is_deterministic(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"

    first = run_package(build_dir, model, output_dir)
    first_manifest = (output_dir / "bundle.json").read_bytes()
    first_version = json.loads(first_manifest)["version"]
    second = run_package(build_dir, model, output_dir)
    second_manifest = (output_dir / "bundle.json").read_bytes()

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert json.loads(second_manifest)["version"] == first_version
    assert second_manifest == first_manifest


def test_model_bytes_contribute_to_bundle_version(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"first model")
    output_dir = tmp_path / "site"

    first = run_package(build_dir, model, output_dir)
    first_version = json.loads((output_dir / "bundle.json").read_bytes())["version"]
    model.write_bytes(b"second model")
    second = run_package(build_dir, model, output_dir)
    second_version = json.loads((output_dir / "bundle.json").read_bytes())["version"]

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert second_version != first_version



def test_environment_contributes_to_bundle_version_and_manifest(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    write_build(build_dir)
    model = tmp_path / "policy.bin"
    model.write_bytes(b"model")
    output_dir = tmp_path / "site"

    inferno = run_package(build_dir, model, output_dir, environment="inferno")
    inferno_manifest = json.loads((output_dir / "bundle.json").read_bytes())
    colosseum = run_package(
        build_dir,
        model,
        output_dir,
        environment="colosseum",
    )
    colosseum_manifest = json.loads((output_dir / "bundle.json").read_bytes())

    assert inferno.returncode == 0, inferno.stderr
    assert colosseum.returncode == 0, colosseum.stderr
    assert inferno_manifest["environment"] == "inferno"
    assert colosseum_manifest["environment"] == "colosseum"
    assert inferno_manifest["version"] != colosseum_manifest["version"]