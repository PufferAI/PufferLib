from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DATA = REPO_ROOT / "ocean" / "fight_caves" / "tools.py"


def load_setup_data():
    spec = importlib.util.spec_from_file_location("fight_caves_setup_data_test", SETUP_DATA)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_archive(path: Path, members: dict[str, bytes]) -> bytes:
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = 0o644
            archive.addfile(info, BytesIO(contents))
    return path.read_bytes()


def bundle_for(archive: Path, archive_data: bytes, payload: bytes) -> dict:
    return {
        "archive": archive.name,
        "url": archive.as_uri(),
        "size_bytes": len(archive_data),
        "sha256": sha256(archive_data),
        "install_prefix": "runtime",
        "files": [
            {
                "path": "runtime/test.map",
                "size_bytes": len(payload),
                "sha256": sha256(payload),
            }
        ],
    }


def test_install_bundle_is_verified_and_transactional(tmp_path, monkeypatch):
    setup_data = load_setup_data()
    install_root = tmp_path / "resources"
    monkeypatch.setattr(setup_data, "RESOURCE_ROOT", install_root)

    payload = b"authoritative arena data"
    archive = tmp_path / "bundle.tar.gz"
    archive_data = make_archive(archive, {"runtime/test.map": payload})
    bundle = bundle_for(archive, archive_data, payload)

    setup_data.install_bundle("core", bundle, force=False)
    installed = install_root / "runtime" / "test.map"
    assert installed.read_bytes() == payload
    assert setup_data.verify_tree(install_root, bundle, exact=True) == []

    installed.write_bytes(b"corrupt")
    assert "wrong size" in setup_data.verify_tree(install_root, bundle, exact=True)[0]


def test_bad_archive_checksum_does_not_replace_existing_assets(tmp_path, monkeypatch):
    setup_data = load_setup_data()
    install_root = tmp_path / "resources"
    existing = install_root / "runtime" / "test.map"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing valid installation")
    monkeypatch.setattr(setup_data, "RESOURCE_ROOT", install_root)

    payload = b"replacement"
    archive = tmp_path / "bundle.tar.gz"
    archive_data = make_archive(archive, {"runtime/test.map": payload})
    bundle = bundle_for(archive, archive_data, payload)
    bundle["sha256"] = "0" * 64

    with pytest.raises(setup_data.AssetError, match="checksum mismatch"):
        setup_data.install_bundle("core", bundle, force=True)
    assert existing.read_bytes() == b"existing valid installation"


def test_unsafe_archive_path_is_rejected_without_partial_install(tmp_path, monkeypatch):
    setup_data = load_setup_data()
    install_root = tmp_path / "resources"
    monkeypatch.setattr(setup_data, "RESOURCE_ROOT", install_root)

    payload = b"map"
    archive = tmp_path / "bundle.tar.gz"
    archive_data = make_archive(
        archive,
        {"runtime/test.map": payload, "../outside": b"must not escape"},
    )
    bundle = bundle_for(archive, archive_data, payload)

    with pytest.raises(setup_data.AssetError, match="unsafe path"):
        setup_data.install_bundle("core", bundle, force=True)
    assert not (install_root / "runtime").exists()
    assert not (tmp_path / "outside").exists()


def test_manifest_rejects_unsafe_and_duplicate_file_paths(tmp_path):
    setup_data = load_setup_data()
    base = {
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    with pytest.raises(setup_data.AssetError, match="unsafe path"):
        setup_data.expected_files({"files": [{"path": "../bad", **base}]})
    with pytest.raises(setup_data.AssetError, match="duplicate file"):
        setup_data.expected_files(
            {"files": [{"path": "runtime/a", **base}, {"path": "runtime/a", **base}]}
        )


def test_download_failure_is_actionable(tmp_path):
    setup_data = load_setup_data()
    missing = (tmp_path / "does-not-exist.tar.gz").as_uri()
    with pytest.raises(setup_data.AssetError, match="download failed"):
        setup_data.download(missing, tmp_path / "download")


@pytest.mark.parametrize("mode, missing", [("core", "clang"), ("cpu", "python")])
def test_preflight_reports_missing_commands_instead_of_continuing(mode, missing):
    preflight = REPO_ROOT / "ocean" / "fight_caves" / "tools.py"
    environment = os.environ.copy()
    environment["PATH"] = ""
    environment.pop("CC", None)
    environment.pop("CXX", None)
    result = subprocess.run(
        [sys.executable, str(preflight), "preflight", "--mode", mode],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert f"required command '{missing}' is unavailable" in result.stderr

@pytest.mark.parametrize("status", [0, 1])
def test_legacy_viewer_build_delegates_to_standard_build(tmp_path, monkeypatch, status):
    tools = load_setup_data()
    monkeypatch.setattr(sys, "argv", [str(SETUP_DATA)])
    monkeypatch.setattr(tools, "REPO_ROOT", tmp_path)
    calls = []
    def build(args, **kwargs):
        calls.append((args, kwargs))
        return status
    monkeypatch.setattr(tools.subprocess, "call", build)
    assert tools.build_viewer_main() == status
    assert calls == [(["bash", "build.sh", "fight_caves", "--fast"], {"cwd": tmp_path})]


def test_compatibility_replay_uses_standard_executable(tmp_path, monkeypatch):
    tools = load_setup_data()
    monkeypatch.setattr(tools, "repo_root", lambda: str(tmp_path))
    assert tools.find_viewer() is None
    viewer = tmp_path / "fight_caves"
    viewer.write_text("#!/bin/sh\nexit 0\n")
    viewer.chmod(0o755)
    assert tools.find_viewer() == str(viewer)


def test_standard_build_prepares_both_asset_bundles():
    build = (REPO_ROOT / "build.sh").read_text()
    branch = build.split('elif [ "$ENV" = "fight_caves" ]; then', 1)[1].split("elif ", 1)[0]
    assert 'tools.py" setup --all' in branch


@pytest.mark.parametrize("broken", [False, True])
def test_replay_asset_check_is_independent_of_display(monkeypatch, broken):
    tools = load_setup_data()
    monkeypatch.delenv("DISPLAY", raising=False)
    def verify(errors, names):
        assert names == ("core", "viewer")
        if broken:
            errors.append("missing viewer asset")
    monkeypatch.setattr(tools, "verify_assets", verify)
    if broken:
        with pytest.raises(tools.AssetError, match="Restore assets"):
            tools.verify_runtime_assets()
    else:
        tools.verify_runtime_assets()


MODULE_PATH = REPO_ROOT / "ocean" / "fight_caves" / "tools.py"
SPEC = importlib.util.spec_from_file_location("fight_caves_eval_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL_CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL_CONTRACT
SPEC.loader.exec_module(EVAL_CONTRACT)


def test_checkpoint_format_accepts_exact_raw_weight_size(tmp_path):
    checkpoint = tmp_path / "raw.bin"
    checkpoint.write_bytes(b"\x00" * 64)
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) == "raw"


def test_checkpoint_format_accepts_pytorch_zip_container(tmp_path):
    checkpoint = tmp_path / "cpu.bin"
    checkpoint.write_bytes(b"PK\x03\x04state-dictionary-placeholder")
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) == "pytorch"


def test_checkpoint_format_rejects_unknown_or_missing_file(tmp_path):
    checkpoint = tmp_path / "wrong.bin"
    checkpoint.write_bytes(b"not a supported checkpoint")
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) is None
    assert EVAL_CONTRACT.checkpoint_format(tmp_path / "missing.bin", 64) is None


@pytest.mark.parametrize("saved,current,accepted", [
    (4, 5, True), (5, 5, True), (5, 4, False), (3, 5, False), (6, 5, False),
])
def test_equipment_hash_checkpoint_migration_is_directional(tmp_path, saved, current, accepted):
    expected = {"state_hash_version": current, "puffer_obs_size": 320,
                "puffer_action_dims": [17, 9, 8], "reward_version": "unchanged"}
    actual = dict(expected, state_hash_version=saved)
    marker = tmp_path / "contract.json"
    marker.write_text(json.dumps({"contract": actual}))
    preflight = {"contract": expected}
    if accepted:
        EVAL_CONTRACT.validate_checkpoint_marker(marker, preflight)
    else:
        with pytest.raises(EVAL_CONTRACT.ContractError, match="does not match"):
            EVAL_CONTRACT.validate_checkpoint_marker(marker, preflight)
    assert expected["state_hash_version"] == current


@pytest.mark.parametrize("field,value", [
    ("puffer_obs_size", 319), ("puffer_action_dims", [17, 9, 8, 14]),
    ("reward_version", "different"), ("unknown_field", 1),
])
def test_equipment_hash_migration_does_not_hide_other_contract_changes(tmp_path, field, value):
    expected = {"state_hash_version": 5, "puffer_obs_size": 320,
                "puffer_action_dims": [17, 9, 8], "reward_version": "unchanged"}
    actual = dict(expected, state_hash_version=4)
    actual[field] = value
    marker = tmp_path / "contract.json"
    marker.write_text(json.dumps({"contract": actual}))
    with pytest.raises(EVAL_CONTRACT.ContractError, match="does not match"):
        EVAL_CONTRACT.validate_checkpoint_marker(marker, {"contract": expected})

ENV_ROOT = REPO_ROOT / "ocean" / "fight_caves"
RESOURCE_ROOT = REPO_ROOT / "resources" / "fight_caves"


def test_environment_uses_flat_implementation_headers():
    for name in ("simulation.h", "fight_caves.h", "assets.h", "ui.h", "render.h",
                 "binding.c", "fight_caves.c", "viewer.c", "tools.py", "CMakeLists.txt"):
        assert (ENV_ROOT / name).is_file()
    assert not (ENV_ROOT / "sources.txt").exists()
    assert '#include "simulation.h"' in (ENV_ROOT / "fight_caves.h").read_text()
    assert 'void fc_step(' in (ENV_ROOT / "simulation.h").read_text()


def test_asset_manifest_is_complete_and_pinned():
    manifest = json.loads(
        (RESOURCE_ROOT / "asset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert set(manifest["bundles"]) == {"core", "viewer"}
    for name, bundle in manifest["bundles"].items():
        assert bundle["url"].startswith("https://github.com/")
        assert len(bundle["sha256"]) == 64
        assert bundle["size_bytes"] > 0
        assert bundle["install_prefix"] == ("runtime" if name == "core" else "viewer")
        paths = [entry["path"] for entry in bundle["files"]]
        assert paths
        assert len(paths) == len(set(paths))
        assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
        assert all(len(entry["sha256"]) == 64 and entry["size_bytes"] > 0 for entry in bundle["files"])


def test_asset_manifest_includes_equipment_parts_and_menu_font():
    manifest = json.loads((RESOURCE_ROOT / "asset_manifest.json").read_text())
    files = {entry["path"] for entry in manifest["bundles"]["viewer"]["files"]}
    assert {"viewer/fc_player.parts", "viewer/fc_player.models",
            "viewer/data/fonts/runescape_bold.ttf",
            "viewer/data/sprites/items/item_28310.png"} <= files
    assert "viewer/data/sprites/items/item_25487.png" not in files


def test_fight_caves_sources_do_not_reference_local_development_trees():
    forbidden = ("/home/joe", "/v38/", "pufferlib_4", "runescape-reference")
    roots = (
        ENV_ROOT,
        RESOURCE_ROOT,
        REPO_ROOT / "config" / "fight_caves.ini",
    )
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            if path.suffix in {".png", ".bin", ".models", ".atlas", ".anims"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for value in forbidden:
                assert value not in text, f"{path} contains forbidden path marker {value!r}"

def fail(message: str) -> None:
    raise AssertionError(f"puffer_contract_test: {message}")


def puffer_main() -> int:
    """Exercise the compiled CPU interface when explicitly invoked as a script."""
    import ctypes
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from pufferlib import _C
    except ImportError as exc:
        raise RuntimeError(
            "Fight Caves Puffer backend is unavailable; run "
            "'./build.sh fight_caves --cpu' first"
        ) from exc
    if getattr(_C, "env_name", None) != "fight_caves":
        fail(f"backend was built for {getattr(_C, 'env_name', None)!r}")
    if getattr(_C, "gpu", None) != 0:
        fail("acceptance test requires the CPU backend")

    from pufferlib.pufferl import load_config

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["puffer_contract_test"]
        args = load_config("fight_caves")
    finally:
        sys.argv = previous_argv
    args["vec"].update(total_agents=8, num_buffers=1, num_threads=1)

    vec = _C.create_vec(args, 0)
    try:
        if vec.obs_size != 320:
            fail(f"expected 320 observations, got {vec.obs_size}")
        if vec.num_atns != 3:
            fail(f"expected 3 action heads, got {vec.num_atns}")
        if list(vec.act_sizes) != [17, 9, 8]:
            fail(f"unexpected action dimensions: {list(vec.act_sizes)}")
        if vec.obs_dtype != "FloatTensor" or vec.obs_elem_size != 4:
            fail(
                f"unexpected observation type: {vec.obs_dtype}/{vec.obs_elem_size}"
            )

        obs_storage = (ctypes.c_float * (vec.total_agents * vec.obs_size)).from_address(
            vec.obs_ptr
        )
        reward_storage = (ctypes.c_float * vec.total_agents).from_address(
            vec.rewards_ptr
        )
        terminal_storage = (ctypes.c_float * vec.total_agents).from_address(
            vec.terminals_ptr
        )
        observations = np.ctypeslib.as_array(obs_storage).reshape(
            vec.total_agents, vec.obs_size
        )
        rewards = np.ctypeslib.as_array(reward_storage)
        terminals = np.ctypeslib.as_array(terminal_storage)

        vec.reset()
        if not np.isfinite(observations).all():
            fail("reset observations contain non-finite values")
        mask = observations[:, -34:]
        if not np.logical_or(mask == 0.0, mask == 1.0).all():
            fail("float action mask contains a value other than zero or one")
        for start, stop in ((0, 17), (17, 26), (26, 34)):
            if not (mask[:, start:stop].sum(axis=1) >= 1).all():
                fail("an action head has no legal action")

        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)
        terminal_count = 0
        for _ in range(6000):
            vec.cpu_step(actions.ctypes.data)
            if not np.isfinite(observations).all():
                fail("step observations contain non-finite values")
            if not np.isfinite(rewards).all():
                fail("rewards contain non-finite values")
            if not np.logical_or(terminals == 0.0, terminals == 1.0).all():
                fail("terminal buffer contains a value other than zero or one")
            terminal_count += int(terminals.sum())
            if terminal_count:
                break
        if terminal_count == 0:
            fail("no terminal/autoreset boundary was observed")
    finally:
        vec.close()

    print(f"puffer_contract_test: passed ({terminal_count} terminal transitions)")
    return 0

if __name__ == "__main__":
    raise SystemExit(puffer_main())
