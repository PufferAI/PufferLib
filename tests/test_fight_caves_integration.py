"""Asset installation and PufferLib integration tests, including the C suite.

Run ./build.sh fight_caves --cpu, then:
    python -m pytest -q tests/test_fight_caves_integration.py
Native/backend checks skip with a setup instruction if prerequisites are absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
import os
import shlex
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DATA = REPO_ROOT / "ocean" / "fight_caves" / "tools.py"
RESOURCE_ROOT = REPO_ROOT / "resources" / "fight_caves"


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

    installed.write_bytes(b"x" * len(payload))
    assert "checksum mismatch" in setup_data.verify_tree(install_root, bundle, exact=True)[0]
    setup_data.install_bundle("core", bundle, force=False)
    assert installed.read_bytes() == payload

    def unexpected_download(*args):
        pytest.fail("Valid installed assets must be reusable offline")
    monkeypatch.setattr(setup_data, "download", unexpected_download)
    setup_data.install_bundle("core", bundle, force=False)


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


def test_manifest_rejects_unsafe_and_duplicate_file_paths():
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


@pytest.fixture(scope="module")
def installed_runtime():
    setup = load_setup_data()
    bundle = setup.load_manifest(setup.DEFAULT_MANIFEST)["bundles"]["core"]
    if not (RESOURCE_ROOT / "runtime").is_dir():
        pytest.skip("Run ./build.sh fight_caves --cpu to install runtime assets")
    errors = setup.verify_tree(RESOURCE_ROOT, bundle, exact=False)
    assert not errors, f"Rebuild Fight Caves to repair runtime assets: {errors}"


def test_native_simulation(tmp_path, installed_runtime):
    """Build/run deterministic simulation and headless adapter checks."""
    compiler = shlex.split(os.environ.get("CC", "clang"))
    raylib = REPO_ROOT / ("raylib-5.5_macos" if sys.platform == "darwin"
                          else "raylib-5.5_linux_amd64")
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("Install a C compiler (clang, or set CC) to run native tests")
    if not (raylib / "lib/libraylib.a").is_file():
        pytest.skip("Run ./build.sh fight_caves --cpu to install Raylib")
    binary = tmp_path / "fight_caves_simulation"
    command = [*compiler, "-std=c11", "-O2", "-Wall",
               "-Iocean/fight_caves", f"-I{raylib / 'include'}",
               "tests/test_fight_caves_simulation.c", str(raylib / "lib/libraylib.a"),
               "-lm", "-lpthread", "-o", str(binary)]
    if sys.platform == "darwin":
        for framework in ("Cocoa", "IOKit", "CoreVideo", "OpenGL"):
            command += ["-framework", framework]
    else:
        command += ["-ldl", "-lGL"]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    environment = os.environ.copy()
    for kind in ("collision", "movement", "los"):
        environment[f"FC_{kind.upper()}_PATH"] = str(
            RESOURCE_ROOT / "runtime" / f"fightcaves.{kind}")
    subprocess.run([str(binary)], cwd=REPO_ROOT, env=environment, check=True)


def fail(message: str) -> None:
    raise AssertionError(f"puffer_contract_test: {message}")


def test_puffer_backend(monkeypatch, installed_runtime):
    """Check vectorized observations, actions, autoreset and episode logs."""
    import ctypes
    import numpy as np

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.chdir(REPO_ROOT)
    _C = pytest.importorskip("pufferlib._C", reason="Run ./build.sh fight_caves --cpu")
    if getattr(_C, "env_name", None) != "fight_caves" or getattr(_C, "gpu", None) != 0:
        pytest.skip("Run ./build.sh fight_caves --cpu to select the Fight Caves CPU backend")
    for kind in ("collision", "movement", "los"):
        monkeypatch.setenv(f"FC_{kind.upper()}_PATH", str(
            RESOURCE_ROOT / "runtime" / f"fightcaves.{kind}"))

    from pufferlib.pufferl import load_config

    monkeypatch.setattr(sys, "argv", ["fight_caves_test"])
    args = load_config("fight_caves")
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
        metrics = vec.log()
        expected_metrics = {
            "zero_progress_ticks", "wave_reached", "wrong_prayer_hits",
            "reached_wave_63", "jad_kill_rate", "prayer_uptime_range",
            "prayer_uptime_melee", "prayer_uptime_magic", "npc_healing_total",
            "jad_healing_total", "episode_length", "n",
        }
        if set(metrics) != expected_metrics:
            fail(f"unexpected episode metrics: {set(metrics) ^ expected_metrics}")
        if metrics["n"] != terminal_count or metrics["episode_length"] <= 0:
            fail("completed episode metrics were lost during autoreset")
        if not all(np.isfinite(value) for value in metrics.values()):
            fail("episode metrics contain non-finite values")
        if vec.log():
            fail("episode metrics were not drained after logging")
    finally:
        vec.close()

    print(f"puffer_contract_test: passed ({terminal_count} terminal transitions)")
