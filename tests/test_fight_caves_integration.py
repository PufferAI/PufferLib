"""Fight Caves maintenance suite: assets, native C/C++ and CPU replay.

Run python3 -m pytest -q tests/test_fight_caves_integration.py.
Set FC_TEST_RENDER=1 to require the graphical checks too (DISPLAY or xvfb-run).
Native prerequisites are required, never silently skipped. GPU training is
reserved for the explicit disposable-checkout acceptance runner.
"""
from __future__ import annotations

import configparser
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "ocean" / "fight_caves"
RESOURCE_ROOT = REPO_ROOT / "resources" / "fight_caves"
TOOLS = ENV_ROOT / "tools.py"
E2E = REPO_ROOT / "tests" / "fight_caves_e2e.py"


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_command(label, *args, timeout=180, env=None):
    result = subprocess.run(list(map(str, args)), cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=timeout, env=env)
    assert result.returncode == 0, f"{label}:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    path = tmp_path_factory.mktemp("fight-caves-cpu") / "fight_caves"
    native_command("CPU build", "./build.sh", "fight_caves", path, "--cpu", timeout=600)
    return path


@pytest.fixture(scope="module")
def native_suites(executable, tmp_path_factory):
    output = tmp_path_factory.mktemp("fight-caves-native")
    return load_module(E2E).build_native_tests(REPO_ROOT, output, native_command)


@pytest.mark.parametrize("suite", ["simulation", "linkage"])
def test_native_suite(native_suites, suite):
    native_command(suite, native_suites[suite])


@pytest.mark.skipif(os.environ.get("FC_TEST_RENDER") != "1",
                    reason="Set FC_TEST_RENDER=1 to require graphical validation")
def test_graphical_suite(native_suites, tmp_path):
    prefix = ["xvfb-run", "-a"] if shutil.which("xvfb-run") else []
    assert prefix or os.environ.get("DISPLAY"), "Graphical tests require DISPLAY or xvfb-run"
    screenshot = tmp_path / "equipment-appearance.png"
    env = dict(os.environ, FC_TEST_OUTPUT=str(screenshot))
    native_command("graphics", *prefix, native_suites["simulation"], "--render", env=env)
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def e2e():
    return load_module(E2E)


def test_e2e_requires_disposable_checkout(e2e, tmp_path):
    (tmp_path / "build").mkdir()
    with pytest.raises(RuntimeError, match="use clean-clone"):
        e2e.Acceptance(tmp_path).execute()
    assert not (tmp_path / "build/fight-caves-e2e").exists()


def test_e2e_rejects_dirty_source_and_wrong_overlay(e2e, monkeypatch, tmp_path):
    monkeypatch.setattr(e2e, "git", lambda *args: b" M source.c\n")
    with pytest.raises(RuntimeError, match="source is dirty"):
        e2e.prepare_clone(str(tmp_path), "HEAD", False)
    with pytest.raises(RuntimeError, match="only supports this repository at HEAD"):
        e2e.prepare_clone(str(tmp_path), "HEAD", True)


def test_e2e_expected_failure_needs_nonzero_and_explanation(e2e, tmp_path):
    test = e2e.Acceptance(tmp_path)
    test.output.mkdir(parents=True)
    test.run("expected", sys.executable, "-c", "raise SystemExit('reason')", expected="reason")
    with pytest.raises(RuntimeError, match="failed acceptance"):
        test.run("unexpected-pass", sys.executable, "-c", "print('reason')", expected="reason")
    with pytest.raises(RuntimeError, match="failed acceptance"):
        test.run("unexplained", sys.executable, "-c", "raise SystemExit(1)", expected="reason")


@pytest.mark.parametrize("fault", [None, "nan", "missing", "steps", "length", "alias"])
def test_e2e_checks_saved_metric_histories(e2e, tmp_path, fault):
    metrics = {f"env/{key}": "0,0" for key in e2e.ENV_METRICS}
    metrics.update({"env/perf": "0,0", "env/score": "0,0", "agent_steps": "1024,2048"})
    if fault == "nan":
        metrics["env/wave_reached"] = "0,nan"
    elif fault == "missing":
        del metrics["env/wave_reached"]
    elif fault == "steps":
        metrics["agent_steps"] = "1024,1024"
    elif fault == "length":
        metrics["env/wave_reached"] = "0"
    elif fault == "alias":
        metrics["env/perf"] = "1,1"
    path = tmp_path / "run.ini"
    path.write_text("[metrics]\n" + "\n".join(f"{key}={value}" for key, value in metrics.items()))
    if fault:
        with pytest.raises(RuntimeError):
            e2e.read_run(path, 2048)
    else:
        assert e2e.read_run(path, 2048)[1]["agent_steps"][-1] == 2048


def make_bundle(directory: Path, members: dict[str, bytes]) -> dict:
    """Write a .tar.gz holding these members and return its manifest entry."""
    archive = directory / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tar.addfile(info, BytesIO(contents))
    return {"archive": archive.name, "url": archive.as_uri(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "install_prefix": "runtime"}


@pytest.fixture
def tools(tmp_path, monkeypatch):
    module = load_module(TOOLS)
    monkeypatch.setattr(module, "RESOURCE_ROOT", tmp_path / "resources")
    module.RESOURCE_ROOT.mkdir()
    return module


def test_install_replaces_the_directory_with_the_verified_archive(tools, tmp_path):
    old = tools.RESOURCE_ROOT / "runtime" / "old.map"
    old.parent.mkdir()
    old.write_bytes(b"previous install")
    bundle = make_bundle(tmp_path, {"runtime/test.map": b"arena data"})
    tools.install_bundle(bundle)
    runtime = tools.RESOURCE_ROOT / "runtime"
    assert (runtime / "test.map").read_bytes() == b"arena data"
    assert not old.exists()
    assert (runtime / tools.STAMP).read_text() == bundle["sha256"] + "\n"
    assert [path.name for path in tools.RESOURCE_ROOT.iterdir()] == ["runtime"]


@pytest.mark.parametrize("fault,message", [
    ("hash", "checksum mismatch"), ("download", "download failed"),
])
def test_failed_update_leaves_the_existing_install(tools, tmp_path, fault, message):
    bundle = make_bundle(tmp_path, {"runtime/test.map": b"installed"})
    tools.install_bundle(bundle)
    if fault == "hash":
        bundle["sha256"] = "0" * 64
    else:
        bundle["url"] = (tmp_path / "missing.tar.gz").as_uri()
    with pytest.raises(tools.AssetError, match=message):
        tools.install_bundle(bundle)
    assert (tools.RESOURCE_ROOT / "runtime/test.map").read_bytes() == b"installed"
    assert [path.name for path in tools.RESOURCE_ROOT.iterdir()] == ["runtime"]


def test_unsafe_archive_path_is_rejected(tools, tmp_path):
    bundle = make_bundle(tmp_path, {"runtime/test.map": b"map", "../../escaped": b"x"})
    with pytest.raises(tools.AssetError, match="outside the destination"):
        tools.install_bundle(bundle)
    assert list(tools.RESOURCE_ROOT.iterdir()) == []


def test_setup_reuses_a_matching_stamp(tools, tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bundles": [make_bundle(tmp_path, {"runtime/a": b"a"})]}))
    monkeypatch.setattr(tools, "MANIFEST", manifest)
    downloads = []
    real_download = tools.download
    def counting_download(url, destination):
        downloads.append(url)
        real_download(url, destination)
    monkeypatch.setattr(tools, "download", counting_download)

    assert tools.main(["setup", "--verify-only"]) == 1
    assert "runtime/ is missing" in capsys.readouterr().err
    assert tools.main(["setup"]) == 0
    assert tools.main(["setup"]) == 0
    assert tools.main(["setup", "--verify-only"]) == 0
    assert len(downloads) == 1
    (tools.RESOURCE_ROOT / "runtime" / tools.STAMP).write_text("0" * 64)
    assert tools.main(["setup", "--verify-only"]) == 1
    assert tools.main(["setup"]) == 0
    assert tools.main(["setup", "--force"]) == 0
    assert len(downloads) == 3


def test_config_contains_only_supported_native_settings():
    default = configparser.ConfigParser(interpolation=None)
    default.read(REPO_ROOT / "config" / "default.ini")
    config = configparser.ConfigParser(interpolation=None)
    config.read(REPO_ROOT / "config" / "fight_caves.ini")
    assert set(config.sections()) == {"base", "env", "vec", "train", "policy", "sweep"}
    binding = (ENV_ROOT / "binding.c").read_text()
    core = (ENV_ROOT / "core.h").read_text()
    env_keys = set(re.findall(r'kwargs,\s*"([^"]+)"', binding))
    reward_keys = set(re.findall(r'\{"(\w+)",\s*offsetof\(FcRewardParams,', core))
    assert set(config["env"]) <= env_keys | reward_keys
    for section in config.sections():
        if section == "env":
            continue
        assert set(config[section]) <= set(default[section]), section


def test_asset_manifest_pins_both_bundles():
    manifest = json.loads((RESOURCE_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
    assert [bundle["install_prefix"] for bundle in manifest["bundles"]] == ["runtime", "viewer"]
    for bundle in manifest["bundles"]:
        assert bundle["url"].startswith("https://github.com/")
        assert bundle["url"].endswith("/" + bundle["archive"])
        assert re.fullmatch("[0-9a-f]{64}", bundle["sha256"])


def test_installed_viewer_bundle_has_equipment_parts_and_menu_font(executable):
    viewer = RESOURCE_ROOT / "viewer"  # installed by the executable fixture's build
    for name in ("fc_player.parts", "fc_player.models", "data/fonts/runescape_bold.ttf",
                 "data/sprites/items/item_28310.png"):
        assert (viewer / name).is_file(), name
    assert not (viewer / "data/sprites/items/item_25487.png").exists()


def test_fight_caves_sources_do_not_reference_local_paths():
    forbidden = ("/home/", "/Users/")
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


FLOATS = 8 * (320 + 35) + 2 * 3 * 8 * 8

def run(executable, *args):
    return subprocess.run(
        [str(executable), *map(str, args)], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )


def check(executable, path, *args):
    return run(executable, "check", path, "--policy.hidden_size=8",
               "--policy.num_layers=2", *args)


def test_valid_cpu_checkpoint_and_two_episodes(executable, tmp_path):
    path = tmp_path / "valid.bin"
    path.write_bytes(b"\0" * (FLOATS * 4))
    result = check(executable, path)
    assert result.returncode == 0, result.stderr
    assert f"file_floats={FLOATS}" in result.stdout
    result = run(executable, "eval", path, "--headless", "--policy.hidden_size=8",
                 "--policy.num_layers=2", "--base.eval_episodes=2")
    assert result.returncode == 0, result.stderr
    assert "CPU_EVAL env=fight_caves" in result.stdout
    assert "games=2 " in result.stdout


def test_manual_viewer_without_display_exits_cleanly(executable):
    result = subprocess.run([str(executable)], cwd=REPO_ROOT,
                            env=dict(os.environ, DISPLAY=""),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    assert "no graphical DISPLAY is configured" in result.stderr
    assert "Initializing raylib" not in result.stdout


@pytest.mark.parametrize("kind", ["missing", "short", "long", "zip", "nan", "inf", "shape"])
def test_invalid_checkpoints_fail_without_opening_viewer(executable, tmp_path, kind):
    path = tmp_path / "invalid.bin"
    data = b"\0" * (FLOATS * 4)
    if kind == "short":
        data = data[:-1]
    elif kind == "long":
        data += b"\0"
    elif kind == "zip":
        data = b"PK\x03\x04pytorch"
    elif kind in ("nan", "inf"):
        data = struct.pack("f", float(kind)) + data[4:]
    elif kind == "shape":
        data = data * 2
    if kind != "missing":
        path.write_bytes(data)
    for mode in ("check", "eval"):
        result = run(executable, mode, path, "--policy.hidden_size=8", "--policy.num_layers=2")
        assert result.returncode != 0
        assert "checkpoint rejected:" in result.stderr
        assert "Fight Caves Viewer" not in result.stdout + result.stderr


@pytest.mark.parametrize("args", [
    [], ["latest"], ["missing.bin", "--policy.hidden_size=7"],
    ["missing.bin", "--policy.hidden_size=999999999999"],
    ["missing.bin", "--policy.hidden_size=8.5"],
    ["missing.bin", "--policy.num_layers=0"],
    ["missing.bin", "--headless", "--base.eval_episodes=0"],
    ["missing.bin", "--base.seed=-1"], ["missing.bin", "--wandb"],
])
def test_invalid_cpu_replay_requests_fail(executable, args):
    result = run(executable, "eval", *args)
    assert result.returncode != 0
    assert result.stderr
    assert "CPU_EVAL" not in result.stdout
