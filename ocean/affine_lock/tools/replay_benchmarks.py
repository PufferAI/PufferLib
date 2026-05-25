#!/usr/bin/env python3
"""Replay fixed affine_lock benchmark profiles and append local history."""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "ocean/affine_lock/benchmarks/affine2_replay_manifest.json"
PROFILE_DIR = ROOT / "config/profiles"
LOG_DIR = ROOT / "logs/affine_lock"

METRIC_KEYS = [
    "env/perf",
    "env/score",
    "env/solve_rate",
    "env/solved_min_win_moves",
    "env/depth_4_rate",
    "env/depth_4_solve_rate",
    "env/depth_8_rate",
    "env/depth_8_solve_rate",
    "env/depth_16_rate",
    "env/depth_16_solve_rate",
    "env/max_depth_solve",
    "SPS",
    "uptime",
    "agent_steps",
]

METRIC_ALIASES = {
    "env/solved_min_win_moves": ["env/solved_target_distance"],
}


def parse_ini_value(value: str):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def profile_path(profile: str) -> Path:
    return PROFILE_DIR / f"{profile}.ini"


def load_profile(profile: str) -> configparser.ConfigParser:
    path = profile_path(profile)
    parser = configparser.ConfigParser()
    read_paths = parser.read(path)
    if str(path) not in read_paths:
        raise ValueError(f"failed to read profile {path}")
    return parser


def values_match(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(actual, int) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    return str(actual) == str(expected)


def validate_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("env_name") != "affine_lock":
        errors.append("manifest env_name must be affine_lock")

    thresholds = manifest.get("perf_thresholds", [])
    if thresholds != sorted(thresholds) or not thresholds:
        errors.append("perf_thresholds must be a nonempty sorted list")

    seen: set[str] = set()
    for bench in manifest.get("benchmarks", []):
        run_id = bench.get("run_id")
        profile = bench.get("profile")
        if not run_id or run_id in seen:
            errors.append(f"duplicate or missing run_id: {run_id}")
            continue
        seen.add(run_id)

        path = profile_path(profile)
        if not path.exists():
            errors.append(f"{run_id}: missing profile {path}")
            continue

        baseline = bench.get("baseline", {})
        if baseline.get("max_perf", 0) < 0.1:
            errors.append(f"{run_id}: baseline max_perf below 0.1")
        if baseline.get("uptime", 9999) > 240:
            errors.append(f"{run_id}: baseline uptime above 4 minutes")
        if baseline.get("avg_sps", 0) < 900_000:
            errors.append(f"{run_id}: baseline avg_sps below 900K")

        try:
            profile_config = load_profile(profile)
        except ValueError as err:
            errors.append(str(err))
            continue

        for section, expected_values in bench.get("config", {}).items():
            if section not in profile_config:
                errors.append(f"{run_id}: missing profile section [{section}]")
                continue
            for key, expected in expected_values.items():
                if key not in profile_config[section]:
                    errors.append(f"{run_id}: missing profile key {section}.{key}")
                    continue
                actual = parse_ini_value(profile_config[section][key])
                if not values_match(actual, expected):
                    errors.append(
                        f"{run_id}: {section}.{key} expected {expected!r}, "
                        f"got {actual!r}")

        if profile_config.get("base", "env_name", fallback=None) != "affine_lock":
            errors.append(f"{run_id}: profile base.env_name must be affine_lock")
        if parse_ini_value(profile_config.get("sweep", "downsample", fallback="0")) < 50:
            errors.append(f"{run_id}: profile sweep.downsample should be >= 50")

    if len(seen) != len(manifest.get("benchmarks", [])):
        errors.append("benchmark run_ids are not unique")
    return errors


def benchmark_by_id(manifest: dict) -> dict[str, dict]:
    return {bench["run_id"]: bench for bench in manifest["benchmarks"]}


def print_benchmark_list(manifest: dict) -> None:
    print("run_id    role                         max_perf  avg_sps   uptime  profile")
    for bench in manifest["benchmarks"]:
        baseline = bench["baseline"]
        print(
            f"{bench['run_id']:<9} "
            f"{bench['role']:<28} "
            f"{baseline['max_perf']:.4f}    "
            f"{baseline['avg_sps'] / 1_000_000:.2f}M   "
            f"{baseline['uptime']:.1f}s  "
            f"{bench['profile']}")


def git_info() -> dict:
    def run_git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    tracked_status = run_git("status", "--short", "--untracked-files=no") or ""
    full_status = run_git("status", "--short") or ""
    untracked_count = sum(1 for line in full_status.splitlines() if line.startswith("?? "))
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(tracked_status),
        "untracked_count": untracked_count,
    }


def metric_series(metrics: dict, key: str) -> list[float]:
    values = metrics.get(key, [])
    if not isinstance(values, list):
        for alias in METRIC_ALIASES.get(key, []):
            values = metrics.get(alias, [])
            if isinstance(values, list):
                break
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def final_metric(metrics: dict, key: str) -> float | None:
    values = metric_series(metrics, key)
    return values[-1] if values else None


def first_threshold_crossings(metrics: dict, thresholds: list[float]) -> dict[str, dict | None]:
    perf = metric_series(metrics, "env/perf")
    uptime = metric_series(metrics, "uptime")
    steps = metric_series(metrics, "agent_steps")
    out: dict[str, dict | None] = {}
    for threshold in thresholds:
        key = f"perf_{threshold:g}"
        out[key] = None
        for idx, value in enumerate(perf):
            if value < threshold:
                continue
            out[key] = {
                "perf": value,
                "uptime": uptime[idx] if idx < len(uptime) else None,
                "agent_steps": steps[idx] if idx < len(steps) else None,
            }
            break
    return out


def summarize_log(log_path: Path, thresholds: list[float]) -> dict:
    with log_path.open("r", encoding="utf-8") as file:
        log = json.load(file)
    metrics = log["metrics"]
    perf = metric_series(metrics, "env/perf")
    sps = metric_series(metrics, "SPS")
    summary = {
        "log_path": str(log_path.relative_to(ROOT)),
        "final": {key: final_metric(metrics, key) for key in METRIC_KEYS},
        "max_perf": max(perf) if perf else None,
        "avg_logged_sps": sum(sps) / len(sps) if sps else None,
        "thresholds": first_threshold_crossings(metrics, thresholds),
    }
    return summary


def newest_log_after(before: set[Path]) -> Path:
    after = set(LOG_DIR.glob("*.json"))
    new_logs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not new_logs:
        raise RuntimeError(f"no new PufferLib log found in {LOG_DIR}")
    return new_logs[-1]


def print_tail(path: Path, lines: int = 80) -> None:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in data[-lines:]:
        print(line)


def run_one(
        bench: dict,
        manifest: dict,
        history_path: Path,
        dry_run: bool,
        quiet: bool) -> dict | None:
    profile = bench["profile"]
    command = [
        sys.executable,
        "-m",
        "pufferlib.pufferl",
        "train",
        "affine_lock",
        "--config-profile",
        profile,
    ]
    print(" ".join(command), flush=True)
    if dry_run:
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    before = set(LOG_DIR.glob("*.json"))
    started = time.perf_counter()
    output_path = None
    if quiet:
        output_path = Path("/tmp") / (
            f"affine_lock_bench_{bench['run_id']}_{int(time.time())}.log")
        with output_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        if output_path is not None:
            print_tail(output_path)
        raise RuntimeError(f"{bench['run_id']} failed with exit code {completed.returncode}")

    log_path = newest_log_after(before)
    summary = summarize_log(log_path, manifest["perf_thresholds"])
    baseline = bench["baseline"]
    final = summary["final"]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": bench["run_id"],
        "profile": profile,
        "role": bench["role"],
        "source_project": manifest["source_project"],
        "git": git_info(),
        "command": command,
        "output_log": str(output_path) if output_path is not None else None,
        "wall_seconds": wall_seconds,
        "baseline": baseline,
        "result": summary,
        "delta": {
            "wall_vs_baseline_uptime": wall_seconds - baseline["uptime"],
            "final_uptime_vs_baseline": (
                final["uptime"] - baseline["uptime"]
                if final["uptime"] is not None else None
            ),
            "max_perf_vs_baseline": (
                summary["max_perf"] - baseline["max_perf"]
                if summary["max_perf"] is not None else None
            ),
            "final_perf_vs_baseline": (
                final["env/perf"] - baseline["final_perf"]
                if final["env/perf"] is not None else None
            ),
            "avg_logged_sps_vs_baseline": (
                summary["avg_logged_sps"] - baseline["avg_sps"]
                if summary["avg_logged_sps"] is not None else None
            ),
        },
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        f"{bench['run_id']}: max_perf={summary['max_perf']:.4f} "
        f"baseline={baseline['max_perf']:.4f} "
        f"wall={wall_seconds:.1f}s baseline={baseline['uptime']:.1f}s "
        f"log={summary['log_path']}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
        help="Validate replay manifest and profile files without training")
    parser.add_argument("--list", action="store_true", help="List benchmark profiles")
    parser.add_argument("--run", action="append", default=[],
        help="Run one benchmark id. Can be passed more than once")
    parser.add_argument("--all", action="store_true", help="Run all replay benchmarks")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--quiet", action="store_true",
        help="Write training console output to /tmp instead of stdout")
    parser.add_argument("--history", type=Path, default=None,
        help="JSONL history path. Defaults to manifest history_path")
    args = parser.parse_args()

    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.check:
        print(f"validated {len(manifest['benchmarks'])} affine replay benchmarks")
        return 0

    if args.list or (not args.run and not args.all):
        print_benchmark_list(manifest)
        return 0

    by_id = benchmark_by_id(manifest)
    selected = manifest["benchmarks"] if args.all else []
    for run_id in args.run:
        if run_id not in by_id:
            print(f"unknown benchmark run_id: {run_id}", file=sys.stderr)
            return 1
        selected.append(by_id[run_id])

    history_path = args.history or ROOT / manifest["history_path"]
    for _ in range(args.repeat):
        for bench in selected:
            run_one(bench, manifest, history_path, args.dry_run, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
