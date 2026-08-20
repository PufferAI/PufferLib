#!/usr/bin/env python3
"""Paired native-trainer throughput benchmarks with isolated artifacts."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_FILE = Path(__file__).with_name("native_cases.toml")
FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class Case:
    name: str
    binary: str
    env: str
    backend: str
    timesteps: int
    precondition_timesteps: int
    args: tuple[str, ...]
    description: str


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument(
        "--cases",
        default="affine",
        help="Comma-separated case or group names from native_cases.toml",
    )
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/tmp"),
        help="A new puffer-throughput-* campaign is created below this path",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Override the scored timestep budget for every selected case",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply and record a native configuration override to every run",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("exact", "record"),
        default="exact",
        help="Abort on checkpoint mismatch or only record it (default: exact)",
    )
    parser.add_argument("--boxoban-map", type=Path)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument(
        "--no-precondition",
        action="store_true",
        help="Skip one short unscored run per case and binary",
    )
    parser.add_argument(
        "--allow-foreign-gpu-processes",
        action="store_true",
        help="Record rather than reject pre-existing GPU compute processes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the campaign plan but do not execute binaries",
    )
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.timesteps is not None and args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.timeout < 0:
        parser.error("--timeout cannot be negative")
    for value in args.override:
        try:
            key, _ = split_override(value)
        except ValueError as error:
            parser.error(str(error))
        if not key:
            parser.error(f"Invalid empty override key: {value!r}")
    return args


def split_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Invalid case override without '=': {value!r}")
    return value.split("=", 1)


def merge_overrides(base: list[str], overrides: dict[str, Any]) -> tuple[str, ...]:
    merged: dict[str, str] = {}
    order: list[str] = []
    for value in base:
        key, setting = split_override(value)
        if key not in merged:
            order.append(key)
        merged[key] = setting
    for key, setting in overrides.items():
        if key not in merged:
            order.append(key)
        merged[key] = str(setting)
    return tuple(f"{key}={merged[key]}" for key in order)


def load_case_file(path: Path) -> tuple[dict[str, Case], dict[str, list[str]]]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    profiles = data.get("profiles", {})
    cases: dict[str, Case] = {}
    for name, raw in data.get("cases", {}).items():
        profile_name = raw.get("profile")
        profile_args: list[str] = []
        if profile_name is not None:
            if profile_name not in profiles:
                raise ValueError(f"Case {name!r} has unknown profile {profile_name!r}")
            profile_args = list(profiles[profile_name].get("args", []))
        profile_args.extend(raw.get("args", []))
        cases[name] = Case(
            name=name,
            binary=str(raw["binary"]),
            env=str(raw["env"]),
            backend=str(raw["backend"]),
            timesteps=int(raw["timesteps"]),
            precondition_timesteps=int(raw.get("precondition_timesteps", 8_388_608)),
            args=merge_overrides(profile_args, raw.get("overrides", {})),
            description=str(raw.get("description", "")),
        )
    groups = {name: list(values) for name, values in data.get("groups", {}).items()}
    return cases, groups


def select_cases(
    selection: str, cases: dict[str, Case], groups: dict[str, list[str]]
) -> list[Case]:
    selected: list[str] = []
    expanding: set[str] = set()

    def add(name: str) -> None:
        if name in cases:
            if name not in selected:
                selected.append(name)
            return
        if name not in groups:
            raise ValueError(f"Unknown case or group: {name!r}")
        if name in expanding:
            raise ValueError(f"Recursive case group: {name!r}")
        expanding.add(name)
        for child in groups[name]:
            add(child)
        expanding.remove(name)

    for item in selection.split(","):
        item = item.strip()
        if item:
            add(item)
    if not selected:
        raise ValueError("No benchmark cases selected")
    return [cases[name] for name in selected]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
        return {"command": command, "returncode": result.returncode, "output": result.stdout}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def foreign_gpu_processes() -> list[str]:
    result = capture_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.get("returncode") != 0:
        return []
    return [line.strip() for line in result.get("output", "").splitlines() if line.strip()]


def machine_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvidia_smi": capture_command(["nvidia-smi"]),
        "gpu_query": capture_command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": capture_command(["nvcc", "--version"]),
    }


def numeric_series(value: str) -> list[float]:
    return [float(match.group(0)) for match in FLOAT_RE.finditer(value)]


def find_run_ini(log_dir: Path, run_id: str) -> Path:
    candidates = list(log_dir.rglob("*.ini"))
    matching = [path for path in candidates if run_id in path.name]
    if len(matching) == 1:
        return matching[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"No INI log produced under {log_dir}")
    raise RuntimeError(
        f"Ambiguous INI logs under {log_dir}: " + ", ".join(str(path) for path in candidates)
    )


def find_final_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.rglob("*.bin"))
    if not candidates:
        raise RuntimeError(f"No checkpoint produced under {checkpoint_dir}")
    if len(candidates) == 1:
        return candidates[0]
    numeric = [path for path in candidates if path.stem.isdigit()]
    if numeric:
        return max(numeric, key=lambda path: int(path.stem))
    raise RuntimeError(
        f"Ambiguous checkpoints under {checkpoint_dir}: "
        + ", ".join(str(path) for path in candidates)
    )


def parse_metric_series(path: Path) -> dict[str, list[float]]:
    wanted = {"agent_steps", "uptime", "env/score", "perf", "SPS"}
    parsed: dict[str, list[float]] = {}
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        with path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
        for section in parser.sections():
            for key, value in parser.items(section):
                normalized = key.strip()
                if normalized in wanted:
                    parsed[normalized] = numeric_series(value)
    except configparser.Error:
        parsed = {}

    if "agent_steps" not in parsed or "uptime" not in parsed:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key in wanted:
                    parsed[key] = numeric_series(value)
    return parsed


def throughput_metrics(metrics: dict[str, list[float]]) -> dict[str, float | int]:
    steps = metrics.get("agent_steps", [])
    uptime = metrics.get("uptime", [])
    count = min(len(steps), len(uptime))
    if count < 2:
        raise RuntimeError(f"Need at least two step/uptime samples, found {count}")
    steps = steps[-count:]
    uptime = uptime[-count:]

    samples = []
    for elapsed, agent_steps in zip(uptime, steps):
        if samples and (elapsed, agent_steps) == samples[-1]:
            continue
        samples.append((elapsed, agent_steps))

    uptime = [elapsed for elapsed, _ in samples]
    steps = [agent_steps for _, agent_steps in samples]
    count = len(samples)
    if count < 2:
        raise RuntimeError("Fewer than two unique uptime samples were recorded")

    if any(b <= a for a, b in zip(uptime, uptime[1:])):
        raise RuntimeError("Uptime samples are not strictly increasing")
    if any(b < a for a, b in zip(steps, steps[1:])):
        raise RuntimeError("Agent-step samples are not monotonic")

    cutoff = steps[0] + 0.10 * (steps[-1] - steps[0])
    retained = [(x, y) for x, y in zip(uptime, steps) if y >= cutoff]
    if len(retained) < 2:
        raise RuntimeError(
            f"Need two samples after 10% warmup removal, found {len(retained)}"
        )
    xs = [item[0] for item in retained]
    ys = [item[1] for item in retained]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise RuntimeError("Cannot fit throughput slope from identical uptime samples")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in retained) / denominator
    full_delta = (steps[-1] - steps[0]) / (uptime[-1] - uptime[0])
    return {
        "steady_sps": slope,
        "native_full_sps": full_delta,
        "metric_samples": count,
        "retained_samples": len(retained),
    }


def command_for_run(
    binary: Path,
    case: Case,
    timesteps: int,
    run_id: str,
    log_dir: Path,
    checkpoint_dir: Path,
    boxoban_map: Path | None,
    run_overrides: tuple[str, ...],
) -> list[str]:
    overrides = list(case.args)
    for value in run_overrides:
        key, setting = split_override(value)
        overrides = list(merge_overrides(overrides, {key: setting}))
    overrides.extend(
        [
            f"train.total_timesteps={timesteps}",
            f"base.run_id={run_id}",
            f"base.log_dir={log_dir}",
            f"base.checkpoint_dir={checkpoint_dir}",
            "base.profile=0",
            "base.eval_episodes=1",
            "base.checkpoint_interval=2147483647",
            "sweep.downsample=64",
        ]
    )
    if case.env == "boxoban":
        if boxoban_map is None:
            raise ValueError("Boxoban requires --boxoban-map with a staged map binary")
    return [str(binary), "train", *(f"--{value}" for value in overrides)]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute_run(
    *,
    campaign: Path,
    runtime_root: Path,
    label: str,
    binary: Path,
    case: Case,
    timesteps: int,
    pair: int,
    sequence: int,
    phase: str,
    timeout: float,
    boxoban_map: Path | None,
    run_overrides: tuple[str, ...],
    allow_foreign: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_id = f"{case.name}_{phase}_p{pair:02d}_{label}_{sequence:03d}"
    run_dir = campaign / "runs" / run_id
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    command = command_for_run(
        binary,
        case,
        timesteps,
        run_id,
        log_dir,
        checkpoint_dir,
        boxoban_map,
        run_overrides,
    )
    foreign = foreign_gpu_processes()
    if foreign and not allow_foreign:
        raise RuntimeError(
            "Foreign GPU compute processes detected before run: " + "; ".join(foreign)
        )
    record: dict[str, Any] = {
        "case": case.name,
        "description": case.description,
        "backend": case.backend,
        "label": label,
        "phase": phase,
        "pair": pair,
        "sequence": sequence,
        "timesteps": timesteps,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "command": command,
        "command_shell": shlex.join(command),
        "foreign_gpu_processes_before": foreign,
        "started_unix": time.time(),
    }
    run_environment = os.environ.copy()
    if case.env == "boxoban":
        assert boxoban_map is not None
        run_environment["BOXOBAN_MAP_BIN"] = str(boxoban_map)
        record["boxoban_map"] = str(boxoban_map)
        record["boxoban_map_sha256"] = sha256(boxoban_map)
    (run_dir / "command.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if dry_run:
        record["dry_run"] = True
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=runtime_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=None if timeout == 0 else timeout,
        env=run_environment,
    )
    wall_seconds = time.perf_counter() - started
    (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    record.update(
        {
            "returncode": result.returncode,
            "wall_seconds": wall_seconds,
            "process_wall_sps": timesteps / wall_seconds,
            "finished_unix": time.time(),
            "foreign_gpu_processes_after": foreign_gpu_processes(),
        }
    )
    if result.returncode != 0:
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"{case.name} {label} exited {result.returncode}; see {run_dir}"
        )

    ini_path = find_run_ini(log_dir, run_id)
    metrics = parse_metric_series(ini_path)
    record.update(throughput_metrics(metrics))
    record["ini_path"] = str(ini_path)
    checkpoint_path = find_final_checkpoint(checkpoint_dir)
    record["checkpoint_path"] = str(checkpoint_path)
    record["checkpoint_sha256"] = sha256(checkpoint_path)
    record["checkpoint_bytes"] = checkpoint_path.stat().st_size
    (run_dir / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_ratios(ratios: list[float]) -> dict[str, float | int]:
    logs = [math.log(value) for value in ratios]
    rng = random.Random(73)
    bootstrap: list[float] = []
    for _ in range(10_000):
        sample = [logs[rng.randrange(len(logs))] for _ in logs]
        bootstrap.append(math.exp(statistics.fmean(sample)))
    bootstrap.sort()
    median = statistics.median(ratios)
    mad = statistics.median(abs(value - median) for value in ratios)
    return {
        "pairs": len(ratios),
        "geomean_speedup": math.exp(statistics.fmean(logs)),
        "median_speedup": median,
        "minimum_speedup": min(ratios),
        "mad": mad,
        "bootstrap_95_low": percentile(bootstrap, 0.025),
        "bootstrap_95_high": percentile(bootstrap, 0.975),
    }


def write_summary(campaign: Path, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[float]] = {}
    for pair in pairs:
        by_case.setdefault(pair["case"], []).append(pair["steady_speedup"])
    case_summary = {
        name: summarize_ratios(ratios) for name, ratios in sorted(by_case.items())
    }
    affine_summaries = [
        values for name, values in case_summary.items() if name.startswith("affine")
    ]
    all_logs = [math.log(pair["steady_speedup"]) for pair in pairs]
    summary = {
        "cases": case_summary,
        "suite_geomean_speedup": math.exp(statistics.fmean(all_logs)),
        "worst_pair_speedup": min(pair["steady_speedup"] for pair in pairs),
        "acceptance": {
            "no_pair_below_0.97": all(pair["steady_speedup"] >= 0.97 for pair in pairs),
            "all_checkpoints_match": all(pair["checkpoint_match"] for pair in pairs),
            "affine_ci_excludes_1": bool(affine_summaries)
            and all(values["bootstrap_95_low"] > 1.0 for values in affine_summaries),
        },
    }
    (campaign / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (campaign / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "pair",
                "baseline_steady_sps",
                "candidate_steady_sps",
                "steady_speedup",
                "baseline_wall_sps",
                "candidate_wall_sps",
                "wall_speedup",
                "checkpoint_match",
                "checkpoint_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)
    return summary


def main() -> int:
    args = parse_cli()
    cases_by_name, groups = load_case_file(args.case_file.resolve())
    cases = select_cases(args.cases, cases_by_name, groups)
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    runtime_root = args.runtime_root.resolve()
    boxoban_map = args.boxoban_map.resolve() if args.boxoban_map else None
    binaries: dict[str, dict[str, Path]] = {"baseline": {}, "candidate": {}}
    for case in cases:
        for label, directory in (
            ("baseline", baseline_dir),
            ("candidate", candidate_dir),
        ):
            binary = directory / case.binary
            if not binary.is_file():
                raise FileNotFoundError(f"Missing {label} binary for {case.name}: {binary}")
            if not os.access(binary, os.X_OK):
                raise PermissionError(f"Binary is not executable: {binary}")
            binaries[label][case.name] = binary
    if boxoban_map is not None and not boxoban_map.is_file():
        raise FileNotFoundError(f"Boxoban map does not exist: {boxoban_map}")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    campaign = Path(
        tempfile.mkdtemp(prefix="puffer-throughput-", dir=args.artifact_root.resolve())
    )
    campaign.chmod(0o700)
    metadata = {
        "created_unix": time.time(),
        "runtime_root": str(runtime_root),
        "case_file": str(args.case_file.resolve()),
        "case_file_sha256": sha256(args.case_file.resolve()),
        "selected_cases": [case.name for case in cases],
        "pairs": args.pairs,
        "timesteps_override": args.timesteps,
        "run_overrides": args.override,
        "checkpoint_policy": args.checkpoint_policy,
        "precondition": not args.no_precondition,
        "dry_run": args.dry_run,
        "machine": machine_metadata(),
    }
    (campaign / "campaign.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Artifacts: {campaign}", flush=True)

    sequence = 0
    raw_path = campaign / "runs.jsonl"
    if not args.no_precondition:
        for case in cases:
            for label in ("baseline", "candidate"):
                sequence += 1
                record = execute_run(
                    campaign=campaign,
                    runtime_root=runtime_root,
                    label=label,
                    binary=binaries[label][case.name],
                    case=case,
                    timesteps=case.precondition_timesteps,
                    pair=-1,
                    sequence=sequence,
                    phase="precondition",
                    timeout=args.timeout,
                    boxoban_map=boxoban_map,
                    run_overrides=tuple(args.override),
                    allow_foreign=args.allow_foreign_gpu_processes,
                    dry_run=args.dry_run,
                )
                append_jsonl(raw_path, record)

    paired_results: list[dict[str, Any]] = []
    for case in cases:
        scored_timesteps = args.timesteps or case.timesteps
        for pair_index in range(args.pairs):
            order = (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            )
            results: dict[str, dict[str, Any]] = {}
            for label in order:
                sequence += 1
                record = execute_run(
                    campaign=campaign,
                    runtime_root=runtime_root,
                    label=label,
                    binary=binaries[label][case.name],
                    case=case,
                    timesteps=scored_timesteps,
                    pair=pair_index,
                    sequence=sequence,
                    phase="scored",
                    timeout=args.timeout,
                    boxoban_map=boxoban_map,
                    run_overrides=tuple(args.override),
                    allow_foreign=args.allow_foreign_gpu_processes,
                    dry_run=args.dry_run,
                )
                append_jsonl(raw_path, record)
                results[label] = record
            if args.dry_run:
                continue
            baseline = results["baseline"]
            candidate = results["candidate"]
            pair_result = {
                "case": case.name,
                "pair": pair_index,
                "baseline_steady_sps": baseline["steady_sps"],
                "candidate_steady_sps": candidate["steady_sps"],
                "steady_speedup": candidate["steady_sps"] / baseline["steady_sps"],
                "baseline_wall_sps": baseline["process_wall_sps"],
                "candidate_wall_sps": candidate["process_wall_sps"],
                "wall_speedup": candidate["process_wall_sps"]
                / baseline["process_wall_sps"],
                "checkpoint_match": baseline["checkpoint_sha256"]
                == candidate["checkpoint_sha256"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
            }
            paired_results.append(pair_result)
            if not pair_result["checkpoint_match"]:
                failure = {
                    "case": case.name,
                    "pair": pair_index,
                    "baseline": baseline["checkpoint_sha256"],
                    "candidate": candidate["checkpoint_sha256"],
                    "baseline_path": baseline["checkpoint_path"],
                    "candidate_path": candidate["checkpoint_path"],
                }
                append_jsonl(campaign / "checkpoint_mismatches.jsonl", failure)
                if args.checkpoint_policy == "record":
                    continue
                failure_path = campaign / "CHECKPOINT_MISMATCH.json"
                failure_path.write_text(
                    json.dumps(failure, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"Checkpoint mismatch for {case.name} pair {pair_index}; "
                    f"see {failure_path}"
                )

    if args.dry_run:
        print(f"Dry-run plan retained at {campaign}")
        return 0
    summary = write_summary(campaign, paired_results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts retained at {campaign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
