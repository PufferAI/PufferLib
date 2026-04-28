"""Compare convergence of Craftax Classic vs Full on overlapping achievements.

Runs both envs through `uv run puffer train` back-to-back (default 10M env
steps each), then parses pufferlib's per-run JSON log and plots:
  - mean episode score over env steps
  - per-achievement unlock rate (for the 22 Classic-compatible achievements)
  - wall-clock time to reach each score threshold

The envs share the first 22 achievement IDs (Classic's entire set). Full
has 67 achievements total; the extra 45 are plotted separately so Full
isn't rewarded twice for reaching the same tier.

Usage:
    uv run python tests/craftax_convergence_bench.py --timesteps 10_000_000
    uv run python tests/craftax_convergence_bench.py --skip-train --plot-only
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs"

CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]

SCORE_THRESHOLDS = [1, 3, 5, 7, 10, 15]


def train(env_name, timesteps):
    env_log_dir = LOG_DIR / env_name
    env_log_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in env_log_dir.glob("*.json")}

    # pufferlib._C is compiled for one env at a time; rebuild before each run.
    build_cmd = [
        "uv", "run", "--with", "pybind11", "--with", "rich_argparse",
        "./build.sh", env_name,
    ]
    print(f"\n=== rebuilding pufferlib._C for {env_name} ===")
    subprocess.check_call(build_cmd, cwd=REPO)

    cmd = [
        "uv", "run", "--with", "pybind11", "--with", "rich_argparse",
        "puffer", "train", env_name,
        "--train.total-timesteps", str(int(timesteps)),
    ]
    print(f"\n=== training {env_name} for {timesteps:,} steps ===")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)
    after = {p.name for p in env_log_dir.glob("*.json")}
    new = sorted(after - before)
    if not new:
        raise RuntimeError(f"no new log file under {env_log_dir}")
    return env_log_dir / new[-1]


def load_run(path):
    with open(path) as f:
        raw = json.load(f)
    m = raw["metrics"]
    steps = np.array(m["agent_steps"], dtype=np.float64)
    uptime = np.array(m["uptime"], dtype=np.float64)
    score = np.array(m.get("env/score", [np.nan] * len(steps)), dtype=np.float64)
    ach = {}
    for name in CLASSIC_ACHIEVEMENTS:
        key = f"env/{name}"
        if key in m:
            ach[name] = np.array(m[key], dtype=np.float64)
    return {"steps": steps, "uptime": uptime, "score": score, "ach": ach, "path": str(path)}


def time_to_threshold(steps, score, threshold):
    above = np.nonzero(score >= threshold)[0]
    if len(above) == 0:
        return None
    return float(steps[above[0]])


def print_summary(label, run):
    print(f"\n--- {label} ({run['path']}) ---")
    total_steps = int(run["steps"][-1])
    wall = run["uptime"][-1]
    peak = float(np.nanmax(run["score"])) if run["score"].size else float("nan")
    final = float(run["score"][-1]) if run["score"].size else float("nan")
    print(f"total env steps: {total_steps:,}   wall: {wall/60:.1f}min   "
          f"final score: {final:.2f}   peak: {peak:.2f}")
    print(f"time to score threshold (env steps):")
    for t in SCORE_THRESHOLDS:
        s = time_to_threshold(run["steps"], run["score"], t)
        if s is None:
            print(f"  >={t:>2}: NOT REACHED")
        else:
            wall_at = run["uptime"][np.nonzero(run["score"] >= t)[0][0]]
            print(f"  >={t:>2}: {int(s):>12,} steps ({wall_at/60:5.1f} min)")
    if run["ach"]:
        print("final per-achievement unlock rate (mean over eval episodes):")
        for name in CLASSIC_ACHIEVEMENTS:
            if name in run["ach"]:
                print(f"  {name:<22s} {run['ach'][name][-1]:.3f}")


def plot(runs, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); skipping plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, run in runs.items():
        axes[0].plot(run["steps"] / 1e6, run["score"], label=label)
    axes[0].set_xlabel("env steps (M)")
    axes[0].set_ylabel("mean episode score (achievements)")
    axes[0].set_title("Convergence: score vs env steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for label, run in runs.items():
        axes[1].plot(run["uptime"] / 60, run["score"], label=label)
    axes[1].set_xlabel("wall time (min)")
    axes[1].set_ylabel("mean episode score")
    axes[1].set_title("Convergence: score vs wall time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nwrote plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=float, default=10_000_000,
                    help="env steps per training run")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training; use most recent log in logs/{env}")
    ap.add_argument("--classic-log", type=str, default=None,
                    help="explicit path to craftax_classic log json")
    ap.add_argument("--full-log", type=str, default=None,
                    help="explicit path to craftax log json")
    ap.add_argument("--out", type=str, default="craftax_convergence.png")
    args = ap.parse_args()

    runs = {}
    for label, env_name, override in [
        ("Classic", "craftax_classic", args.classic_log),
        ("Full",    "craftax",         args.full_log),
    ]:
        if override:
            path = Path(override)
        elif args.skip_train:
            candidates = sorted((LOG_DIR / env_name).glob("*.json"))
            if not candidates:
                print(f"no logs for {env_name} under {LOG_DIR/env_name}; skipping.")
                continue
            path = candidates[-1]
        else:
            path = train(env_name, args.timesteps)
        runs[label] = load_run(path)
        print_summary(label, runs[label])

    if len(runs) >= 1:
        plot(runs, args.out)


if __name__ == "__main__":
    main()
