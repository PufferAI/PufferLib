"""Freeze and evaluate the ten-pole C controller on C RK4 and MuJoCo.

Both engines receive identical explicit starts from the C reset generator.
Every outcome and complete/partial recording is retained. This does not modify
the historical Python controller or its final cohorts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial(task):
    engine, seed, library, model, out = task
    from test_npole import independent_mjcf
    import mujoco as mj
    lib = ctypes.CDLL(library)
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.np_test_load.argtypes = [ctypes.c_char_p]
    lib.np_test_reset.argtypes = [ctypes.c_uint32, vector]
    lib.np_test_policy.argtypes = [vector, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, vector, vector, vector]
    lib.np_test_step.argtypes = [vector, ctypes.c_double, vector, vector]
    if lib.np_test_load(model.encode()):
        raise RuntimeError("C controller bundle load failed")
    state = np.empty(22)
    lib.np_test_reset(seed, state)
    initial = state.copy()
    env = None
    if engine == "mujoco":
        model_mj = mj.MjModel.from_xml_string(independent_mjcf())
        data = mj.MjData(model_mj)
        data.qpos[0] = initial[0]
        data.qpos[1:] = np.diff(np.r_[0., initial[1:11]])
        data.qvel[0] = initial[11]
        data.qvel[1:] = np.diff(np.r_[0., initial[12:]])
        mj.mj_forward(model_mj, data)
        env = (model_mj, data)
    last, pending = np.zeros(1), np.zeros(1)
    command, estimate, diagnostics = np.empty(1), np.empty(22), np.empty(2)
    states, commands, applied, queued = [state.copy()], [], [], []
    inferred, residuals, conditions = [], [], []
    error = None
    start = time.perf_counter()
    for tick in range(1000):
        if lib.np_test_policy(state, last[0], pending[0], tick, command, estimate, diagnostics):
            error = "strict C encoder inversion failed"
            break
        inferred.append(estimate.copy())
        residuals.append(diagnostics[0])
        conditions.append(diagnostics[1])
        try:
            if env:
                model_mj, data = env
                previous = float(data.time)
                data.qfrc_applied[0] = 10.0 * pending[0]
                for _ in range(20):
                    mj.mj_step(model_mj, data)
                    for warning in (mj.mjtWarning.mjWARN_BADQPOS,
                                    mj.mjtWarning.mjWARN_BADQVEL,
                                    mj.mjtWarning.mjWARN_BADQACC):
                        if data.warning[int(warning)].number:
                            raise RuntimeError("MuJoCo numeric warning or automatic reset")
                if abs(float(data.time) - previous - .02) > 1e-9:
                    raise RuntimeError("MuJoCo time failed to advance by one control tick")
                state = np.r_[data.qpos[0], np.cumsum(data.qpos[1:]),
                              data.qvel[0], np.cumsum(data.qvel[1:])]
                last[0], pending[0] = pending[0], command[0]
            elif lib.np_test_step(state, command[0], last, pending):
                error = "C physics failed"
                break
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            break
        states.append(state.copy())
        commands.append(command[0])
        applied.append(last[0])
        queued.append(pending[0])
        if abs(state[0]) >= 2.4:
            error = "cart rail departure"
            break
    elapsed = time.perf_counter() - start
    states = np.asarray(states)
    angles = np.arctan2(np.sin(states[:, 1:11]), np.cos(states[:, 1:11]))
    good = (np.abs(states[:, 0]) < 2.4) & (np.max(np.abs(angles), axis=1) < .22) \
        & (np.max(np.abs(states[:, 12:]), axis=1) < 1.) & np.isfinite(states).all(axis=1)
    hold = 0
    for value in good[1:][::-1]:
        if not value:
            break
        hold += 1
    complete = len(states) == 1001 and error is None
    endpoints = bool(complete and np.all(good[-251:]))
    success = bool(endpoints and np.max(np.abs(states[:, 0])) < 2.4)
    replay = Path(out) / f"{engine}-{seed}.npz"
    np.savez_compressed(replay, states=states, commands=commands, applied_actions=applied,
                        pending_actions=queued, inferred_states=inferred,
                        inverse_residuals=residuals, inverse_conditions=conditions)
    decode_error = float(np.max(np.abs(np.asarray(inferred)[:len(states)-1] - states[:-1]))) if len(states) > 1 else None
    row = {"engine": engine, "seed": seed, "success": success, "ticks": len(states) - 1,
           "final_hold_ticks": hold, "all_251_final_endpoints_good": endpoints,
           "max_abs_cart_m": float(np.max(np.abs(states[:, 0]))),
           "max_decode_error": decode_error, "max_inverse_residual": float(max(residuals, default=0.)),
           "wall_seconds": elapsed, "recording": replay.name, "replay_sha256": sha(replay)}
    if error:
        row["error"] = error
    if env:
        row["mujoco_version"] = mj.__version__
        row["mjcf_sha256"] = hashlib.sha256(independent_mjcf().encode()).hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puffer-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--library", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final and args.episodes != 32:
        parser.error("Final validation requires exactly 32 starts per engine")
    if args.out.exists() or not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error("Output must be new and seeds must fit uint32")
    args.out.mkdir(parents=True)
    if args.library is None:
        import subprocess
        args.library = args.out / "npole.so"
        subprocess.run(["cc", "-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off",
            "-fPIC", "-shared", str(args.puffer_root / "tests/npole_bridge.c"),
            "-o", str(args.library), "-lm"], check=True)
    files = list((args.puffer_root / "ocean/npole").glob("*"))
    files += [args.puffer_root / "tests/npole_bridge.c", args.puffer_root / "config/npole.ini", args.puffer_root / "tests/test_npole.py", Path(__file__)]
    sources = {str(p.relative_to(args.puffer_root)): sha(p) for p in files if p.is_file()}
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(),
              "partition": "final" if args.final else "development", "n": 10,
              "seeds": list(range(args.seed, args.seed + args.episodes)),
              "engines": ["c_rk4", "mujoco"], "control_hz": 50, "physics_hz": 1000,
              "delay_ms": 20, "force_limit_N": 10, "horizon_ticks": 1000,
              "controller_bundle_sha256": sha(args.model), "library_sha256": sha(args.library),
              "source_sha256": sources, "training_during_evaluation": False}
    freeze_path = args.out / "frozen.json"
    freeze_path.write_text(json.dumps(frozen, indent=2) + "\n")
    tasks = [(engine, seed, str(args.library.resolve()), str(args.model.resolve()),
              str(args.out.resolve())) for engine in frozen["engines"] for seed in frozen["seeds"]]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(trial, tasks):
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("engine", "seed", "success", "ticks", "final_hold_ticks")}), flush=True)
    if sha(args.model) != frozen["controller_bundle_sha256"] or sha(args.library) != frozen["library_sha256"]:
        raise RuntimeError("Controller or library changed during evaluation")
    for name, digest in sources.items():
        if sha(args.puffer_root / name) != digest:
            raise RuntimeError(f"Source changed during evaluation: {name}")
    receipt = {"schema": "rage_mill.puffer_c_validation.v1", "configuration": frozen,
               "frozen_configuration_sha256": sha(freeze_path), "trials": rows,
               "successes": {engine: sum(r["success"] for r in rows if r["engine"] == engine)
                             for engine in frozen["engines"]}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.out / "receipt.json"), "successes": receipt["successes"]}), flush=True)


if __name__ == "__main__":
    main()
