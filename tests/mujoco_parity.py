"""Parity test: ocean/mujoco envs vs MuJoCo through gymnasium.

Writes a reference trajectory (random actions) with gymnasium + mujoco, compiles
tests/test_mujoco_parity.c against ocean/mujoco/mjc_<env>.h and runs it with
resources/mujoco/<env>.bin. Requires pip install "gymnasium[mujoco]".

    python tests/mujoco_parity.py mjc_half_cheetah [--steps 300] [--seed 123]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

GYM_IDS = {"half_cheetah": "HalfCheetah-v5", "hopper": "Hopper-v5", "walker2d": "Walker2d-v5",
    "ant": "Ant-v5", "humanoid": "Humanoid-v5", "swimmer": "Swimmer-v5"}
# reward weights passed to the harness (gymnasium defaults)
DEFINES = {"half_cheetah": ["-DCTRL_COST_WEIGHT=0.1"],
    "hopper": ["-DCTRL_COST_WEIGHT=1e-3", "-DHEALTHY_REWARD=1.0"],
    "walker2d": ["-DCTRL_COST_WEIGHT=1e-3", "-DHEALTHY_REWARD=1.0"],
    "ant": ["-DCTRL_COST_WEIGHT=0.5", "-DHEALTHY_REWARD=1.0", "-DCONTACT_COST_WEIGHT=5e-4"],
    "humanoid": ["-DCTRL_COST_WEIGHT=0.1", "-DHEALTHY_REWARD=5.0", "-DCONTACT_COST_WEIGHT=5e-7",
        "-DFORWARD_REWARD_WEIGHT=1.25"],
    "swimmer": ["-DCTRL_COST_WEIGHT=1e-4"]}


def write_reference(path, gym_id, steps, seed):
    import gymnasium as gym
    env = gym.make(gym_id)
    env.reset(seed=seed)
    u = env.unwrapped
    m, d = u.model, u.data
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        f.write("%d %d %d %d %d\n" % (steps, m.nq, m.nv, m.nu, u.observation_space.shape[0]))
        for _ in range(steps):
            # start state, action, resulting state, reward, ncon, terminated, obs
            f.write(" ".join("%.17g" % v for v in d.qpos) + "\n")
            f.write(" ".join("%.17g" % v for v in d.qvel) + "\n")
            a = rng.uniform(u.action_space.low, u.action_space.high)
            obs, r, term, trunc, _ = env.step(a)
            f.write(" ".join("%.17g" % v for v in a) + "\n")
            f.write(" ".join("%.17g" % v for v in d.qpos) + "\n")
            f.write(" ".join("%.17g" % v for v in d.qvel) + "\n")
            f.write("%.17g %d %d\n" % (r, d.ncon, term))
            f.write(" ".join("%.17g" % v for v in obs) + "\n")
            if term:
                env.reset(seed=int(rng.integers(1 << 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ref", help="write/keep the reference file here")
    ap.add_argument("--cc", default=os.environ.get("CC", "clang"))
    args = ap.parse_args()
    name = args.env[4:] if args.env.startswith("mjc_") else args.env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raylib = os.path.join(root, "raylib-5.5_linux_amd64")
    with tempfile.TemporaryDirectory() as tmp:
        ref = args.ref or os.path.join(tmp, "ref.txt")
        exe = os.path.join(tmp, "test_mujoco_parity")
        write_reference(ref, GYM_IDS[name], args.steps, args.seed)
        subprocess.check_call([
            args.cc, "-O2", "-Wno-narrowing", "-Wno-unused-function",
            "-DENV_HEADER=\"../ocean/mujoco/mjc_%s.h\"" % name] + DEFINES[name] + [
            "-I" + os.path.join(root, "src"), "-I" + os.path.join(root, "ocean", "mujoco"),
            "-I" + os.path.join(raylib, "include"),
            os.path.join(root, "tests", "test_mujoco_parity.c"),
            os.path.join(raylib, "lib", "libraylib.a"),
            "-lGL", "-lm", "-lpthread", "-ldl", "-o", exe])
        model = os.path.join(root, "resources", "mujoco", name + ".bin")
        out = subprocess.check_output([exe, ref, model], text=True, cwd=root)
    print(out)
    first = out.splitlines()[0]
    max_dq = float(first.split("max|dq|")[1].split()[0])
    max_dv = float(first.split("max|dv|")[1].split()[0])
    ok = max_dq < 1e-4 and max_dv < 1e-2
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
