#!/usr/bin/env bash
set -euo pipefail

# CPU/PyTorch curriculum run for Clifford synthesis.
# Tune with env vars, e.g.:
#   CURRICULUM_PROFILE=auto scripts/train_clifford_curriculum.sh  # fast for <=3q, steady for >=4q
#   CURRICULUM_PROFILE=fast scripts/train_clifford_curriculum.sh  # old 3q defaults, no fixed stage budget
#   CURRICULUM_PROFILE=steady N_QUBITS=4 scripts/train_clifford_curriculum.sh  # 4q steady recipe
#   MAX_DIFFICULTY=12 scripts/train_clifford_curriculum.sh
#   TIMESTEPS_PER_STAGE=4000000 scripts/train_clifford_curriculum.sh
#   FIRST_STAGE_TIMESTEPS=2000000 TIMESTEPS_PER_STAGE=500000 scripts/train_clifford_curriculum.sh
#   MASTERY_THRESHOLD=0.95 scripts/train_clifford_curriculum.sh
#   ENT_COEF=0.05 GOAL_BONUS=1.0 scripts/train_clifford_curriculum.sh
#   OPTIMIZER=muon scripts/train_clifford_curriculum.sh
#   MAX_STAGE_ATTEMPTS=3 scripts/train_clifford_curriculum.sh
#   MIN_MAX_STEPS=4 MAX_STEPS_BASE_SLACK=2 scripts/train_clifford_curriculum.sh
#   MAX_STEPS_STDDEVS=5 MAX_STEPS_STDDEVS_AFTER_DIFFICULTY=32 MAX_STEPS_SLACK=64 MAX_STEPS_SLACK_AFTER_DIFFICULTY=32 scripts/train_clifford_curriculum.sh
#   N_QUBITS=4 scripts/train_clifford_curriculum.sh
#   FORCE_BUILD=1 scripts/train_clifford_curriculum.sh

if [[ -x /opt/homebrew/opt/llvm/bin/clang && -z "${CC:-}" ]]; then
  export CC=/opt/homebrew/opt/llvm/bin/clang
fi
if [[ -x /opt/homebrew/opt/llvm/bin/clang++ && -z "${CXX:-}" ]]; then
  export CXX=/opt/homebrew/opt/llvm/bin/clang++
fi

USE_SHORTCUT_GATES="${USE_SHORTCUT_GATES:-1}"
N_QUBITS="${N_QUBITS:-3}"
export EXTRA_CFLAGS="${EXTRA_CFLAGS:-"-DCLIFFORD_N_QUBITS=$N_QUBITS -DCLIFFORD_USE_SHORTCUT_GATES=$USE_SHORTCUT_GATES"}"
FORCE_BUILD="${FORCE_BUILD:-0}"

has_correct_build() {
  python - <<'PY'
import sys
import ctypes
import os

import numpy as np

try:
    from pufferlib import _C
except Exception:
    raise SystemExit(1)

if getattr(_C, "env_name", None) != "clifford" or getattr(_C, "gpu", None) != 0:
    raise SystemExit(1)

use_shortcut_gates = int(os.environ.get("USE_SHORTCUT_GATES", "1"))
n_qubits = int(os.environ.get("N_QUBITS", "3"))
expected_obs_size = (2 * n_qubits) ** 2
expected_actions = (5 if use_shortcut_gates else 2) * n_qubits + n_qubits * (n_qubits - 1) // 2
args = {
    "vec": {
        "total_agents": 1,
        "num_buffers": 1,
    },
    "env": {
        "n_qubits": n_qubits,
        "difficulty": 0,
        "max_steps": 1,
        "single_qubit_cost": 0.001,
        "cz_cost": 0.1,
        "goal_bonus": 1.0,
        "failure_penalty": -1.0,
        "use_shortcut_gates": use_shortcut_gates,
        "seed": 0,
    },
}

try:
    vec = _C.create_vec(args, 0)
except Exception:
    raise SystemExit(1)

try:
    if vec.obs_size != expected_obs_size or vec.act_sizes != [expected_actions] or vec.num_atns != 1:
        raise SystemExit(1)
    actions = np.zeros((1, 1), dtype=np.float32)
    vec.cpu_step(actions.ctypes.data)
    rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
    if rewards[0] > -1.0:
        raise SystemExit(1)
    vec.cpu_step(actions.ctypes.data)
    log = vec.log()
    if "difficulty" not in log or "max_steps" not in log:
        raise SystemExit(1)
finally:
    vec.close()
PY
}

if [[ "$FORCE_BUILD" != "1" ]] && has_correct_build >/dev/null 2>&1; then
  echo "Using existing ${N_QUBITS}-qubit Clifford CPU backend."
else
  echo "Building ${N_QUBITS}-qubit Clifford CPU backend..."
  bash build.sh clifford --cpu
fi

fix_macos_openmp() {
  [[ "$(uname -s)" == "Darwin" ]] || return 0

  local ext_suffix output torch_libomp linked_libomp
  ext_suffix="$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
  output="pufferlib/_C${ext_suffix}"
  [[ -f "$output" ]] || return 0

  torch_libomp="$(
    python - <<'PY'
import os
try:
    import torch
    path = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
    print(path if os.path.exists(path) else "")
except Exception:
    print("")
PY
  )"
  [[ -n "$torch_libomp" ]] || return 0

  linked_libomp="$(otool -L "$output" | awk '/libomp\.dylib/{print $1; exit}')"
  if [[ -n "$linked_libomp" && "$linked_libomp" != "$torch_libomp" ]]; then
    echo "Pointing $output at PyTorch libomp to avoid duplicate OpenMP runtimes..."
    install_name_tool -change "$linked_libomp" "$torch_libomp" "$output"
  fi
}

fix_macos_openmp

exec python scripts/train_clifford_curriculum.py "$@"
