#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pufferlib import _C
from pufferlib.torch_pufferl import load_policy

Action = tuple[str, int, int]
BASE_SINGLE_QUBIT_GATES = ("h", "s")
SHORTCUT_SINGLE_QUBIT_GATES = ("v", "hs", "hv")


def build_actions(n_qubits: int, use_shortcut_gates: bool = True) -> list[Action]:
    actions: list[Action] = []
    single_qubit_gates = BASE_SINGLE_QUBIT_GATES
    if use_shortcut_gates:
        single_qubit_gates = single_qubit_gates + SHORTCUT_SINGLE_QUBIT_GATES
    for gate in single_qubit_gates:
        for qubit in range(n_qubits):
            actions.append((gate, qubit, -1))

    for src in range(n_qubits):
        for dst in range(src + 1, n_qubits):
            actions.append(("cz", src, dst))

    return actions


def identity_symplectic(n_qubits):
    return np.eye(2 * n_qubits, dtype=np.uint8)


def symplectic_form(n_qubits):
    omega = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=np.uint8)
    eye = np.eye(n_qubits, dtype=np.uint8)
    omega[:n_qubits, n_qubits:] = eye
    omega[n_qubits:, :n_qubits] = eye
    return omega


def is_symplectic(matrix):
    matrix_u8 = np.asarray(matrix, dtype=np.uint8)
    if matrix_u8.ndim != 2 or matrix_u8.shape[0] != matrix_u8.shape[1]:
        return False
    if matrix_u8.shape[0] % 2 != 0:
        return False
    n_qubits = matrix_u8.shape[0] // 2
    omega = symplectic_form(n_qubits)
    lhs = (matrix_u8.T @ omega @ matrix_u8) % 2
    return bool(np.array_equal(lhs.astype(np.uint8), omega))


def xor_columns_inplace(matrix, dst_idx, src_col):
    np.bitwise_xor(matrix[:, dst_idx], src_col, out=matrix[:, dst_idx])


def apply_action_inplace(matrix, action):
    gate, q0, q1 = action
    n_qubits = matrix.shape[0] // 2
    if gate == "h":
        z_col = n_qubits + q0
        matrix[:, [q0, z_col]] = matrix[:, [z_col, q0]]
    elif gate == "s":
        xor_columns_inplace(matrix, n_qubits + q0, matrix[:, q0].copy())
    elif gate == "v":
        apply_action_inplace(matrix, ("s", q0, -1))
        apply_action_inplace(matrix, ("h", q0, -1))
        apply_action_inplace(matrix, ("s", q0, -1))
    elif gate == "hs":
        apply_action_inplace(matrix, ("h", q0, -1))
        apply_action_inplace(matrix, ("s", q0, -1))
    elif gate == "hv":
        apply_action_inplace(matrix, ("h", q0, -1))
        apply_action_inplace(matrix, ("v", q0, -1))
    elif gate == "cz":
        src_x = matrix[:, q0].copy()
        dst_x = matrix[:, q1].copy()
        xor_columns_inplace(matrix, n_qubits + q0, dst_x)
        xor_columns_inplace(matrix, n_qubits + q1, src_x)
    else:
        raise ValueError(f"unknown gate {gate}")


def latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "clifford", "**", "*.bin")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found at {pattern}")
    return max(candidates, key=os.path.getctime)


def n_qubits_from_matrix(matrix):
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] % 2 != 0
    ):
        raise ValueError(f"expected an even square tableau, got {matrix.shape}")
    return matrix.shape[0] // 2


def load_matrix(path, n_qubits=None):
    if path.endswith(".npy"):
        matrix = np.load(path)
    else:
        with open(path) as f:
            matrix = np.asarray(json.load(f), dtype=np.uint8)
    matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
    matrix_n_qubits = n_qubits_from_matrix(matrix)
    if n_qubits is not None and matrix_n_qubits != n_qubits:
        expected_shape = (2 * n_qubits, 2 * n_qubits)
        raise ValueError(f"expected a {expected_shape} tableau, got {matrix.shape}")
    if not is_symplectic(matrix):
        raise ValueError("matrix is not symplectic")
    return matrix


def random_tableau(n_qubits, seed, random_steps, use_shortcut_gates=True):
    if random_steps <= 0:
        return identity_symplectic(n_qubits)

    actions = build_actions(n_qubits, use_shortcut_gates=use_shortcut_gates)
    rng = np.random.default_rng(seed)
    while True:
        matrix = identity_symplectic(n_qubits)
        for _ in range(random_steps):
            apply_action_inplace(matrix, actions[int(rng.integers(len(actions)))])
        if not np.array_equal(matrix, identity_symplectic(n_qubits)):
            return matrix


def default_checkpoint_dir(n_qubits, hidden_size, use_shortcut_gates=True):
    action_suffix = "" if use_shortcut_gates else "_hs_cz"
    return f"checkpoints/clifford_{n_qubits}q{action_suffix}_mlp{hidden_size}_long"


def make_policy_args(args, checkpoint, n_qubits):
    return {
        "env_name": "clifford",
        "checkpoint_dir": args.checkpoint_dir,
        "load_model_path": checkpoint,
        "load_id": None,
        "wandb": False,
        "vec": {
            "total_agents": 1,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": {
            "n_qubits": n_qubits,
            "difficulty": 0,
            "max_steps": args.max_steps + 1,
            "single_qubit_cost": 0.001,
            "cz_cost": 0.1,
            "goal_bonus": 0.0,
            "failure_penalty": -1.0,
            "use_shortcut_gates": int(args.use_shortcut_gates),
            "seed": args.seed,
        },
        "policy": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "expansion_factor": 1,
        },
        "torch": {
            "network": args.network,
            "encoder": "DefaultEncoder",
            "decoder": "DefaultDecoder",
        },
    }


def synthesize(policy, matrix, max_steps, use_shortcut_gates=True):
    matrix = matrix.copy()
    n_qubits = matrix.shape[0] // 2
    actions = build_actions(n_qubits, use_shortcut_gates=use_shortcut_gates)
    identity = identity_symplectic(n_qubits)
    device = next(policy.parameters()).device
    state = policy.initial_state(1, device)
    sequence = []

    policy.eval()
    with torch.no_grad():
        for step in range(max_steps + 1):
            if np.array_equal(matrix, identity):
                return sequence, True
            obs_t = torch.as_tensor(matrix.reshape(1, -1), device=device)
            logits, _value, state = policy.forward_eval(obs_t, state)
            action_idx = int(torch.argmax(logits, dim=-1).item())
            sequence.append(actions[action_idx])
            apply_action_inplace(matrix, actions[action_idx])
            if np.array_equal(matrix, identity):
                return sequence, True

    return sequence, False


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize Clifford tableaus with a trained Puffer policy"
    )
    parser.add_argument(
        "--checkpoint", default="latest", help="Checkpoint path, or 'latest'"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Checkpoint directory. Defaults to checkpoints/clifford_<n>q_mlp<hidden>_long",
    )
    parser.add_argument(
        "--matrix", help="Path to a .npy or JSON tableau. Omit for random."
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        help="Number of qubits. Inferred from --matrix, otherwise defaults to 3.",
    )
    parser.add_argument("--random-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--network", default="MLP")
    parser.add_argument("--use-shortcut-gates", action="store_true", default=True)
    parser.add_argument(
        "--no-shortcut-gates", action="store_false", dest="use_shortcut_gates"
    )
    args = parser.parse_args()

    if args.n_qubits is not None and args.n_qubits <= 0:
        raise ValueError("--n-qubits must be positive")

    if args.matrix:
        matrix = load_matrix(args.matrix, n_qubits=args.n_qubits)
        n_qubits = n_qubits_from_matrix(matrix)
    else:
        n_qubits = args.n_qubits or 3
        matrix = random_tableau(
            n_qubits,
            args.seed,
            args.random_steps,
            use_shortcut_gates=args.use_shortcut_gates,
        )

    if args.checkpoint_dir is None:
        args.checkpoint_dir = default_checkpoint_dir(
            n_qubits,
            args.hidden_size,
            use_shortcut_gates=args.use_shortcut_gates,
        )

    if getattr(_C, "env_name", None) != "clifford":
        raise RuntimeError(
            "Build Clifford first, e.g. "
            f"EXTRA_CFLAGS='-DCLIFFORD_N_QUBITS={n_qubits} "
            f"-DCLIFFORD_USE_SHORTCUT_GATES={int(args.use_shortcut_gates)}' "
            "bash build.sh clifford --cpu"
        )

    checkpoint = (
        latest_checkpoint(args.checkpoint_dir)
        if args.checkpoint == "latest"
        else args.checkpoint
    )
    policy_args = make_policy_args(args, checkpoint, n_qubits)
    vec = _C.create_vec(policy_args, 0)
    try:
        expected_obs_size = (2 * n_qubits) ** 2
        expected_actions = len(
            build_actions(n_qubits, use_shortcut_gates=args.use_shortcut_gates)
        )
        if vec.obs_size != expected_obs_size or vec.act_sizes != [expected_actions]:
            raise RuntimeError(
                f"This synthesizer needs a {n_qubits}-qubit Clifford build "
                f"(obs_size={expected_obs_size}, act_sizes={[expected_actions]}); "
                f"got obs_size={vec.obs_size}, act_sizes={vec.act_sizes}"
            )
        policy = load_policy(policy_args, vec)
    finally:
        vec.close()

    sequence, solved = synthesize(
        policy,
        matrix,
        args.max_steps,
        use_shortcut_gates=args.use_shortcut_gates,
    )
    for idx, (gate, q0, q1) in enumerate(sequence, 1):
        if q1 < 0:
            print(f"{idx:02d}: {gate} {q0}")
        else:
            print(f"{idx:02d}: {gate} {q0} {q1}")

    print(f"solved={solved} steps={len(sequence)} checkpoint={checkpoint}")
    if not solved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
