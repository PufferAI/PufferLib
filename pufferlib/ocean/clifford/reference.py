from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

Action = tuple[str, int, int]
SINGLE_QUBIT_GATES = ("h", "s", "v", "hs", "hv")
REWARD_MODES = {"gate_cost", "hamming_left"}


def _build_actions(n_qubits: int) -> list[Action]:
    actions: list[Action] = []
    for gate in SINGLE_QUBIT_GATES:
        for qubit in range(n_qubits):
            actions.append((gate, qubit, -1))

    for src in range(n_qubits):
        for dst in range(src + 1, n_qubits):
            actions.append(("cz", src, dst))

    return actions


def _identity_symplectic(n_qubits: int) -> np.ndarray:
    return np.eye(2 * n_qubits, dtype=np.uint8)


def _symplectic_form(n_qubits: int) -> np.ndarray:
    omega = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=np.uint8)
    eye = np.eye(n_qubits, dtype=np.uint8)
    omega[:n_qubits, n_qubits:] = eye
    omega[n_qubits:, :n_qubits] = eye
    return omega


def _is_symplectic(matrix: np.ndarray) -> bool:
    matrix_u8 = np.asarray(matrix, dtype=np.uint8)
    if matrix_u8.ndim != 2 or matrix_u8.shape[0] != matrix_u8.shape[1]:
        return False
    if matrix_u8.shape[0] % 2 != 0:
        return False
    n_qubits = matrix_u8.shape[0] // 2
    omega = _symplectic_form(n_qubits)
    lhs = (matrix_u8.T @ omega @ matrix_u8) % 2
    return bool(np.array_equal(lhs.astype(np.uint8), omega))


def _identity_hamming_distance(matrix: np.ndarray) -> int:
    identity = np.eye(matrix.shape[0], dtype=np.uint8)
    return int(np.count_nonzero(matrix != identity))


def _normalized_identity_hamming_distance(matrix: np.ndarray) -> float:
    return float(_identity_hamming_distance(matrix) / matrix.size)


def _xor_columns_inplace(matrix: np.ndarray, dst_idx: int, src_col: np.ndarray) -> None:
    np.bitwise_xor(matrix[:, dst_idx], src_col, out=matrix[:, dst_idx])


def apply_h_inplace(matrix: np.ndarray, qubit: int) -> None:
    n_qubits = matrix.shape[0] // 2
    z_col = n_qubits + qubit
    matrix[:, [qubit, z_col]] = matrix[:, [z_col, qubit]]


def apply_s_inplace(matrix: np.ndarray, qubit: int) -> None:
    n_qubits = matrix.shape[0] // 2
    _xor_columns_inplace(matrix, n_qubits + qubit, matrix[:, qubit].copy())


def apply_v_inplace(matrix: np.ndarray, qubit: int) -> None:
    apply_s_inplace(matrix, qubit)
    apply_h_inplace(matrix, qubit)
    apply_s_inplace(matrix, qubit)


def apply_hs_inplace(matrix: np.ndarray, qubit: int) -> None:
    apply_h_inplace(matrix, qubit)
    apply_s_inplace(matrix, qubit)


def apply_hv_inplace(matrix: np.ndarray, qubit: int) -> None:
    apply_h_inplace(matrix, qubit)
    apply_v_inplace(matrix, qubit)


def apply_cz_inplace(matrix: np.ndarray, src: int, dst: int) -> None:
    if src == dst:
        raise ValueError("CZ requires distinct qubits")
    n_qubits = matrix.shape[0] // 2
    src_x = matrix[:, src].copy()
    dst_x = matrix[:, dst].copy()
    _xor_columns_inplace(matrix, n_qubits + src, dst_x)
    _xor_columns_inplace(matrix, n_qubits + dst, src_x)


class ReferenceCliffordEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_qubits: int = 6,
        difficulty: int = 10,
        max_steps: int = 200,
        single_qubit_cost: float = 0.01,
        goal_bonus: float = 0.0,
        reward_mode: str = "gate_cost",
        hamming_left_scale: float = 0.5,
        render_mode=None,
    ):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if difficulty < 0:
            raise ValueError("difficulty must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if goal_bonus < 0.0:
            raise ValueError("goal_bonus must be non-negative")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {sorted(REWARD_MODES)}")
        if hamming_left_scale < 0.0:
            raise ValueError("hamming_left_scale must be non-negative")

        self.n_qubits = int(n_qubits)
        self.difficulty = int(difficulty)
        self.max_steps = int(max_steps)
        self.single_qubit_cost = float(single_qubit_cost)
        self.goal_bonus = float(goal_bonus)
        self.reward_mode = str(reward_mode)
        self.hamming_left_scale = float(hamming_left_scale)
        self.render_mode = render_mode
        self._actions = _build_actions(self.n_qubits)
        self._identity = _identity_symplectic(self.n_qubits)
        self._matrix = self._identity.copy()
        self.steps = 0

        obs_size = self._identity.size
        self.action_space = gymnasium.spaces.Discrete(len(self._actions))
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.uint8,
        )

    def _get_obs(self) -> np.ndarray:
        return self._matrix.reshape(-1).copy()

    def set_difficulty(self, difficulty: int) -> None:
        difficulty = int(difficulty)
        if difficulty < 0:
            raise ValueError("difficulty must be non-negative")
        self.difficulty = difficulty

    def set_max_steps(self, max_steps: int) -> None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps

    def set_matrix(self, matrix: np.ndarray) -> None:
        matrix_u8 = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint8))
        expected_shape = (2 * self.n_qubits, 2 * self.n_qubits)
        if matrix_u8.shape != expected_shape:
            raise ValueError(f"Expected matrix shape {expected_shape}, got {matrix_u8.shape}")
        if not _is_symplectic(matrix_u8):
            raise ValueError("matrix must be symplectic")
        self._matrix = matrix_u8.copy()
        self.steps = 0

    def _sample_action(self) -> int:
        return int(self.np_random.integers(len(self._actions)))

    def _apply_action(self, action_idx: int) -> str:
        gate, q0, q1 = self._actions[action_idx]
        if gate == "h":
            apply_h_inplace(self._matrix, q0)
        elif gate == "s":
            apply_s_inplace(self._matrix, q0)
        elif gate == "v":
            apply_v_inplace(self._matrix, q0)
        elif gate == "hs":
            apply_hs_inplace(self._matrix, q0)
        elif gate == "hv":
            apply_hv_inplace(self._matrix, q0)
        elif gate == "cz":
            apply_cz_inplace(self._matrix, q0, q1)
        else:
            raise ValueError(f"Unknown gate {gate}")
        return gate

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options and "matrix" in options:
            self.set_matrix(options["matrix"])
            return self._get_obs(), {}

        if self.difficulty <= 0:
            self._matrix = self._identity.copy()
            self.steps = 0
            return self._get_obs(), {}

        while True:
            self._matrix = self._identity.copy()
            for _ in range(self.difficulty):
                self._apply_action(self._sample_action())
            if not np.array_equal(self._matrix, self._identity):
                break

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_idx = int(action) % len(self._actions)
        gate = self._apply_action(action_idx)
        self.steps += 1

        terminated = bool(np.array_equal(self._matrix, self._identity))
        truncated = bool(not terminated and self.steps >= self.max_steps)
        reward = -1.0 if gate == "cz" else -self.single_qubit_cost
        if self.reward_mode == "hamming_left":
            reward -= self.hamming_left_scale * _normalized_identity_hamming_distance(self._matrix)
        if terminated:
            reward += self.goal_bonus

        return self._get_obs(), float(reward), terminated, truncated, {}


__all__ = [
    "ReferenceCliffordEnv",
]
