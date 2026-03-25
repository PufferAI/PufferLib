import numpy as np
import pytest

from pufferlib.ocean.clifford.reference import (
    ReferenceCliffordEnv,
)

try:
    import pufferlib.ocean.clifford.binding  # noqa: F401
except ImportError:
    BINDING_AVAILABLE = False
else:
    BINDING_AVAILABLE = True
    from pufferlib.ocean.clifford.clifford import Clifford


def _apply_explicit(matrix, gate):
    return ((matrix.astype(np.uint8) @ gate.astype(np.uint8)) % 2).astype(np.uint8)


def _identity_symplectic(n_qubits):
    return np.eye(2 * n_qubits, dtype=np.uint8)


def _build_actions(n_qubits):
    actions = []
    for gate in ("h", "s", "v", "hs", "hv"):
        for qubit in range(n_qubits):
            actions.append((gate, qubit, -1))
    for src in range(n_qubits):
        for dst in range(src + 1, n_qubits):
            actions.append(("cz", src, dst))
    return actions


def _symplectic_matrix_h(n_qubits, qubit):
    gate = _identity_symplectic(n_qubits)
    z_col = n_qubits + qubit
    gate[:, [qubit, z_col]] = gate[:, [z_col, qubit]]
    return gate


def _symplectic_matrix_s(n_qubits, qubit):
    gate = _identity_symplectic(n_qubits)
    gate[qubit, n_qubits + qubit] = 1
    return gate


def _symplectic_matrix_v(n_qubits, qubit):
    gate = _symplectic_matrix_s(n_qubits, qubit)
    gate = (gate @ _symplectic_matrix_h(n_qubits, qubit)) % 2
    gate = (gate @ _symplectic_matrix_s(n_qubits, qubit)) % 2
    return gate.astype(np.uint8, copy=False)


def _symplectic_matrix_hs(n_qubits, qubit):
    gate = _symplectic_matrix_h(n_qubits, qubit)
    gate = (gate @ _symplectic_matrix_s(n_qubits, qubit)) % 2
    return gate.astype(np.uint8, copy=False)


def _symplectic_matrix_hv(n_qubits, qubit):
    gate = _symplectic_matrix_h(n_qubits, qubit)
    gate = (gate @ _symplectic_matrix_v(n_qubits, qubit)) % 2
    return gate.astype(np.uint8, copy=False)


def _symplectic_matrix_cz(n_qubits, src, dst):
    gate = _identity_symplectic(n_qubits)
    gate[src, n_qubits + dst] = 1
    gate[dst, n_qubits + src] = 1
    return gate


def _find_action(actions, gate_name, q0, q1=-1):
    for idx, action in enumerate(actions):
        if action == (gate_name, q0, q1):
            return idx
    raise AssertionError(f"Action {(gate_name, q0, q1)} not found")


def _normalized_identity_hamming_distance(matrix):
    identity = _identity_symplectic(matrix.shape[0] // 2)
    return float(np.count_nonzero(matrix != identity) / matrix.size)


def test_build_actions_full_connectivity_order():
    actions = _build_actions(3)
    assert len(actions) == 18
    assert actions[:5] == [
        ("h", 0, -1),
        ("h", 1, -1),
        ("h", 2, -1),
        ("s", 0, -1),
        ("s", 1, -1),
    ]
    assert actions[-3:] == [
        ("cz", 0, 1),
        ("cz", 0, 2),
        ("cz", 1, 2),
    ]


def test_reference_reset_with_difficulty_zero_is_identity():
    env = ReferenceCliffordEnv(n_qubits=3, difficulty=0, max_steps=8)
    obs, info = env.reset(seed=7)
    np.testing.assert_array_equal(obs, _identity_symplectic(3).reshape(-1))
    assert info == {}


def test_reference_reset_with_difficulty_nonzero_is_not_identity():
    env = ReferenceCliffordEnv(n_qubits=3, difficulty=4, max_steps=8)
    obs, _info = env.reset(seed=7)
    assert not np.array_equal(obs, _identity_symplectic(3).reshape(-1))


def test_reference_set_matrix_rejects_nonsymplectic_input():
    env = ReferenceCliffordEnv(n_qubits=2, difficulty=0, max_steps=8)
    bad = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="symplectic"):
        env.set_matrix(bad)


@pytest.mark.parametrize("reward_mode", ["bad_mode", "dense_hamming", "end_hamming_left"])
def test_reference_rejects_invalid_reward_mode(reward_mode):
    with pytest.raises(ValueError, match="reward_mode"):
        ReferenceCliffordEnv(n_qubits=2, reward_mode=reward_mode)


def test_reference_rejects_negative_hamming_left_scale():
    with pytest.raises(ValueError, match="hamming_left_scale"):
        ReferenceCliffordEnv(n_qubits=2, hamming_left_scale=-0.1)


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_env_shapes_and_dtypes():
    env = Clifford(n_qubits=3, difficulty=0, max_steps=8, num_envs=4)
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (4, 36)
        assert obs.dtype == np.uint8
        assert info == []
        assert env.actions.dtype == np.int32
        assert env.rewards.dtype == np.float32
        assert env.terminals.dtype == np.bool_
        assert env.truncations.dtype == np.bool_
    finally:
        env.close()


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
@pytest.mark.parametrize(
    "action",
    [
        ("h", 0, -1),
        ("s", 1, -1),
        ("v", 2, -1),
        ("hs", 1, -1),
        ("hv", 0, -1),
        ("cz", 0, 2),
    ],
)
def test_native_matches_reference_for_gate_updates(action):
    env_ref = ReferenceCliffordEnv(n_qubits=3, difficulty=0, max_steps=10)
    env_native = Clifford(n_qubits=3, difficulty=0, max_steps=10, num_envs=1)
    try:
        base = _identity_symplectic(3)
        base = _apply_explicit(base, _symplectic_matrix_h(3, 1))
        base = _apply_explicit(base, _symplectic_matrix_s(3, 2))
        base = _apply_explicit(base, _symplectic_matrix_cz(3, 0, 2))
        action_idx = _find_action(env_ref._actions, *action)

        env_ref.reset(options={"matrix": base})
        env_native.reset(seed=0)
        env_native.set_matrix(base)

        obs_ref, reward_ref, term_ref, trunc_ref, info_ref = env_ref.step(action_idx)
        obs_native, rewards, terminals, truncations, info_native = env_native.step(
            np.asarray([action_idx], dtype=np.int32)
        )

        np.testing.assert_array_equal(obs_native[0], obs_ref)
        assert rewards[0] == pytest.approx(reward_ref, abs=1e-6)
        assert bool(terminals[0]) == term_ref
        assert bool(truncations[0]) == trunc_ref
        assert info_ref == {}
        assert info_native == []
    finally:
        env_native.close()


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
@pytest.mark.parametrize(
    "action",
    [
        ("h", 0, -1),
        ("s", 1, -1),
        ("v", 2, -1),
        ("hs", 1, -1),
        ("hv", 0, -1),
        ("cz", 0, 2),
    ],
)
def test_native_matches_reference_for_hamming_left_rewards(action):
    env_ref = ReferenceCliffordEnv(
        n_qubits=3,
        difficulty=0,
        max_steps=10,
        reward_mode="hamming_left",
        hamming_left_scale=0.5,
    )
    env_native = Clifford(
        n_qubits=3,
        difficulty=0,
        max_steps=10,
        num_envs=1,
        reward_mode="hamming_left",
        hamming_left_scale=0.5,
    )
    try:
        base = _identity_symplectic(3)
        base = _apply_explicit(base, _symplectic_matrix_h(3, 1))
        base = _apply_explicit(base, _symplectic_matrix_s(3, 2))
        base = _apply_explicit(base, _symplectic_matrix_cz(3, 0, 2))
        action_idx = _find_action(env_ref._actions, *action)

        env_ref.reset(options={"matrix": base})
        env_native.reset(seed=0)
        env_native.set_matrix(base)

        obs_ref, reward_ref, term_ref, trunc_ref, _info_ref = env_ref.step(action_idx)
        obs_native, rewards, terminals, truncations, info_native = env_native.step(
            np.asarray([action_idx], dtype=np.int32)
        )

        np.testing.assert_array_equal(obs_native[0], obs_ref)
        assert rewards[0] == pytest.approx(reward_ref, abs=1e-6)
        assert bool(terminals[0]) == term_ref
        assert bool(truncations[0]) == trunc_ref
        assert info_native == []
    finally:
        env_native.close()


def test_reference_hamming_left_penalizes_remaining_distance():
    env = ReferenceCliffordEnv(
        n_qubits=2,
        difficulty=0,
        max_steps=8,
        single_qubit_cost=0.01,
        reward_mode="hamming_left",
        hamming_left_scale=0.5,
    )
    env.reset(options={"matrix": _symplectic_matrix_h(2, 1)})
    action_idx = _find_action(_build_actions(2), "s", 0)
    obs, reward, terminated, truncated, _info = env.step(action_idx)
    matrix = obs.reshape(4, 4)
    expected = -0.01 - 0.5 * _normalized_identity_hamming_distance(matrix)
    assert reward == pytest.approx(expected, abs=1e-6)
    assert not terminated
    assert not truncated


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_terminal_step_auto_resets_to_identity():
    env = Clifford(
        n_qubits=2,
        difficulty=0,
        max_steps=8,
        single_qubit_cost=0.01,
        goal_bonus=5.0,
        num_envs=1,
    )
    try:
        env.reset(seed=0)
        env.set_matrix(_symplectic_matrix_h(2, 0))
        action_idx = _find_action(_build_actions(2), "h", 0)
        obs, rewards, terminals, truncations, info = env.step(np.asarray([action_idx], dtype=np.int32))
        assert rewards[0] == pytest.approx(4.99, abs=1e-6)
        assert bool(terminals[0])
        assert not bool(truncations[0])
        np.testing.assert_array_equal(obs[0], _identity_symplectic(2).reshape(-1))
        assert info == []
    finally:
        env.close()


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_hamming_left_terminal_reward_matches_reference():
    env = Clifford(
        n_qubits=2,
        difficulty=0,
        max_steps=8,
        single_qubit_cost=0.01,
        goal_bonus=5.0,
        reward_mode="hamming_left",
        hamming_left_scale=0.5,
        num_envs=1,
    )
    try:
        env.reset(seed=0)
        env.set_matrix(_symplectic_matrix_h(2, 0))
        action_idx = _find_action(_build_actions(2), "h", 0)
        obs, rewards, terminals, truncations, info = env.step(np.asarray([action_idx], dtype=np.int32))
        assert rewards[0] == pytest.approx(4.99, abs=1e-6)
        assert bool(terminals[0])
        assert not bool(truncations[0])
        np.testing.assert_array_equal(obs[0], _identity_symplectic(2).reshape(-1))
        assert info == []
    finally:
        env.close()


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_set_matrix_rejects_nonsymplectic_input():
    env = Clifford(n_qubits=2, difficulty=0, max_steps=8, num_envs=1)
    try:
        bad = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="symplectic"):
            env.set_matrix(bad)
    finally:
        env.close()


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_rejects_invalid_reward_mode():
    with pytest.raises(ValueError, match="reward_mode"):
        Clifford(n_qubits=2, reward_mode="dense_hamming")


@pytest.mark.skipif(not BINDING_AVAILABLE, reason="native clifford binding is not built")
def test_native_rejects_negative_hamming_left_scale():
    with pytest.raises(ValueError, match="hamming_left_scale"):
        Clifford(n_qubits=2, hamming_left_scale=-0.1)
