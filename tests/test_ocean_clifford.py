import ctypes

import numpy as np
import pytest

try:
    from pufferlib import _C
except ImportError:
    _C = None

BINDING_AVAILABLE = _C is not None and getattr(_C, "env_name", None) == "clifford"
DEFAULT_N_QUBITS = 6


def _identity_symplectic(n_qubits=DEFAULT_N_QUBITS):
    return np.eye(2 * n_qubits, dtype=np.uint8)


def _build_actions(n_qubits=DEFAULT_N_QUBITS, use_shortcut_gates=True):
    actions = []
    single_qubit_gates = ("h", "s")
    if use_shortcut_gates:
        single_qubit_gates = single_qubit_gates + ("v", "hs", "hv")
    for gate in single_qubit_gates:
        for qubit in range(n_qubits):
            actions.append((gate, qubit, -1))
    for src in range(n_qubits):
        for dst in range(src + 1, n_qubits):
            actions.append(("cz", src, dst))
    return actions


def _find_action(actions, gate_name, q0, q1=-1):
    for idx, action in enumerate(actions):
        if action == (gate_name, q0, q1):
            return idx
    raise AssertionError(f"Action {(gate_name, q0, q1)} not found")


def _expected_obs_after_identity_gate(n_qubits, gate_name, q0, q1=-1):
    matrix = _identity_symplectic(n_qubits)
    x_col = matrix[:, q0].copy()
    z_col = matrix[:, n_qubits + q0].copy()
    if gate_name == "h":
        matrix[:, q0] = z_col
        matrix[:, n_qubits + q0] = x_col
    elif gate_name == "s":
        matrix[:, n_qubits + q0] ^= x_col
    elif gate_name == "v":
        matrix[:, q0] = x_col ^ z_col
    elif gate_name == "hs":
        matrix[:, q0] = z_col
        matrix[:, n_qubits + q0] = x_col ^ z_col
    elif gate_name == "hv":
        matrix[:, q0] = x_col ^ z_col
        matrix[:, n_qubits + q0] = x_col
    elif gate_name == "cz":
        q1_x_col = matrix[:, q1].copy()
        matrix[:, n_qubits + q0] ^= q1_x_col
        matrix[:, n_qubits + q1] ^= x_col
    else:
        raise AssertionError(f"Unknown gate {gate_name}")
    return matrix.reshape(-1)


def _obs_array(vec):
    raw = (ctypes.c_uint8 * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
    return np.ctypeslib.as_array(raw).reshape(vec.total_agents, vec.obs_size)


def _float_array(ptr, length):
    raw = (ctypes.c_float * length).from_address(ptr)
    return np.ctypeslib.as_array(raw)


def _make_args(
    num_envs=1,
    n_qubits=None,
    difficulty=0,
    max_steps=8,
    goal_bonus=0.0,
    failure_penalty=-1.0,
    single_qubit_cost=0.001,
    cz_cost=0.1,
    use_shortcut_gates=1,
    seed=0,
):
    env_args = {
        "difficulty": difficulty,
        "max_steps": max_steps,
        "single_qubit_cost": single_qubit_cost,
        "cz_cost": cz_cost,
        "goal_bonus": goal_bonus,
        "failure_penalty": failure_penalty,
        "use_shortcut_gates": use_shortcut_gates,
        "seed": seed,
    }
    if n_qubits is not None:
        env_args["n_qubits"] = n_qubits

    return {
        "vec": {
            "total_agents": num_envs,
            "num_buffers": 1,
        },
        "env": env_args,
    }


def _make_vec(**kwargs):
    vec = _C.create_vec(_make_args(**kwargs), 0)
    vec.reset()
    return vec


def _step(vec, action_idx, num_envs=1):
    actions = np.full((num_envs, 1), float(action_idx), dtype=np.float32)
    vec.cpu_step(actions.ctypes.data)
    obs = _obs_array(vec).copy()
    rewards = _float_array(vec.rewards_ptr, num_envs).copy()
    terminals = _float_array(vec.terminals_ptr, num_envs).copy()
    return obs, rewards, terminals


def _n_qubits_from_vec(vec):
    dim = int(round(vec.obs_size**0.5))
    assert dim * dim == vec.obs_size
    assert dim % 2 == 0
    return dim // 2


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
    assert actions[6:9] == [
        ("v", 0, -1),
        ("v", 1, -1),
        ("v", 2, -1),
    ]
    assert len(_build_actions(3, use_shortcut_gates=False)) == 9


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_vec_shapes_and_dtypes():
    vec = _make_vec(num_envs=4, difficulty=0)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        obs_size = (2 * n_qubits) ** 2
        num_actions = 5 * n_qubits + n_qubits * (n_qubits - 1) // 2
        assert vec.total_agents == 4
        assert vec.obs_size == obs_size
        assert vec.num_atns == 1
        assert vec.act_sizes == [num_actions]
        assert vec.obs_dtype == "ByteTensor"
        obs = _obs_array(vec)
        assert obs.shape == (4, obs_size)
        assert obs.dtype == np.uint8
        np.testing.assert_array_equal(
            obs[0], _identity_symplectic(n_qubits).reshape(-1)
        )
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
@pytest.mark.parametrize(
    "gate_name",
    [
        "h",
        "s",
        "v",
        "hs",
        "hv",
        "cz",
    ],
)
def test_native_applies_known_gate_from_identity(gate_name):
    vec = _make_vec(difficulty=0, max_steps=10)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        actions = _build_actions(n_qubits)
        if gate_name == "cz":
            if n_qubits < 2:
                pytest.skip("CZ requires at least two compiled qubits")
            action = (gate_name, 0, 1)
        else:
            action = (gate_name, 0, -1)
        action_idx = _find_action(actions, *action)
        expected = _expected_obs_after_identity_gate(n_qubits, *action)
        obs, rewards, terminals = _step(vec, action_idx)
        np.testing.assert_array_equal(obs[0], expected)
        assert rewards[0] == pytest.approx(
            -0.1 if gate_name == "cz" else -0.001, abs=1e-6
        )
        assert not bool(terminals[0])
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_terminal_step_auto_resets_to_identity():
    vec = _make_vec(difficulty=0, max_steps=8, goal_bonus=5.0)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        action_idx = _find_action(_build_actions(n_qubits), "h", 0)
        _step(vec, action_idx)
        obs, rewards, terminals = _step(vec, action_idx)
        assert rewards[0] == pytest.approx(4.999, abs=1e-6)
        assert bool(terminals[0])
        np.testing.assert_array_equal(
            obs[0], _identity_symplectic(n_qubits).reshape(-1)
        )
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_log_reports_completed_episodes():
    vec = _make_vec(difficulty=0, max_steps=8, goal_bonus=5.0)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        action_idx = _find_action(_build_actions(n_qubits), "h", 0)
        _step(vec, action_idx)
        _step(vec, action_idx)
        log = vec.log()
        assert log["n"] == pytest.approx(1.0)
        assert log["score"] == pytest.approx(4.998, abs=1e-6)
        assert log["success_rate"] == pytest.approx(1.0)
        assert log["difficulty"] == pytest.approx(0.0)
        assert log["max_steps"] == pytest.approx(8.0)
        assert log["episode_length"] == pytest.approx(2.0)
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_cz_cost_is_configurable():
    vec = _make_vec(difficulty=0, max_steps=8, cz_cost=0.05)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        if n_qubits < 2:
            pytest.skip("CZ requires at least two compiled qubits")
        action_idx = _find_action(_build_actions(n_qubits), "cz", 0, 1)
        _obs, rewards, terminals = _step(vec, action_idx)
        assert rewards[0] == pytest.approx(-0.05, abs=1e-6)
        assert not bool(terminals[0])
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_failure_penalty_is_configurable():
    vec = _make_vec(difficulty=0, max_steps=1, failure_penalty=-0.25)
    try:
        n_qubits = _n_qubits_from_vec(vec)
        action_idx = _find_action(_build_actions(n_qubits), "h", 0)
        _obs, rewards, terminals = _step(vec, action_idx)
        assert rewards[0] == pytest.approx(-0.251, abs=1e-6)
        assert bool(terminals[0])
    finally:
        vec.close()


@pytest.mark.skipif(
    not BINDING_AVAILABLE, reason="native clifford binding is not built"
)
def test_native_fractional_difficulty_is_reported():
    vec = _make_vec(num_envs=8, difficulty=1.25, max_steps=1)
    try:
        action_idx = _find_action(_build_actions(_n_qubits_from_vec(vec)), "h", 0)
        for _ in range(4):
            _step(vec, action_idx, num_envs=8)
        log = vec.log()
        assert log["n"] > 0
        assert log["difficulty"] == pytest.approx(1.25)
        assert log["max_steps"] == pytest.approx(1.0)
    finally:
        vec.close()
