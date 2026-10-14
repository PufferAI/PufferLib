"""Independent ten-pendulum mechanics and task tests. No GPU is required.

Run with pytest and NumPy; MuJoCo checks run when its package is available.
The core bridge needs only a C compiler. The API check also uses Raylib headers
downloaded by build.sh, but does not open a window or link graphics libraries.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    binary = tmp_path_factory.mktemp("npole") / "npole.so"
    flags = ["-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-fno-fast-math",
             "-ffp-contract=off", "-fPIC", "-shared", "-I" + str(ROOT / "src")]
    headers = sorted(ROOT.glob("raylib-*/include/raylib.h"))
    if headers:
        flags += ["-DNP_TEST_ENV_API", "-I" + str(headers[0].parent)]
    subprocess.run([compiler, *flags, str(ROOT / "tests/npole_bridge.c"),
                    "-o", str(binary), "-lm"], check=True, capture_output=True)
    lib = ctypes.CDLL(str(binary))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.np_test_acceleration.argtypes = [vector, ctypes.c_double, vector]
    lib.np_test_integrate.argtypes = [vector, ctypes.c_double, ctypes.c_int, ctypes.c_double]
    lib.np_test_reset.argtypes = [ctypes.c_uint32, vector]
    lib.np_test_goal.argtypes = [vector]
    return lib


def independent_point_mass_terms(state, action):
    """Sum endpoint Cartesian Jacobians, not the analytic coefficient table."""
    angles = state[1:11]
    omega = state[12:]
    mass = np.zeros((11, 11))
    mass[0, 0] = 1.0
    centrifugal = np.zeros(11)
    applied = np.zeros(11)
    applied[0] = 10 * np.clip(action, -1, 1)
    for endpoint in range(10):
        jacobian = np.zeros((2, 11))
        jacobian[0, 0] = 1
        jacobian[0, 1:endpoint + 2] = 0.5 * np.cos(angles[:endpoint + 1])
        jacobian[1, 1:endpoint + 2] = -0.5 * np.sin(angles[:endpoint + 1])
        centripetal = -0.5 * np.array([
            np.sum(np.sin(angles[:endpoint + 1]) * omega[:endpoint + 1] ** 2),
            np.sum(np.cos(angles[:endpoint + 1]) * omega[:endpoint + 1] ** 2)])
        mass += 0.1 * jacobian.T @ jacobian
        centrifugal += 0.1 * jacobian.T @ centripetal
        applied += jacobian.T @ np.array([0., -0.981])
    mass += 1e-6 * np.eye(11)
    rhs = applied - centrifugal - 0.05 * state[11:]
    return mass, rhs


@pytest.mark.parametrize("center", [0., np.pi, 1.2])
def test_acceleration_against_cartesian_jacobians(bridge, center):
    rng = np.random.default_rng(910001)
    for _ in range(20):
        state = np.r_[rng.uniform(-2, 2), rng.uniform(-1, 1, 10) + center,
                      rng.uniform(-3, 3, 11)]
        force = rng.uniform(-1, 1)
        actual = np.empty(11)
        assert bridge.np_test_acceleration(state, force, actual) == 0
        mass, rhs = independent_point_mass_terms(state, force)
        expected = np.linalg.solve(mass, rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-10)


def test_external_force_clamps_without_clipping_the_state(bridge):
    state = np.r_[0., np.linspace(-2, 2, 10), np.linspace(-3, 3, 11)]
    a, b = np.empty(11), np.empty(11)
    assert bridge.np_test_acceleration(state, 50., a) == 0
    assert bridge.np_test_acceleration(state, 1., b) == 0
    np.testing.assert_array_equal(a, b)


def point_mass_energy(state):
    angles = state[1:11]
    omega = state[12:]
    kinetic = .5 * state[11] ** 2 + .5e-6 * np.sum(state[11:] ** 2)
    potential = 0.
    for endpoint in range(10):
        vx = state[11] + .5 * np.sum(np.cos(angles[:endpoint + 1]) * omega[:endpoint + 1])
        vz = -.5 * np.sum(np.sin(angles[:endpoint + 1]) * omega[:endpoint + 1])
        kinetic += .05 * (vx * vx + vz * vz)
        potential += .1 * 9.81 * .5 * np.sum(np.cos(angles[:endpoint + 1]))
    return kinetic + potential


def test_energy_rate_equals_cart_work_minus_declared_damping(bridge):
    rng = np.random.default_rng(910003)
    for _ in range(12):
        state = np.r_[.02, rng.uniform(-2, 2, 10), rng.uniform(-1, 1, 11)]
        action = rng.uniform(-1, 1)
        acceleration = np.empty(11)
        assert bridge.np_test_acceleration(state, action, acceleration) == 0
        tangent = np.r_[state[11:], acceleration]
        epsilon = 1e-7
        rate = (point_mass_energy(state + epsilon * tangent)
                - point_mass_energy(state - epsilon * tangent)) / (2 * epsilon)
        work = 10 * action * state[11] - .05 * np.sum(state[11:] ** 2)
        assert rate == pytest.approx(work, rel=2e-5, abs=5e-7)


@pytest.mark.parametrize("index,value", [(10,.23),(21,1.01),(0,2.4),(11,np.nan)])
def test_goal_checks_the_last_link_speed_rail_and_finite_state(bridge, index, value):
    state = np.zeros(22)
    assert bridge.np_test_goal(state) == 1
    state[index] = value
    assert bridge.np_test_goal(state) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_inputs_fail(bridge, bad):
    state = np.zeros(22)
    out = np.empty(11)
    assert bridge.np_test_acceleration(state, bad, out) != 0
    state[20] = bad
    assert bridge.np_test_acceleration(state, 0., out) != 0


def test_rk4_converges_on_a_moving_coupled_chain(bridge):
    start = np.r_[0.03, np.linspace(-0.4, 0.8, 10), np.linspace(-0.5, 0.5, 11)]
    states = []
    for steps in (1, 2, 4, 128):
        state = start.copy()
        assert bridge.np_test_integrate(state, 0.3, steps, 0.02) == 0
        states.append(state)
    errors = [np.linalg.norm(s - states[-1]) for s in states[:-1]]
    assert errors[1] < errors[0] / 8
    assert errors[2] < errors[1] / 8


def independent_mjcf():
    bodies, closes = [], []
    for i in range(10):
        bodies.append(f'<body pos="0 0 {0 if i == 0 else 0.5}">'
            f'<joint name="h{i}" type="hinge" axis="0 1 0"/>'
            '<inertial pos="0 0 0.5" mass="0.1" diaginertia="0.000001 0.000001 0.000001"/>')
        closes.append('</body>')
    tendons = []
    for i in range(10):
        joints = ''.join(f'<joint joint="h{j}" coef="1"/>' for j in range(i + 1))
        tendons.append(f'<fixed damping="0.05">{joints}</fixed>')
    return '<mujoco><option gravity="0 0 -9.81" timestep="0.001" integrator="RK4"/>' \
        '<worldbody><body><joint type="slide" axis="1 0 0" damping="0.05"/>' \
        '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>' \
        + ''.join(bodies + closes) + '</body></worldbody><tendon>' \
        + ''.join(tendons) + '</tendon></mujoco>'


def test_acceleration_against_independent_mujoco(bridge):
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string(independent_mjcf())
    data = mj.MjData(model)
    rng = np.random.default_rng(910002)
    for _ in range(30):
        state = np.r_[0.03, rng.uniform(-0.4, 0.4, 10), rng.uniform(-0.5, 0.5, 11)]
        action = rng.uniform(-1, 1)
        data.qpos[0] = state[0]
        data.qpos[1:] = np.diff(np.r_[0., state[1:11]])
        data.qvel[0] = state[11]
        data.qvel[1:] = np.diff(np.r_[0., state[12:]])
        data.qfrc_applied[:] = 0.
        data.qfrc_applied[0] = 10 * action
        mj.mj_forward(model, data)
        expected = np.r_[data.qacc[0], np.cumsum(data.qacc[1:])]
        actual = np.empty(11)
        assert bridge.np_test_acceleration(state, action, actual) == 0
        # The independent MJCF's tiny tip rotational inertias differ from the
        # analytic 1e-6 diagonal by its cart term; do not claim bitwise equality.
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=8e-6)


def test_hanging_reset_distribution_and_determinism(bridge):
    a, b = np.empty(22), np.empty(22)
    for seed in range(32):
        bridge.np_test_reset(seed, a)
        bridge.np_test_reset(seed, b)
        np.testing.assert_array_equal(a, b)
        assert abs(a[0]) <= .08
        assert np.max(np.abs(a[1:11] - np.pi)) <= .04
        assert np.max(np.abs(a[11:])) <= .02


def test_puffer_api_queue_and_invalid_episode_logging(bridge):
    result = bridge.np_test_api()
    if result == -1:
        pytest.skip("Run build.sh first to download Raylib headers for API check")
    assert result == 0
