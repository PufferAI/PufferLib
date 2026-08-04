import ctypes

import numpy as np

from pufferlib import _C


BASE_ENV = "kinematic_bicycle"
INTEGRAL_ENV = "kinematic_bicycle_integral"


def float_view(ptr, count):
    array_t = ctypes.c_float * count
    return np.ctypeslib.as_array(array_t.from_address(ptr))


def default_env_args():
    return {
        "dt": 0.05,
        "v": 2.0,
        "wheelbase": 2.5,
        "psi_max": 0.785,
        "y_max": 2.0,
        "max_steps": 500,
        "d_y": 0.0,
        "regime_switch_step": -1,
        "d_y_after_switch": 0.0,
        "k_psi": 1.0,
        "action_delay_steps": 0,
        "sensor_noise_y_std": 0.0,
        "sensor_noise_theta_std": 0.0,
        "y0_min": 0.0,
        "y0_max": 0.0,
        "theta0_min": 0.0,
        "theta0_max": 0.0,
        "w_y": 1.0,
        "w_theta": 0.5,
        "w_psi": 0.01,
        "alive_bonus": 0.1,
        "failure_penalty": 100.0,
        "w_center4": 0.0,
        "z_clip": 5.0,
        "reference": 0.0,
        "lambda": 0.0,
    }


def make_vec(env_args):
    args = {
        "vec": {
            "total_agents": 1,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": env_args,
    }

    vec = _C.create_vec(args, 0)
    vec.reset()

    obs = float_view(vec.obs_ptr, vec.obs_size)
    rewards = float_view(vec.rewards_ptr, 1)
    terminals = float_view(vec.terminals_ptr, 1)
    actions = np.zeros((1, 1), dtype=np.float32)

    return vec, obs, rewards, terminals, actions


def test_zero_action_straight_line():
    env_args = default_env_args()
    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        np.testing.assert_allclose(obs[:2], [0.0, 0.0], atol=1e-7)

        vec.cpu_step(actions.ctypes.data)

        np.testing.assert_allclose(obs[:2], [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(rewards[0], 0.1, atol=1e-6)
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_positive_steering_changes_heading():
    env_args = default_env_args()
    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        actions[0, 0] = 1.0
        vec.cpu_step(actions.ctypes.data)

        expected_theta = (
            env_args["v"]
            / env_args["wheelbase"]
            * np.tan(env_args["psi_max"])
            * env_args["dt"]
        )

        np.testing.assert_allclose(obs[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(obs[1], expected_theta, rtol=1e-5)
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_lateral_disturbance():
    env_args = default_env_args()
    env_args["d_y"] = 0.4

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)
        expected_y = env_args["d_y"] * env_args["dt"]
        np.testing.assert_allclose(obs[0], expected_y, atol=1e-7)
    finally:
        vec.close()


def test_failure_penalty_and_auto_reset():
    env_args = default_env_args()
    env_args["y0_min"] = 1.99
    env_args["y0_max"] = 1.99
    env_args["d_y"] = 1.0

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)

        expected_failed_y = 1.99 + env_args["d_y"] * env_args["dt"]
        expected_reward = (
            env_args["alive_bonus"]
            - env_args["w_y"] * expected_failed_y**2
            - env_args["failure_penalty"]
        )

        np.testing.assert_allclose(rewards[0], expected_reward, rtol=1e-6)
        assert terminals[0] == 1.0

        # The native vector automatically resets after termination.
        np.testing.assert_allclose(obs[0], 1.99, atol=1e-7)
        np.testing.assert_allclose(obs[1], 0.0, atol=1e-7)
    finally:
        vec.close()



def test_negative_steering_changes_heading():
    env_args = default_env_args()
    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        actions[0, 0] = -1.0
        vec.cpu_step(actions.ctypes.data)

        expected_theta = (
            env_args["v"]
            / env_args["wheelbase"]
            * np.tan(-env_args["psi_max"])
            * env_args["dt"]
        )

        np.testing.assert_allclose(obs[1], expected_theta, rtol=1e-5)
        assert obs[1] < 0.0
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_action_clamping():
    env_args = default_env_args()

    vec_one, obs_one, _, _, actions_one = make_vec(env_args)
    vec_large, obs_large, _, _, actions_large = make_vec(env_args)

    try:
        actions_one[0, 0] = 1.0
        actions_large[0, 0] = 4.0

        vec_one.cpu_step(actions_one.ctypes.data)
        vec_large.cpu_step(actions_large.ctypes.data)

        np.testing.assert_allclose(
            obs_large[:2],
            obs_one[:2],
            rtol=1e-6,
            atol=1e-7,
        )
    finally:
        vec_one.close()
        vec_large.close()


def test_steering_effectiveness():
    env_args = default_env_args()
    env_args["k_psi"] = 0.5

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        actions[0, 0] = 1.0
        vec.cpu_step(actions.ctypes.data)

        applied_psi = env_args["k_psi"] * env_args["psi_max"]
        expected_theta = (
            env_args["v"]
            / env_args["wheelbase"]
            * np.tan(applied_psi)
            * env_args["dt"]
        )

        np.testing.assert_allclose(obs[1], expected_theta, rtol=1e-5)
    finally:
        vec.close()


def test_action_delay():
    env_args = default_env_args()
    env_args["action_delay_steps"] = 2

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        actions[0, 0] = 1.0
        vec.cpu_step(actions.ctypes.data)
        np.testing.assert_allclose(obs[1], 0.0, atol=1e-7)

        actions[0, 0] = 0.0
        vec.cpu_step(actions.ctypes.data)
        np.testing.assert_allclose(obs[1], 0.0, atol=1e-7)

        vec.cpu_step(actions.ctypes.data)

        expected_theta = (
            env_args["v"]
            / env_args["wheelbase"]
            * np.tan(env_args["psi_max"])
            * env_args["dt"]
        )

        np.testing.assert_allclose(obs[1], expected_theta, rtol=1e-5)
    finally:
        vec.close()


def test_disturbance_regime_switch():
    env_args = default_env_args()
    env_args["d_y"] = 0.2
    env_args["regime_switch_step"] = 1
    env_args["d_y_after_switch"] = -0.4

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)
        expected_first_y = 0.2 * env_args["dt"]
        np.testing.assert_allclose(obs[0], expected_first_y, atol=1e-7)

        vec.cpu_step(actions.ctypes.data)
        expected_second_y = expected_first_y - 0.4 * env_args["dt"]
        np.testing.assert_allclose(obs[0], expected_second_y, atol=1e-7)
    finally:
        vec.close()



def test_max_steps_truncation():
    env_args = default_env_args()
    env_args["max_steps"] = 2

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)
        assert terminals[0] == 0.0

        vec.cpu_step(actions.ctypes.data)
        assert terminals[0] == 1.0

        np.testing.assert_allclose(obs[:2], [0.0, 0.0], atol=1e-7)
    finally:
        vec.close()


def test_centerline_fourth_power_penalty():
    env_args = default_env_args()
    env_args["y0_min"] = 0.5
    env_args["y0_max"] = 0.5
    env_args["w_center4"] = 2.0

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)

        expected_reward = (
            env_args["alive_bonus"]
            - env_args["w_y"] * 0.5**2
            - env_args["w_center4"] * 0.5**4
        )

        np.testing.assert_allclose(rewards[0], expected_reward, rtol=1e-6)
    finally:
        vec.close()




def test_reset_ranges():
    env_args = default_env_args()
    env_args["y0_min"] = -0.35
    env_args["y0_max"] = 0.45
    env_args["theta0_min"] = -0.15
    env_args["theta0_max"] = 0.25

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        observed_y = []
        observed_theta = []

        for _ in range(100):
            vec.reset()

            y_value = float(obs[0])
            theta_value = float(obs[1])

            assert env_args["y0_min"] <= y_value <= env_args["y0_max"]
            assert (
                env_args["theta0_min"]
                <= theta_value
                <= env_args["theta0_max"]
            )

            observed_y.append(y_value)
            observed_theta.append(theta_value)

        assert np.ptp(observed_y) > 0.1
        assert np.ptp(observed_theta) > 0.05
    finally:
        vec.close()


def test_zero_sensor_noise_is_deterministic():
    env_args = default_env_args()
    env_args["y0_min"] = 0.25
    env_args["y0_max"] = 0.25
    env_args["theta0_min"] = -0.1
    env_args["theta0_max"] = -0.1

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        for _ in range(20):
            vec.reset()
            np.testing.assert_allclose(
                obs[:2],
                [0.25, -0.1],
                atol=1e-7,
            )
    finally:
        vec.close()


def test_sensor_noise_is_finite_and_variable():
    env_args = default_env_args()
    env_args["sensor_noise_y_std"] = 0.1
    env_args["sensor_noise_theta_std"] = 0.05

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        samples = []

        for _ in range(200):
            vec.reset()
            samples.append(obs[:2].copy())

        samples = np.asarray(samples)

        assert np.isfinite(samples).all()
        assert np.std(samples[:, 0]) > 0.03
        assert np.std(samples[:, 1]) > 0.015

        np.testing.assert_allclose(
            np.mean(samples[:, 0]),
            0.0,
            atol=0.04,
        )
        np.testing.assert_allclose(
            np.mean(samples[:, 1]),
            0.0,
            atol=0.025,
        )
    finally:
        vec.close()



def test_multi_environment_vector():
    num_envs = 32
    env_args = default_env_args()
    env_args["y0_min"] = -0.4
    env_args["y0_max"] = 0.4
    # Keep heading deterministic so steering direction can be checked.
    env_args["theta0_min"] = 0.0
    env_args["theta0_max"] = 0.0

    args = {
        "vec": {
            "total_agents": num_envs,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": env_args,
    }

    vec = _C.create_vec(args, 0)
    vec.reset()

    obs = float_view(
        vec.obs_ptr,
        num_envs * vec.obs_size,
    ).reshape(num_envs, vec.obs_size)
    rewards = float_view(vec.rewards_ptr, num_envs)
    terminals = float_view(vec.terminals_ptr, num_envs)
    actions = np.linspace(
        -1.0,
        1.0,
        num_envs,
        dtype=np.float32,
    ).reshape(num_envs, 1)

    try:
        assert np.isfinite(obs).all()
        assert np.ptp(obs[:, 0]) > 0.1

        vec.cpu_step(actions.ctypes.data)

        assert np.isfinite(obs).all()
        assert np.isfinite(rewards).all()
        assert np.isfinite(terminals).all()

        assert obs[0, 1] < 0.0
        assert obs[-1, 1] > 0.0
    finally:
        vec.close()


def test_repeated_create_reset_close():
    for _ in range(100):
        env_args = default_env_args()
        env_args["action_delay_steps"] = 3

        vec, obs, rewards, terminals, actions = make_vec(env_args)

        try:
            for _ in range(5):
                vec.reset()
                actions[0, 0] = 0.5
                vec.cpu_step(actions.ctypes.data)

                assert np.isfinite(obs).all()
                assert np.isfinite(rewards).all()
                assert np.isfinite(terminals).all()
        finally:
            vec.close()


def test_long_vector_stress():
    num_envs = 128
    num_steps = 5000

    env_args = default_env_args()
    env_args["d_y"] = 0.1
    env_args["regime_switch_step"] = 100
    env_args["d_y_after_switch"] = -0.1
    env_args["action_delay_steps"] = 2
    env_args["sensor_noise_y_std"] = 0.01
    env_args["sensor_noise_theta_std"] = 0.005

    args = {
        "vec": {
            "total_agents": num_envs,
            "num_buffers": 1,
            "num_threads": 1,
        },
        "env": env_args,
    }

    vec = _C.create_vec(args, 0)
    vec.reset()

    obs = float_view(
        vec.obs_ptr,
        num_envs * vec.obs_size,
    ).reshape(num_envs, vec.obs_size)
    rewards = float_view(vec.rewards_ptr, num_envs)
    terminals = float_view(vec.terminals_ptr, num_envs)

    rng = np.random.default_rng(1234)
    actions = np.zeros((num_envs, 1), dtype=np.float32)

    try:
        for _ in range(num_steps):
            actions[:, 0] = rng.uniform(
                -1.5,
                1.5,
                size=num_envs,
            )
            vec.cpu_step(actions.ctypes.data)

        assert np.isfinite(obs).all()
        assert np.isfinite(rewards).all()
        assert np.isfinite(terminals).all()
    finally:
        vec.close()


def test_integral_lambda_penalty():
    if _C.env_name != INTEGRAL_ENV:
        return

    env_args = default_env_args()
    env_args["y0_min"] = 0.5
    env_args["y0_max"] = 0.5
    env_args["lambda"] = 2.0

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)

        expected_z = -env_args["dt"] * 0.5
        expected_reward = (
            env_args["alive_bonus"]
            - env_args["w_y"] * 0.5**2
            - env_args["lambda"] * expected_z**2
        )

        np.testing.assert_allclose(obs[2], expected_z, atol=1e-7)
        np.testing.assert_allclose(
            rewards[0],
            expected_reward,
            rtol=1e-6,
        )
    finally:
        vec.close()


def test_integral_clip():
    if _C.env_name != INTEGRAL_ENV:
        return

    env_args = default_env_args()
    env_args["y0_min"] = 1.0
    env_args["y0_max"] = 1.0
    env_args["z_clip"] = 0.02

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)
        np.testing.assert_allclose(obs[2], -0.02, atol=1e-7)

        vec.cpu_step(actions.ctypes.data)
        np.testing.assert_allclose(obs[2], -0.02, atol=1e-7)
    finally:
        vec.close()


def test_integral_nonzero_reference():
    if _C.env_name != INTEGRAL_ENV:
        return

    env_args = default_env_args()
    env_args["y0_min"] = 0.25
    env_args["y0_max"] = 0.25
    env_args["reference"] = 0.75

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        vec.cpu_step(actions.ctypes.data)

        expected_z = (
            env_args["dt"]
            * (env_args["reference"] - 0.25)
        )

        np.testing.assert_allclose(obs[2], expected_z, atol=1e-7)
    finally:
        vec.close()


def test_integral_observation():
    if _C.env_name != INTEGRAL_ENV:
        return

    env_args = default_env_args()
    env_args["y0_min"] = 0.5
    env_args["y0_max"] = 0.5

    vec, obs, rewards, terminals, actions = make_vec(env_args)

    try:
        assert vec.obs_size == 3
        np.testing.assert_allclose(obs, [0.5, 0.0, 0.0], atol=1e-7)

        vec.cpu_step(actions.ctypes.data)

        expected_z = -env_args["dt"] * 0.5
        np.testing.assert_allclose(obs[2], expected_z, atol=1e-7)
    finally:
        vec.close()


def main():
    assert _C.env_name in {BASE_ENV, INTEGRAL_ENV}
    expected_obs_size = 2 if _C.env_name == BASE_ENV else 3

    assert _C.gpu == 0

    probe_args = default_env_args()
    vec, _, _, _, _ = make_vec(probe_args)
    try:
        assert vec.obs_size == expected_obs_size
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [1]
    finally:
        vec.close()

    test_zero_action_straight_line()
    test_positive_steering_changes_heading()
    test_negative_steering_changes_heading()
    test_action_clamping()
    test_steering_effectiveness()
    test_action_delay()
    test_lateral_disturbance()
    test_disturbance_regime_switch()
    test_failure_penalty_and_auto_reset()
    test_max_steps_truncation()
    test_centerline_fourth_power_penalty()
    test_reset_ranges()
    test_zero_sensor_noise_is_deterministic()
    test_sensor_noise_is_finite_and_variable()
    test_multi_environment_vector()
    test_repeated_create_reset_close()
    test_long_vector_stress()
    test_integral_lambda_penalty()
    test_integral_clip()
    test_integral_nonzero_reference()
    test_integral_observation()

    print(f"PASS: {_C.env_name} deterministic parity checks")


if __name__ == "__main__":
    main()
