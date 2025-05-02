import functools
import numpy as np

import pufferlib

train_levels = [
    "s/h0_weak_thrust",
    "s/h7_unicycle_left",
    "s/h3_point_the_thruster",
    "s/h4_thrust_aim",
    "s/h1_thrust_over_ball",
    "s/h5_rotate_fall",
    "s/h9_explode_then_thrust_over",
    "s/h6_unicycle_right",
    "s/h8_unicycle_balance",
    "s/h2_one_wheel_car",
    "m/h0_unicycle",
    "m/h1_car_left",
    "m/h2_car_right",
    "m/h3_car_thrust",
    "m/h4_thrust_the_needle",
    "m/h5_angry_birds",
    "m/h6_thrust_over",
    "m/h7_car_flip",
    "m/h8_weird_vehicle",
    "m/h9_spin_the_right_way",
    "m/h10_thrust_right_easy",
    "m/h11_thrust_left_easy",
    "m/h12_thrustfall_left",
    "m/h13_thrustfall_right",
    "m/h14_thrustblock",
    "m/h15_thrustshoot",
    "m/h16_thrustcontrol_right",
    "m/h17_thrustcontrol_left",
    "m/h18_thrust_right_very_easy",
    "m/h19_thrust_left_very_easy",
    "m/arm_left",
    "m/arm_right",
    "m/arm_up",
    "m/arm_hard",
    "l/h0_angrybirds",
    "l/h1_car_left",
    "l/h2_car_ramp",
    "l/h3_car_right",
    "l/h4_cartpole",
    "l/h5_flappy_bird",
    "l/h6_lorry",
    "l/h7_maze_1",
    "l/h8_maze_2",
    "l/h9_morph_direction",
    "l/h10_morph_direction_2",
    "l/h11_obstacle_avoidance",
    "l/h12_platformer_1",
    "l/h13_platformer_2",
    "l/h14_simple_thruster",
    "l/h15_swing_up",
    "l/h16_thruster_goal",
    "l/h17_unicycle",
    "l/hard_beam_balance",
    "l/hard_cartpole_thrust",
    "l/hard_cartpole_wheels",
    "l/hard_lunar_lander",
    "l/hard_pinball",
    "l/grasp_hard",
    "l/grasp_easy",
    "l/mjc_half_cheetah",
    "l/mjc_half_cheetah_easy",
    "l/mjc_hopper",
    "l/mjc_hopper_easy",
    "l/mjc_swimmer",
    "l/mjc_walker",
    "l/mjc_walker_easy",
    "l/car_launch",
    "l/car_swing_around",
    "l/chain_lander",
    "l/chain_thrust",
    "l/gears",
    "l/lever_puzzle",
    "l/pr",
    "l/rail",
]


def env_creator(name="kinetix"):
    from kinetix.environment.env import ObservationType, ActionType

    _, obs, act = name.split("-")
    if obs == "symbolic": obs = "symbolic_flat"

    try:
        obs = ObservationType.from_string(obs)
    except ValueError:
        raise ValueError(f"Unknown observation type: {obs}.")
    try:
        act = ActionType.from_string(act)
    except ValueError:
        raise ValueError(f"Unknown action type: {act}.")

    return functools.partial(KinetixPufferEnv, observation_type=obs, action_type=act)


def make(name, *args, **kwargs):
    return KinetixPufferEnv(*args, **kwargs)


class KinetixPufferEnv(pufferlib.environment.PufferEnv):
    def __init__(self, observation_type, action_type, num_envs=1, buf=None):

        from kinetix.environment.env import make_kinetix_env, ObservationType, ActionType
        from kinetix.environment.env_state import EnvParams, StaticEnvParams
        from kinetix.environment.ued.ued_state import UEDParams
        from kinetix.environment.ued.ued import make_reset_fn_list_of_levels

        import jax
        from gymnax.environments.spaces import gymnax_space_to_gym_space

        self.observation_type = observation_type
        self.action_type = action_type

        # Use default parameters
        env_params = EnvParams()
        static_env_params = StaticEnvParams().replace()

        # Create the environment
        env = make_kinetix_env(
            observation_type=observation_type,  # ObservationType.PIXELS,
            action_type=action_type,  # ActionType.DISCRETE,
            reset_fn=make_reset_fn_list_of_levels(train_levels, static_env_params),
            env_params=env_params,
            static_env_params=static_env_params,
        )

        self.single_observation_space = gymnax_space_to_gym_space(env.observation_space(env_params))
        self.single_action_space = gymnax_space_to_gym_space(env.action_space(env_params))
        self.num_agents = num_envs

        self.env = env

        super().__init__(buf)
        self.env_params = env_params
        self.env = env

        self.reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        self.step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        self.rng = jax.random.PRNGKey(0)

    def reset(self, rng, params=None):
        import jax
        from torch.utils import dlpack as torch_dlpack

        self.rng, _rng = jax.random.split(self.rng)
        self.rngs = jax.random.split(_rng, self.num_agents)
        obs, self.state = self.reset_fn(self.rngs, params)
        obs = self._obs_to_tensor(obs)

        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        return self.observations, []

    def step(self, action):
        import jax
        from torch.utils import dlpack as torch_dlpack

        obs, self.state, reward, done, info = self.step_fn(self.rngs, self.state, action, self.env_params)
        obs = self._obs_to_tensor(obs)

        # Convert JAX array to DLPack, then to PyTorch tensor
        self.observations = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
        self.rewards = np.asarray(reward)
        self.terminals = np.asarray(done)
        infos = [{k: v.mean().item() for k, v in info.items()} | {"reward": self.rewards.mean()} ]
        return self.observations, self.rewards, self.terminals, self.terminals, infos

    def _obs_to_tensor(self, obs):
        from kinetix.environment.env import ObservationType
        if self.observation_type == ObservationType.PIXELS:
            return obs.image
        elif self.observation_type == ObservationType.SYMBOLIC_FLAT:
            return obs
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}.")
