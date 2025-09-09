import functools

import gymnasium
import numpy as np

import pufferlib
from metta.mettagrid.builder.envs import make_arena
from metta.mettagrid.mettagrid_env import MettaGridEnv


def env_creator(name="metta"):
    return functools.partial(make, name)


def make(
    name,
    config="pufferlib/environments/metta/metta.yaml",
    render_mode="auto",
    buf=None,
    seed=0,
    ore_reward=0.1,
    battery_reward=0.8,
    heart_reward=1.0,
    num_agents=60,
):
    """Metta creation function"""

    # Create a basic arena configuration using the make_arena function
    mettagrid_cfg = make_arena(num_agents=num_agents)

    # Apply arena_basic_easy_shaped configuration
    # Set inventory rewards (matching arena_basic_easy_shaped)
    mettagrid_cfg.game.agent.rewards.inventory = {
        "heart": float(heart_reward),
        "ore_red": float(ore_reward),
        "battery_red": float(battery_reward),
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }

    # Set inventory max limits (matching arena_basic_easy_shaped)
    mettagrid_cfg.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    mettagrid_cfg.game.objects["altar"].input_resources = {"battery_red": 1}

    return MettaPuff(mettagrid_cfg, render_mode=render_mode, buf=buf, seed=seed)


def oc_divide(a, b):
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result


class MettaPuff(MettaGridEnv):
    def __init__(self, env_cfg, render_mode="human", buf=None, seed=0):
        self.replay_writer = None
        # if render_mode == 'auto':
        #    self.replay_writer = ReplayWriter("metta/")

        super().__init__(env_cfg=env_cfg, render_mode=render_mode, replay_writer=self.replay_writer)
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)

    @property
    def single_action_space(self):
        # Prefer exposing a flattened Discrete action space matching Metta's
        # internal "full" action logits when we can determine the action
        # parameterization. This keeps PufferLib's sampling path (which
        # expects a single discrete action) compatible with Metta's joint
        # action representation.
        try:
            # MettaGridEnv exposes `max_action_args` describing per-action
            # parameter counts; the flattened action count is sum(max_param+1).
            max_args = getattr(self, "max_action_args", None)
            if max_args is not None:
                total = int(sum([int(x) + 1 for x in max_args]))
                # Return a 1-D MultiDiscrete so atn_shape is non-empty
                # and the shared-memory actions buffer is 2-D (workers, agents, 1)
                return gymnasium.spaces.MultiDiscrete([total], dtype=np.int32)
        except Exception:
            pass

        # Fallback to previous behavior
        return gymnasium.spaces.MultiDiscrete(super().single_action_space.nvec, dtype=np.int32)

    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        if all(term) or all(trunc):
            self.reset()
            if "agent_raw" in info:
                del info["agent_raw"]
            if "episode_rewards" in info:
                info["score"] = info["episode_rewards"]

        else:
            info = []

        return obs, rew, term, trunc, [info]
