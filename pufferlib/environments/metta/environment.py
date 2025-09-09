import functools
from typing import Optional
import numpy as np

from metta.mettagrid.builder.envs import make_arena
from metta.mettagrid.puffer_base import MettaGridPufferBase


def env_creator(name="metta"):
    return functools.partial(make, name)


def make(
    name,
    config: Optional[str] = None,
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


class MettaPuff(MettaGridPufferBase):
    def __init__(self, env_cfg, render_mode="human", buf=None, seed=0):
        # Initialize the parent PufferBase class
        super().__init__(mg_config=env_cfg, render_mode=render_mode, buf=buf)

        # Set seed if provided
        if seed != 0:
            self._current_seed = seed

        # Ensure actions are int32 for PufferLib compatibility
        self.actions = self.actions.astype(np.int32)

    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        # Handle episode completion
        if all(term) or all(trunc):
            # Note: MettaGridPufferBase handles auto-reset internally
            # Clean up info dictionary if it exists
            if isinstance(info, dict):
                if "agent_raw" in info:
                    del info["agent_raw"]
                if "episode_rewards" in info:
                    info["score"] = info["episode_rewards"]
                return obs, rew, term, trunc, [info]
            else:
                return obs, rew, term, trunc, [{}]
        else:
            # Return empty info list for non-terminal steps
            return obs, rew, term, trunc, []
