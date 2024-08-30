
from pdb import set_trace as T

import functools

import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def single_env_creator(env_name, capture_video, gamma, run_name=None, idx=None, obs_norm=True, pufferl=False):
    if capture_video and idx == 0:
        assert run_name is not None, "run_name must be specified when capturing videos"
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gymnasium.make(env_name)

    env = pufferlib.postprocess.ClipAction(env)  # NOTE: this changed actions space
    env = pufferlib.postprocess.EpisodeStats(env)

    if obs_norm:
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    if pufferl is True:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env)

    return env


def cleanrl_env_creator(env_name, run_name, capture_video, gamma, idx):
    kwargs = {
        "env_name": env_name,
        "run_name": run_name,
        "capture_video": capture_video,
        "gamma": gamma,
        "idx": idx,
        "pufferl": False,
    }
    return functools.partial(single_env_creator, **kwargs)


# Keep it simple for pufferl demo, for now
def env_creator(env_name="HalfCheetah-v4", gamma=0.99):
    default_kwargs = {
        "env_name": env_name,
        "capture_video": False,
        "gamma": gamma,
        "pufferl": True,
    }
    return functools.partial(single_env_creator, **default_kwargs)
