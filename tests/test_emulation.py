from pdb import set_trace as T
import numpy as np

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel

import nmmo

import pufferlib


def setup():
    kaz = aec_to_parallel(knights_archers_zombies_v10.env(max_zombies=0))
    mmo = nmmo.Env()

    return [mmo]


def test_pack_obs_space():
    for env in setup():
        obs = env.reset()
        obs_space = env.observation_space(0)
        packed = pufferlib.emulation.pack_obs_space(obs_space)

        ob = list(obs.values())[0]
        flat = pufferlib.emulation.flatten_ob(ob)
        assert packed.shape == flat.shape

def test_pack_atn_space():
    for env in setup():
        obs = env.reset()
        atn_space = env.action_space(0)
        packed = pufferlib.emulation.pack_atn_space(atn_space)

def test_pack_and_batch_obs():
    for env in setup():
        obs = env.reset()
        packed = pufferlib.emulation.pack_and_batch_obs(obs)
        assert type(packed) == np.ndarray
        assert len(packed) == len(obs)

def test_unpack_batched_obs():
    for env in setup():
        obs = env.reset()
        packed = pufferlib.emulation.pack_and_batch_obs(obs)

        obs_space = env.observation_space(0)
        unpacked = pufferlib.emulation.unpack_batched_obs(obs_space, packed)


if __name__ == '__main__':
    test_unpack_batched_obs()
    test_pack_and_batch_obs()
    test_pack_obs_space()
    test_pack_atn_space()