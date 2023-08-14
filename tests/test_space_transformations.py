from pdb import set_trace as T

import pufferlib.emulation

import mock_environments

def test_flatten_unflatten():
    for space in mock_environments.MOCK_OBSERVATION_SPACES:
        flat_space = pufferlib.emulation.flatten_space(space)
        sample = flat_space.sample()
        array = pufferlib.emulation.flatten_to_array(sample, flat_space)