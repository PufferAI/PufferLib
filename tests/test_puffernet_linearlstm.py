import torch
import numpy as np

from pufferlib.ocean import env_creator
from pufferlib.models import Default, LSTMWrapper

import pyximport
pyximport.install(
    setup_args={"include_dirs": [
        np.get_include(),
        'pufferlib/extensions',
    ]},
)

from pufferlib.extensions import puffernet


def make_dummy_data(*shape, seed=42):
    np.random.seed(seed)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)

def assert_near(a, b, tolerance=1e-4):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert np.all(np.abs(a - b) < tolerance), f"Value mismatch exceeds tolerance {tolerance}"

def test_tetris_puffernet(model_path='puffer_tetris_weights.bin'):
    # Load the environment to get parameters
    env = env_creator('puffer_tetris')()

    ### Instantiate the pytorch model
    policy = Default(env)
    policy = LSTMWrapper(env, policy)

    # Load and assign weights to the pytorch model
    with open(model_path, 'rb') as f:
        weights_blob = np.fromfile(f, dtype=np.float32)

    current_pos = 0
    layer_weights = []
    for name, param in policy.named_parameters():
        print(name, param.shape)
        num_params = param.numel()
        weights = weights_blob[current_pos:current_pos+num_params]
        param.data = torch.from_numpy(weights).view_as(param)
        layer_weights.append(weights)
        current_pos += num_params

    ### Prepare dummy input data
    batch_size = 1
    obs_shape = env.single_observation_space.shape
    atn_dim = env.single_action_space.n
    dummy_obs_np = make_dummy_data(batch_size, *obs_shape)
    dummy_obs_torch = torch.from_numpy(dummy_obs_np)

    dummy_h_np = make_dummy_data(batch_size, policy.hidden_size, seed=43)
    dummy_h_torch = torch.from_numpy(dummy_h_np)
    dummy_c_np = make_dummy_data(batch_size, policy.hidden_size, seed=44)
    dummy_c_torch = torch.from_numpy(dummy_c_np)

    ### PyTorch Forward Pass
    policy.eval()
    with torch.no_grad():
        hidden_torch = policy.policy.encode_observations(dummy_obs_torch)
        logits_torch, values_torch = policy.forward_eval(dummy_obs_torch, {'lstm_h': dummy_h_torch, 'lstm_c': dummy_c_torch})
    hidden_np = hidden_torch.detach().numpy()
    logits_np = logits_torch.detach().numpy()
    values_np = values_torch.detach().numpy()

    ### PufferNet Forward Pass and compare
    hidden_puffer = np.zeros((batch_size, policy.hidden_size), dtype=np.float32)
    puffernet.puf_linear_layer(dummy_obs_np, layer_weights[0], layer_weights[1], hidden_puffer,
        batch_size, obs_shape[0], policy.hidden_size)
    puffernet.puf_gelu(hidden_puffer, hidden_puffer, hidden_puffer.size)

    assert_near(hidden_np, hidden_puffer)
    print("encoder output matches!")

    # LSTM -> decoder, value
    lstm_buffer = np.zeros((batch_size * policy.hidden_size * 4), dtype=np.float32)
    puffernet.puf_lstm(hidden_puffer, dummy_h_np, dummy_c_np,
        layer_weights[6], layer_weights[7], layer_weights[8], layer_weights[9],
        lstm_buffer, batch_size, policy.hidden_size, policy.hidden_size)
    new_hidden = dummy_h_np

    # actor
    logits_puffer = np.zeros((batch_size, atn_dim), dtype=np.float32)
    puffernet.puf_linear_layer(new_hidden, layer_weights[2], layer_weights[3], logits_puffer,
        batch_size, policy.hidden_size, atn_dim)
    assert_near(logits_np, logits_puffer)
    print("decoder output matches!")

    # value_fn
    values_puffer = np.zeros((batch_size, 1), dtype=np.float32)
    puffernet.puf_linear_layer(new_hidden, layer_weights[4], layer_weights[5], values_puffer,
        batch_size, policy.hidden_size, 1)
    assert_near(values_np, values_puffer)
    print("value_fn output matches!")

if __name__ == '__main__':
    test_tetris_puffernet()
    print("All tests passed!")
