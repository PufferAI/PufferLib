import torch
import numpy as np

from pufferlib.environments.ocean.moba import puffernet

def make_dummy_data(*shape):
    np.random.seed(42)
    n = np.prod(shape)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)

def assert_near(a, b):
    assert a.shape == b.shape
    assert np.all(np.abs(a - b) < 1e-4)

def test_puffernet_relu(batch_size=16, input_size=128):
    input_puffer = make_dummy_data(batch_size, input_size)

    input_torch = torch.from_numpy(input_puffer)
    output_torch = torch.relu(input_torch).detach()
    
    # PufferNet done second because it is in-place on the input
    puffernet.puf_relu(input_puffer, batch_size*input_size)

    assert_near(input_puffer, output_torch.numpy())

def test_puffernet_sigmoid(n=1024, epsilon=1e-4):
    input_np = make_dummy_data(n)

    input_torch = torch.from_numpy(input_np)
    output_torch = torch.sigmoid(input_torch).detach()

    for i in range(n):
        out_torch = output_torch[i]
        out_puffer = puffernet.puf_sigmoid(input_np[i])
        assert abs(out_puffer - out_torch) < epsilon

def test_puffernet_linear_layer(batch_size=16, input_size=128, hidden_size=128):
    input_np = make_dummy_data(batch_size, input_size)
    weights_np = make_dummy_data(hidden_size, input_size)
    bias_np = make_dummy_data(hidden_size)
    output_puffer = np.zeros((batch_size, hidden_size), dtype=np.float32)
    puffernet.puf_linear_layer(input_np, weights_np, bias_np, output_puffer,
        batch_size, input_size, hidden_size)

    input_torch = torch.from_numpy(input_np)
    weights_torch = torch.from_numpy(weights_np)
    bias_torch = torch.from_numpy(bias_np)
    torch_linear = torch.nn.Linear(input_size, hidden_size)
    torch_linear.weight.data = weights_torch
    torch_linear.bias.data = bias_torch
    output_torch = torch_linear(input_torch).detach()

    assert_near(output_puffer, output_torch.numpy())

def test_puffernet_convolution_layer(batch_size=16, in_width=11, in_height=11,
        in_channels=19, out_channels=32, kernel_size=5, stride=3):
    input_np = make_dummy_data(batch_size, in_channels, in_height, in_width)
    weights_np = make_dummy_data(out_channels, in_channels, kernel_size, kernel_size)
    bias_np = make_dummy_data(out_channels)
    out_height = int((in_height - kernel_size)/stride + 1)
    out_width = int((in_width - kernel_size)/stride + 1)
    output_puffer = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
    puffernet.puf_convolution_layer(input_np, weights_np, bias_np, output_puffer,
        batch_size, in_width, in_height, in_channels, out_channels, kernel_size, stride)

    input_torch = torch.from_numpy(input_np)
    weights_torch = torch.from_numpy(weights_np)
    bias_torch = torch.from_numpy(bias_np)
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    torch_conv.weight.data = weights_torch
    torch_conv.bias.data = bias_torch
    output_torch = torch_conv(input_torch).detach()

    assert_near(output_puffer, output_torch.numpy())

def test_puffernet_lstm(batch_size=16, input_size=128, hidden_size=128):
    input_np = make_dummy_data(batch_size, input_size)
    state_h_np = make_dummy_data(batch_size, hidden_size)
    state_c_np = make_dummy_data(batch_size, hidden_size)
    weights_input_np = make_dummy_data(4*hidden_size, input_size)
    weights_state_np = make_dummy_data(4*hidden_size, hidden_size)
    bias_input_np = make_dummy_data(4*hidden_size)
    bias_state_np = make_dummy_data(4*hidden_size)
    buffer_np = make_dummy_data(4*batch_size*hidden_size)

    input_torch = torch.from_numpy(input_np).view(1, batch_size, input_size)
    state_h_torch = torch.from_numpy(state_h_np).view(1, batch_size, hidden_size)
    state_c_torch = torch.from_numpy(state_c_np).view(1, batch_size, hidden_size)
    weights_input_torch = torch.from_numpy(weights_input_np)
    weights_state_torch = torch.from_numpy(weights_state_np)
    bias_input_torch = torch.from_numpy(bias_input_np)
    bias_state_torch = torch.from_numpy(bias_state_np)
    torch_lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1)
    torch_lstm.weight_ih_l0.data = weights_input_torch
    torch_lstm.weight_hh_l0.data = weights_state_torch
    torch_lstm.bias_ih_l0.data = bias_input_torch
    torch_lstm.bias_hh_l0.data = bias_state_torch
    output_torch, (state_h_torch, state_c_torch) = torch_lstm(input_torch, (state_h_torch, state_c_torch))
    state_h_torch = state_h_torch.detach()
    state_c_torch = state_c_torch.detach()

    # PufferNet done second because it is in-place on the state vars
    puffernet.puf_lstm(input_np, state_h_np, state_c_np, weights_input_np,
        weights_state_np, bias_input_np, bias_state_np, buffer_np,
        batch_size, input_size, hidden_size)

    assert_near(state_h_np, state_h_torch.numpy()[0])
    assert_near(state_c_np, state_c_torch.numpy()[0])

if __name__ == '__main__':
    test_puffernet_relu()
    test_puffernet_sigmoid()
    test_puffernet_linear_layer()
    test_puffernet_convolution_layer()
    test_puffernet_lstm()
