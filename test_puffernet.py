import torch
import numpy as np

from pufferlib.environments.ocean.moba import puffernet

# TODO: Should probably add a safe mode that type checks input arrays
# It's user error, but it is a big foot gun

def make_dummy_data(*shape, seed=42):
    np.random.seed(seed)
    n = np.prod(shape)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)

def make_dummy_int_data(num_classes, *shape):
    np.random.seed(42)
    n = np.prod(shape)
    ary = np.random.randint(0, num_classes, shape).astype(np.int32)
    return np.ascontiguousarray(ary)

def assert_near(a, b):
    assert a.shape == b.shape
    assert np.all(np.abs(a - b) < 1e-4)

def test_puffernet_relu(batch_size=16, input_size=128):
    input_puffer = make_dummy_data(batch_size, input_size)

    input_torch = torch.from_numpy(input_puffer)
    output_torch = torch.relu(input_torch).detach()
    
    # PufferNet done second because it is in-place on the input
    puffernet.puf_relu(input_puffer, input_puffer, batch_size*input_size)

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
    input_np = make_dummy_data(batch_size, input_size, seed=42)
    weights_np = make_dummy_data(hidden_size, input_size, seed=43)
    bias_np = make_dummy_data(hidden_size, seed=44)
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
    input_np = make_dummy_data(batch_size, input_size, seed=42)
    state_h_np = make_dummy_data(batch_size, hidden_size, seed=43)
    state_c_np = make_dummy_data(batch_size, hidden_size, seed=44)
    weights_input_np = make_dummy_data(4*hidden_size, input_size, seed=45)
    weights_state_np = make_dummy_data(4*hidden_size, hidden_size, seed=46)
    bias_input_np = make_dummy_data(4*hidden_size, seed=47)
    bias_state_np = make_dummy_data(4*hidden_size, seed=48)
    buffer_np = make_dummy_data(4*batch_size*hidden_size, seed=49)

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

def test_puffernet_one_hot(batch_size=16, input_size=128, num_classes=10):
    input_np = make_dummy_int_data(num_classes, batch_size, input_size)
    output_puffer = np.zeros((batch_size, input_size, num_classes), dtype=np.int32)
    puffernet.puf_one_hot(input_np, output_puffer, batch_size, input_size, num_classes)

    input_torch = torch.from_numpy(input_np).long()
    output_torch = torch.nn.functional.one_hot(input_torch, num_classes).int().detach()

    assert_near(output_puffer, output_torch.numpy())

def test_puffernet_cat_dim1(batch_size=16, x_size=32, y_size=64):
    x_np = make_dummy_data(batch_size, x_size)
    y_np = make_dummy_data(batch_size, y_size)
    output_puffer = np.zeros((batch_size, x_size + y_size), dtype=np.float32)
    puffernet.puf_cat_dim1(x_np, y_np, output_puffer, batch_size, x_size, y_size)

    x_torch = torch.from_numpy(x_np)
    y_torch = torch.from_numpy(y_np)
    output_torch = torch.cat([x_torch, y_torch], dim=1).detach()

    assert_near(output_puffer, output_torch.numpy())

def test_puffernet_argmax_multidiscrete(batch_size=16, logit_sizes=[5,7,2]):
    logit_sizes = np.array(logit_sizes).astype(np.int32)
    num_actions = len(logit_sizes)
    input_np = make_dummy_data(batch_size, logit_sizes.sum())
    output_puffer = np.zeros((batch_size, num_actions), dtype=np.int32)
    puffernet.puf_argmax_multidiscrete(input_np, output_puffer, batch_size, logit_sizes, num_actions)

    input_torch = torch.from_numpy(input_np)
    action_slices = torch.split(input_torch, logit_sizes.tolist(), dim=1)
    output_torch = torch.stack([torch.argmax(s, dim=1) for s in action_slices], dim=1).detach()

    assert_near(output_puffer, output_torch.numpy())

if __name__ == '__main__':
    test_puffernet_relu()
    test_puffernet_sigmoid()
    test_puffernet_linear_layer()
    test_puffernet_convolution_layer()
    test_puffernet_lstm()
    test_puffernet_one_hot()
    test_puffernet_cat_dim1()
    test_puffernet_argmax_multidiscrete()
