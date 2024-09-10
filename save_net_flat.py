import torch
from torch.nn import functional as F
import numpy as np

def save_model_weights(model, filename):
    weights = []
    for name, param in model.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    weights = np.concatenate(weights)
    weights.tofile(filename)
    # Save the model architecture (you may want to adjust this based on your specific model)
    #with open(filename + "_architecture.txt", "w") as f:
    #    for name, param in model.named_parameters():
    #        f.write(f"{name}: {param.shape}\n")

def test_model(model):
    model = model.cpu().policy
    batch_size = 16
    obs_window = 11
    obs_window_channels = 4
    obs_flat = 26
    x = torch.arange(
        0, batch_size*(obs_window*obs_window*obs_window_channels + obs_flat)
        ).reshape(batch_size, -1) % 16

    cnn_features = x[:, :-obs_flat].view(
        batch_size, obs_window, obs_window, obs_window_channels).long()
    map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
    extra_map_features = (cnn_features[:, :, :, -3:].float() / 255.0).permute(0, 3, 1, 2)
    cnn_features = torch.cat([map_features, extra_map_features], dim=1)
    cnn = model.policy.cnn

    cnn_features = torch.from_numpy(
        np.arange(batch_size*11*11*19).reshape(
        batch_size, 19, obs_window, obs_window)
    ).float()
    conv1_out = cnn[0](cnn_features)

    #(cnn[0].weight[0] * cnn_features[0, :, :5, :5]).sum() + cnn[0].bias[0]

    breakpoint()
    hidden = model.encoder(x)
    output = model.decoder(hidden)
    atn = output.argmax(dim=1)
    print('Encode weight sum:', model.encoder.weight.sum())
    print('encode decode weight and bias sum:', model.encoder.weight.sum() + model.encoder.bias.sum() + model.decoder.weight.sum() + model.decoder.bias.sum())
    print('X sum:', x.sum())
    print('Hidden sum:', hidden.sum())
    print('Hidden 1-10:', hidden[0, :10])
    print('Output sum:', output.sum())
    print('Atn sum:', atn.sum())
    breakpoint()
    exit(0)

def test_lstm():
    batch_size = 16
    input_size = 128
    hidden_size = 128

    input = torch.arange(batch_size*input_size).reshape(1, batch_size, -1).float()/ 100000
    state = (
        torch.arange(batch_size*hidden_size).reshape(1, batch_size, -1).float()/ 100000,
        torch.arange(batch_size*hidden_size).reshape(1, batch_size, -1).float() / 100000
    )
    weights_input = torch.arange(4*hidden_size*input_size).reshape(4*hidden_size, -1).float()/ 100000
    weights_state = torch.arange(4*hidden_size*hidden_size).reshape(4*hidden_size, -1).float()/ 100000
    bias_input = torch.arange(4*hidden_size).reshape(4*hidden_size).float() / 100000
    bias_state = torch.arange(4*hidden_size).reshape(4*hidden_size).float() / 100000

    lstm = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
    lstm.weight_ih_l0.data = weights_input
    lstm.weight_hh_l0.data = weights_state
    lstm.bias_ih_l0.data = bias_input
    lstm.bias_hh_l0.data = bias_state

    output, new_state = lstm(input, state)

    input = input.squeeze(0)
    h, c = state

    buffer = (
        torch.matmul(input, weights_input.T) + bias_input
        + torch.matmul(h, weights_state.T) + bias_state
    )[0]

    i, f, g, o = torch.split(buffer, hidden_size, dim=1)

    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)

    c = f*c + i*g
    h = o*torch.tanh(c)

    breakpoint()
    print('Output:', output)

def test_model_forward(model):
    data = torch.arange(10*(11*11*4 + 26)) % 16
    data[(11*11*4 + 26):] = 0
    data = data.reshape(10, -1).float()
    output = model(data)
    breakpoint()
    pass

	
if __name__ == '__main__':
    #test_lstm()
    model = torch.load('moba.pt', map_location='cpu')
    test_model_forward(model)
    #test_model(model)

    #save_model_weights(model, 'moba_weights.bin')
