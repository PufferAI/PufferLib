import torch
from torch.nn import functional as F
import numpy as np

def save_model_weights(model, filename):
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(param.shape)
    
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

	
if __name__ == '__main__':
    model = torch.load('moba.pt')
    test_model(model)

    #save_model_weights(model, 'moba_weights.pt')
