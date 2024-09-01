import torch
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
    x = torch.arange(0, 121*16).view(16, 121).float() / 1000
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
    model = torch.load('snake_mlp.pt')
    #test_model(model)
    save_model_weights(model, 'flat_snake.pt')
