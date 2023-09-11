from pdb import set_trace as T
import numpy as np
import pickle
import os

import torch
from torch import nn

from pufferlib.frameworks import cleanrl

def save_model(model, optimizer=None, path='model'):
    """
    Save a PyTorch model and optionally its optimizer state to separate files.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to save.
    - optimizer (torch.optim.Optimizer, optional): The optimizer whose state you want to save.
    - path_prefix (str): The path prefix for saved files.

    Returns:
    - None
    """
    pt_path = f"{path}.pt"
    pickle_path = f"{path}.pickle"
    
    # Save the state_dict using PyTorch's save function
    torch.save(model.state_dict(), pt_path)
    
    # Save the model class using pickle
    try:
        with open(pickle_path, 'wb') as f:
            # Dumb hack for now because dependency injection breaks serialization
            # Literally the only two things that don't are inheritance
            # (prevents multi-framework policies) and pure prodecural (makes people mad)
            if isinstance(model, (cleanrl.Policy, cleanrl.RecurrentPolicy)):
                pickle.dump((model.__class__, model.policy.__class__), f)
            else:
                pickle.dump(model.__class__, f)
    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        raise RuntimeError(f"Failed to save the model class due to file error: {e}")
    except pickle.PickleError as e:
        raise RuntimeError(f"Failed to pickle the model class: {e}")
    
    # Optionally save the optimizer state
    if optimizer is not None:
        opt_path = f"{path}_optimizer.pt"
        torch.save(optimizer.state_dict(), opt_path)


def load_model(path, model_args=[], model_kwargs={}, map_location=None):
    """
    Load a PyTorch model from separate files.

    Parameters:
    - path (str): The path prefix for saved files.
    - map_location (torch.device): The device to which the model should be loaded.

    Returns:
    - model (torch.nn.Module): The loaded PyTorch model.
    - optimizer_state (dict, optional): The state of the optimizer if it was saved; otherwise None.
    """
    pt_path = f"{path}.pt"
    pickle_path = f"{path}.pickle"
    opt_path = f"{path}_optimizer.pt"
    
    # Load the model class using pickle
    try:
        with open(pickle_path, 'rb') as f:
            model_class = pickle.load(f)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        raise RuntimeError(f"Failed to load the model class due to file error: {e}")
    except pickle.PickleError as e:
        raise RuntimeError(f"Failed to unpickle the model class: {e}")

    # Dumb hack for loading puffer models
    puffer_class = None
    if isinstance(model_class, tuple):
        puffer_class, model_class = model_class
    
    # Instantiate the model and load the state_dict
    model = model_class(*model_args, **model_kwargs)
    if puffer_class is not None:
        model = puffer_class(model)

    model.load_state_dict(torch.load(pt_path, map_location=map_location))
    model.to(map_location)
    
    # Optionally load the optimizer state
    optimizer_state = None
    if os.path.exists(opt_path):
        optimizer_state = torch.load(opt_path, map_location=map_location)
    
    return model, optimizer_state


class BatchFirstLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        '''
        input: B x T x H
        h&c: B x T x H
        '''
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        hidden, hx = super().forward(input, [h, c])
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        return hidden, [h, c]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''CleanRL's default layer initialization'''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
