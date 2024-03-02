from pdb import set_trace as T
import os
import torch


def get_policy_names(path: str) -> list:
    # Assumeing that all pt files other than trainer_state.pt in the path are policy files
    names = []
    for file in os.listdir(path):
        if file.endswith(".pt") and file != 'trainer_state.pt':
            names.append(file[:-3])
    return sorted(names)

class PolicyStore:
    def __init__(self, path: str):
        self.path = path

    def policy_names(self) -> list:
        return get_policy_names(self.path)

    def get_policy(self, name: str) -> torch.nn.Module:
        path = os.path.join(self.path, name + '.pt')
        try:
            return torch.load(path)
        except:
            return torch.load(path, map_location=torch.device('cpu'))
