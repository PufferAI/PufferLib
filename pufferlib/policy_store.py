from pdb import set_trace as T
from typing import Dict, Set, List, Callable

import torch

import logging

import copy
import os
import numpy as np

class PolicyStore:
    def __init__(self, path: str):
        self.path = path

    def policy_names(self) -> list:
        names = []
        for file in os.listdir(self.path):
            if file.endswith(".pt") and file != 'trainer_state.pt':
                names.append(file[:-3])

        return names

    def get_policy(self, name: str) -> torch.nn.Module:
        path = os.path.join(self.path, name + '.pt')
        try:
            return torch.load(path)
        except:
            return torch.load(path, map_location=torch.device('cpu'))
