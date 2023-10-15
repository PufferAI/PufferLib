from pdb import set_trace as T
from typing import Dict, Set, List, Callable

import torch

import logging

import copy
import os
import numpy as np

from pufferlib.models import Policy


class PolicySelector():
    def __init__(self, num: int, exclude_names: set = None):
        self._exclude_names = exclude_names or set()
        self._num = num

    def select_policies(self, policy_records: dict) -> list:
        available_names = set(policy_records) - set(self._exclude_names)
        if len(available_names) < self._num:
          return []

        selected_names = np.random.choice(
                list(available_names), self._num, replace=True).tolist()

        return [policy_records[name] for name in selected_names]

class PolicyRecord:
    def __init__(self, name: str, path: str, policy: Policy = None):
        self.name = name
        self._policy = policy
        self._path = path

    def save(self):
        assert self._policy is not None
        if os.path.exists(self._path):
            raise ValueError(f"Policy {self._path} already exists")

        logging.info(f"Saving policy to {self._path}")
        torch.save(self._policy, self._path)

    def policy(self, policy_args=[], policy_kwargs={}, device=None) -> Policy:
        if self._policy is not None:
            return self._policy
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return torch.load(self._path + '.pt', map_location=device)

class PolicyStore:
    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.path = path

    def add_policy(self, name: str, policy: Policy) -> PolicyRecord:
        path = os.path.join(self.path, name)
        torch.save(policy, path + '.pt')
        pr = PolicyRecord(name, path, policy)

    def policy_names(self) -> set:
        names = set()
        for file in os.listdir(self.path):
            if file.endswith(".pt"):
                names.add(file[:-3])

        return names

    def get_policy(self, name: str) -> PolicyRecord:
        path = os.path.join(self.path, name)
        try:
            return torch.load(path)
        except:
            return torch.load(path, map_location=torch.device('cpu'))
