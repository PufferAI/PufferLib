from pdb import set_trace as T
from typing import Dict, Set, List, Callable

import torch

import logging

import copy
import os
import numpy as np

from pufferlib.models import Policy


class PolicyRecord():
  def __init__(self, name: str, policy: Policy):
    self.name = name
    self._policy = policy

  def policy(self, policy_args=[], policy_kwargs={}, device=None) -> Policy:
    return self._policy

class PolicySelector():
  def __init__(self, num: int, exclude_names: Set[str] = None):
    self._num = num
    self._exclude_names = exclude_names or set()

  def select_policies(self, policies: Dict[str, PolicyRecord]) -> List[PolicyRecord]:
    available_names = list(set(policies.keys()) - set(self._exclude_names))
    if len(available_names) < self._num:
      return []
    selected_names = np.random.choice(available_names, self._num, replace=True).tolist()
    return [policies[name] for name in selected_names]

class PolicyStore():
  def add_policy(self, name: str, policy: Policy) -> PolicyRecord:
    raise NotImplementedError

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def _all_policies(self) -> Dict[str, PolicyRecord]:
    raise NotImplementedError

  def select_policies(self, selector: PolicySelector) -> List[PolicyRecord]:
    return selector.select_policies(self._all_policies())

  def get_policy(self, name: str) -> PolicyRecord:
    return self._all_policies()[name]

class MemoryPolicyStore(PolicyStore):
  def __init__(self):
    super().__init__()
    self._policies = dict()

  def add_policy(self, name: str, policy: Policy) -> PolicyRecord:
    if name in self._policies:
        raise ValueError(f"Policy with name {name} already exists")

    self._policies[name] = PolicyRecord(name, policy)
    return self._policies[name]

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    policy_copy = copy.deepcopy(self._policies[src_name].policy())
    return self.add_policy(name, policy_copy)

  def _all_policies(self) -> List:
    return self._policies

class FilePolicyRecord(PolicyRecord):
  def __init__(self, name: str, path: str, policy: Policy = None):
    super().__init__(name, policy)
    self._path = path

  def save(self):
    assert self._policy is not None
    if os.path.exists(self._path):
      raise ValueError(f"Policy {self._path} already exists")
    logging.info(f"Saving policy to {self._path}")
    temp_path = self._path + ".tmp"
    torch.save(self._policy, temp_path)
    os.rename(temp_path, self._path + '.pt')

  def policy(self, policy_args=[], policy_kwargs={}, device=None) -> Policy:
    if self._policy is not None:
      return self._policy
    if device is None:
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.load(self._path + '.pt', map_location=device)

class DirectoryPolicyStore(PolicyStore):
  def __init__(self, path: str):
    self._path = path
    os.makedirs(self._path, exist_ok=True)

  def add_policy(self, name: str, policy: Policy) -> PolicyRecord:
    path = os.path.join(self._path, name)
    pr = FilePolicyRecord(name, path, policy)
    pr.save()
    return pr

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def _all_policies(self) -> List:
    policies = dict()
    for file in os.listdir(self._path):
      if file.endswith(".pt"):
        name = file[:-3]
        policies[name] = FilePolicyRecord(name, os.path.join(self._path, name))
    return policies
