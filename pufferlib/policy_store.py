from typing import Dict, Set, List, Callable

import torch

from pufferlib.models import Policy
import logging

import copy
import os
import numpy as np

class PolicyRecord():
  def __init__(self, name: str, policy: Policy, policy_args = None):
    self.name = name
    self._policy_args = policy_args
    self._policy = policy

  def policy(self) -> Policy:
    return self._policy

  def policy_args(self) -> Dict:
    return self._policy_args

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
  def add_policy(self, name: str, policy: Policy, policy_args = None) -> PolicyRecord:
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

  def add_policy(self, name: str, policy: Policy, policy_args = None) -> PolicyRecord:
    if name in self._policies:
        raise ValueError(f"Policy with name {name} already exists")

    self._policies[name] = PolicyRecord(name, policy, policy_args)
    return self._policies[name]

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    return self.add_policy(
      name,
      copy.deepcopy(self._policies[src_name].policy()),
      copy.deepcopy(self._policies[src_name].policy_args))

  def _all_policies(self) -> List:
    return self._policies

class FilePolicyRecord(PolicyRecord):
  def __init__(self, name: str, path: str, policy: Policy = None, policy_args = None):
    super().__init__(name, policy, policy_args)
    self._path = path

  def save(self):
    assert self._policy is not None
    if os.path.exists(self._path):
      raise ValueError(f"Policy {self._path} already exists")
    logging.info(f"Saving policy to {self._path}")
    temp_path = self._path + ".tmp"
    torch.save({
        "policy_state_dict": self._policy.state_dict(),
        "policy_args": self._policy_args
    }, temp_path)
    os.rename(temp_path, self._path)

  def load(self, create_policy_func: Callable, envs: None, device = None):
    data = self._load_data(device)
    policy = create_policy_func(envs)  # CHECK ME: do we need additional args to pass in?
    policy.load_state_dict(data["policy_state_dict"])
    policy.is_recurrent = hasattr(policy, "lstm")
    return policy

  def _load_data(self, device = None):
    if not os.path.exists(self._path):
      raise ValueError(f"Policy with name {self.name} does not exist")
    data = torch.load(self._path, map_location=device)
    self._policy_args = data["policy_args"]
    return data

  def policy_args(self) -> Dict:
    if self._policy_args is None:
      self._load_data()
    return self._policy_args

  def policy(self, create_policy_func: Callable = None, envs = None, device = None) -> Policy:
    if device is None:
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if self._policy is None:
      self._policy = self.load(create_policy_func, envs, device).to(device)
    return self._policy

class DirectoryPolicyStore(PolicyStore):
  def __init__(self, path: str):
    self._path = path
    os.makedirs(self._path, exist_ok=True)

  def add_policy(self, name: str, policy: Policy, policy_args = None) -> PolicyRecord:
    path = os.path.join(self._path, name + ".pt")
    pr = FilePolicyRecord(name, path, policy, policy_args)
    pr.save()
    return pr

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def _all_policies(self) -> List:
    policies = dict()
    for file in os.listdir(self._path):
      if file.endswith(".pt"):
        name = file[:-3]
        policies[name] = FilePolicyRecord(name, os.path.join(self._path, file))
    return policies
