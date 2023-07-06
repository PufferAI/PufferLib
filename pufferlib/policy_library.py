from ast import List
from typing import OrderedDict
from pufferlib.rating import OpenSkillRating

import copy

import numpy as np

import pufferlib

class PolicyLoader():
    def load_policy():
        raise NotImplementedError

class PolicyLibrary():
  def __init__(self):
      self._policies = set()

  def add_policy(self, name, overwrite_existing=True):
      if not overwrite_existing and name in self._policies:
          raise ValueError(f"Policy with name {name} already exists in the library")
      self._policies.add(name)

  def select_policies(self, num_needed, exclude=None):
    if exclude is None:
        exclude = []

    available_policy_names = set(self._library._model_paths.keys()) - set(exclude)
    return np.random.choice(
        available_policy_names, num_needed, replace=True).tolist()

  class PolicyPool(pufferlib.policy_pool.PolicyPool):
    def __init__(self,
                library,
                batch_size: int,
                sample_weights: List[int],
                policy_loader: PolicyLoader):

        super().__init__(batch_size, sample_weights)
        self._library = library
        self._policy_loader = policy_loader
        self._loaded_policies = OrderedDict()

    # Update the active policies to be used for the next batch. Always
    # include the required policies, and then randomly sample the rest
    # from the available policies.
    def update_active_policies(self, required_policy_names=None):
      if required_policy_names is None:
          required_policy_names = []

      num_needed = self._num_active_policies - len(required_policy_names)
      new_policy_names = required_policy_names + self._library.select_policies(num_needed, exclude=required_policy_names)
      new_policies = OrderedDict()
      for policy_name in new_policy_names:
        new_policies[policy_name] = self._loaded_policies.get(
            policy_name,
            self._policy_loader.load_policy(policy_name))
      self._active_policies = list(new_policies.values())
      self._loaded_policies = new_policies

# A PolicyLibrary that maintains OpenSkill ratings for each policy
class OpenSkillPolicyLibrary(PolicyLibrary):
  def __init__(self, mu=1000, anchor_mu=1000, sigma=100/3):
    super().__init__()
    self._tournament = OpenSkillRating(mu, anchor_mu, sigma)
    self._default_mu = mu
    self._default_sigma = sigma
    self._anchor_mu = anchor_mu

  def add_policy(
          self, name: str, model_path: str, overwrite_existing=True,
          mu=None, sigma=None, anchor=False):

    super.add_policy(name, model_path, overwrite_existing)

    # TODO: Figure out anchoring
    if anchor:
        self._tournament.set_anchor(name)
    else:
        self._tournament.add_policy(name)
        self._tournament.ratings[name].mu = mu if mu is not None else self._default_mu
        self._tournament.ratings[name].sigma = sigma if sigma is not None else self._default_sigma

  def update_ranks(self):
    # Update the tournament rankings
    self._tournament.update(
        list(self._scores.keys()),
        list(self._scores.values())
    )

    def ratings(self):
        return self._tournament.ratings

