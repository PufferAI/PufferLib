
from typing import Dict
from pufferlib.evaluation import OpenSkillRating

import pickle

from pufferlib.policy_store import PolicySelector

class OpenSkillPolicySelector(PolicySelector):
  pass

class OpenSkillRanker():
  def __init__(self, anchor, mu=1000, anchor_mu=1000, sigma=100/3):
    super().__init__()

    self._tournament = OpenSkillRating(mu, anchor_mu, sigma)
    self._anchor = anchor
    self._default_mu = mu
    self._default_sigma = sigma
    self._anchor_mu = anchor_mu

  def update_ranks(self, scores: Dict[str, float]):
    for policy in scores.keys():
      if policy not in self._tournament.ratings:
          self.add_policy(policy, anchor=policy == self._anchor)

    self._tournament.update(list(scores.keys()), list(scores.values()))

  def add_policy(self, name: str, mu=None, sigma=None, anchor=False):
    if name in self._tournament.ratings:
        raise ValueError(f"Policy with name {name} already exists")

    if anchor:
        self._tournament.set_anchor(name)
        self._tournament.ratings[name].mu = self._anchor_mu
    else:
        self._tournament.add_policy(name)
        self._tournament.ratings[name].mu = mu if mu is not None else self._default_mu
        self._tournament.ratings[name].sigma = sigma if sigma is not None else self._default_sigma

  def ratings(self):
      return self._tournament.ratings

  def selector(self, num_policies, exclude=[]):
    return OpenSkillPolicySelector(num_policies, exclude)

  def save_to_file(self, file_path):
      with open(file_path, 'wb') as f:
          pickle.dump(self, f)

  @classmethod
  def load_from_file(cls, file_path):
      with open(file_path, 'rb') as f:
          instance = pickle.load(f)
      return instance
