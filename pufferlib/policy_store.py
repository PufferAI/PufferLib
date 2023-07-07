from abc import abstractmethod
from ast import List
from dataclasses import dataclass
from email.policy import Policy
from importlib import metadata
import stat
from typing import Dict, OrderedDict, Set

from click import File
from pufferlib.policy_pool import PolicyPool
from pufferlib.rating import OpenSkillRating

import copy

import numpy as np

import pufferlib

class PolicyRegistry():
  def policy_class(self, name: str) -> Policy:
    raise NotImplementedError

class FilePolicySerializer():
  def __init__(self, policy_registry: PolicyRegistry):
    self._policy_registry = policy_registry

  def save_policy(self, policy: Policy, path: str, metadata = None):
    temp_path = path + ".tmp"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "policy_class": self._policy_registry.class_name(policy),
        "metadata": metadata
    })
    os.rename(temp_path, path)

  def load_policy(self, path: str) -> Policy:
    if not os.path.exists(path):
      raise ValueError(f"Policy with name {name} does not exist")
    data = torch.load(path)
    policy = self._policy_registry.new_policy(data["policy_class"])
    policy.load_state_dict(data["policy_state_dict"])
    return policy


class PolicyRecord():
  def __init__(self, name: str, metadata = None):
    self.name = name
    self.metadata = metadata

  def policy(self) -> Policy:
      raise NotImplementedError

class PolicyStore():
  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    raise NotImplementedError

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def select_policies(self, num_policies) -> List[PolicyRecord]:
    raise NotImplementedError

class MemoryPolicyStore(PolicyStore):
  def __init__(self):
    super().__init__()
    self._policies = OrderedDict()

  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    if name in self._policies:
        raise ValueError(f"Policy with name {name} already exists")

    pr = PolicyRecord(name, policy, metadata)
    pr.policy = lambda: policy
    self._policies[name] = pr

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    self.add_policy(
      name,
      copy.deepcopy(self._policies[src_name].policy()),
      copy.deepcopy(self._policies[src_name].metadata))

  def select_policies(self, num_policies, exclude=None) -> List[PolicyRecord]:
    if exclude is None:
        exclude = []

    available_policy_names = set(self._library._model_paths.keys()) - set(exclude)
    return np.random.choice(
        available_policy_names, num_needed, replace=True).tolist()

class DirecroryPolicyStore(PolicyStore):
  def __init__(self, path: str, policy_registry: PolicyRegistry):
    self._path = path
    self._serializer = FilePolicySerializer(policy_registry)

  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    path = os.path.join(self._path, name, ".pt")
    if os.path.exists(path):
      raise ValueError(f"Policy {path} already exists")
    self._serializer.save_policy(policy, path, metadata)
    return PolicyRecord(name, policy, metadata)

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def _get_policy_paths(self):
    return [file in os.listdir(self._path) if file.endswith(".pt")]

# # Uses a sqlite database to store the metadata and paths of policies
# class SqlitePolicyStore(DirecroryPolicyStore):
#   def __init__(self, db_path: str, policy_files_path: str, policy_registry: PolicyRegistry):
#     super().__init__(policy_files_path, policy_registry)

#     self._db_engine = create_engine(path, echo=False)
#     Base.metadata.create_all(self._db_engine)
#     Session = sessionmaker(bind=self._db_engine)
#     self._db_session = Session()
#     self._db_connection = self._db_engine.connect()
#     self._db_connection.execute(text("PRAGMA journal_mode=WAL;"))

#   def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
#     self._db_session.add(policy)
#     self._db_session.commit()

#   def query_policy_by_name(self, name):
#     return self._db_session.query(PolicyRecord).filter_by(name=name).first()

#   def query_tenured_policies(self):
#     return self._db_session.query(PolicyRecord).filter(
#         cast(PolicyRecord.additional_data['tenured'], Boolean) == True
#     ).all()

#   def query_untenured_policies(self):
#     return self._db_session.query(PolicyRecord).filter(
#         cast(PolicyRecord.additional_data['tenured'], Boolean) != True
#     ).all()

#   def delete_policy(self, policy):
#     self._db_session.delete(policy)
#     self._db_session.commit()

#   def query_all_policies(self):
#     return self._db_session.query(PolicyRecord).all()

#   def update_policy(self, policy):
#     self._db_session.commit()
