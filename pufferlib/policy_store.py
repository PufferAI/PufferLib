from typing import Dict, Set, List, Callable

import torch

from pufferlib.models import Policy

import copy
import os
import numpy as np

class PolicyRecord():
  def __init__(self, name: str, policy: Policy, metadata = None):
    self.name = name
    self.metadata = metadata
    self._policy = policy

  def policy(self) -> Policy:
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
  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    raise NotImplementedError

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    raise NotImplementedError

  def _all_policies(self) -> Dict[str, PolicyRecord]:
    raise NotImplementedError

  def select_policies(self, selector: PolicySelector) -> List[PolicyRecord]:
    return selector.select_policies(self._all_policies())

class MemoryPolicyStore(PolicyStore):
  def __init__(self):
    super().__init__()
    self._policies = dict()

  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    if name in self._policies:
        raise ValueError(f"Policy with name {name} already exists")

    self._policies[name] = PolicyRecord(name, policy, metadata)
    return self._policies[name]

  def add_policy_copy(self, name: str, src_name: str) -> PolicyRecord:
    return self.add_policy(
      name,
      copy.deepcopy(self._policies[src_name].policy()),
      copy.deepcopy(self._policies[src_name].metadata))

  def _all_policies(self) -> List:
    return self._policies

class FilePolicyRecord(PolicyRecord):
  def __init__(self, name: str, path: str, policy: Policy = None, metadata = None):
    super().__init__(name, policy, metadata)
    self._path = path

  def save(self):
    assert self._policy is not None
    if os.path.exists(self._path):
      raise ValueError(f"Policy {self.path} already exists")
    temp_path = self._path + ".tmp"
    torch.save({
        "policy_state_dict": self._policy.state_dict(),
        "policy_class": self._policy.__class__.__name__,
        "metadata": self.metadata
    })
    os.rename(temp_path, self._path)

  def load(self, create_policy_func: Callable[[PolicyRecord], Policy]):
    if not os.path.exists(self._path):
      raise ValueError(f"Policy with name {self.name} does not exist")
    data = torch.load(self._path)
    policy = create_policy_func(self)
    policy.load_state_dict(data["policy_state_dict"])
    return policy

  def policy(self, create_policy_func: Callable[[PolicyRecord], Policy] = None) -> Policy:
    if self._policy is None:
      self._policy = create_policy_func(self)
    return self._policy

class DirectoryPolicyStore(PolicyStore):
  def __init__(self, path: str, create_policy_func: Callable[[PolicyRecord], Policy]):
    self._path = path
    self._create_policy_func = create_policy_func

  def add_policy(self, name: str, policy: Policy, metadata = None) -> PolicyRecord:
    path = os.path.join(self._path, name + ".pt")
    pr = FilePolicyRecord(name, path, policy, metadata)
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
