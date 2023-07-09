Base = declarative_base()

from ast import List
from pdb import set_trace as T
from collections import defaultdict
from typing import OrderedDict
from pufferlib.rating import OpenSkillRating

import torch
import copy

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, Boolean, String, Float, JSON, text, cast
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pufferlib

class PolicyRecord(Base):
  __tablename__ = 'policies'

  id = Column(Integer, primary_key=True)
  model_path = Column(String)
  model_class = Column(String)
  name = Column(String, unique=True)
  mu = Column(Float)
  sigma = Column(Float)
  episodes = Column(Integer)
  additional_data = Column(JSON)



class DBPolicyLoader(PolicyLoader):
    def __init__(self, database):
        self._database = database

    def load_policy(self, name):
        record = self._database.query_policy_by_name(name)
        torch.load(record.model_path)

class DBPolicyPool(PolicyPool):
    pass
            path='pool',

        # Assign policies used for evaluation
        if learner is not None:
            self.add_policy(learner, name, tenured=True, mu=mu, sigma=sigma, anchor=True)
            self.update_active_policies()


    def add_policy_copy(self, key, name, tenured=False, anchor=False):
        # Retrieve the policy from the database using the key
        original_policy = self.database.query_policy_by_name(key)
        assert original_policy is not None, f"Policy with name '{key}' does not exist."

        # Use add_policy method to add the new policy
        self.add_policy(original_policy.model, name, tenured=tenured, mu=original_policy.mu, sigma=original_policy.sigma, anchor=anchor)

    def load(self, path):
        '''Load all models in path'''
        records = self.session.query(PolicyRecord).all()
        for record in records:
            model = eval(record.model_class)
            model.load_state_dict(torch.load(record.model_path))

            policy = PolicyRecord(model, record.name, record.model_path,
                                      record.mu, record.sigma, ...) # additional attributes

            self.policies[record.name] = policy

    def to_table(self):
        policies = self.session.query(PolicyRecord).all()

        data = []
        for policy in policies:
            model_name = policy.model_path.split('/')[-1]
            experiment = policy.model_path.split('/')[-2]
            checkpoint = int(model_name.split('.')[0])
            rank = self.tournament.ratings[policy.name].mu
            num_samples = policy.episodes
            data.append([model_name, rank, num_samples, experiment, checkpoint])

        table = pd.DataFrame(data, columns=["Model", "Rank", "Num Samples", "Experiment", "Checkpoint"]).sort_values(by='Rank', ascending=False)

        print(table[["Model", "Rank"]])


    # Update the mu and sigma values of each policy in the database
    for name, rating in self._tournament.ratings.items():
        policy = self.database.query_policy_by_name(name)
        if policy:
            policy.mu = rating.mu
            policy.sigma = rating.sigma
            self.database.update_policy(policy)

    # Reset the scores
    self.scores = defaultdict(list)


      for policy in self.active_policies:
          policy.load_model(copy.deepcopy(self.learner))

      for policy_name in required_policy_names:
          if policy_name not in [policy.name for policy in self.active_policies]:
              policy = self.database.query_policy_by_name(policy_name)
              policy.load_model(copy.deepcopy(self.learner))
              self.active_policies.append(policy)
