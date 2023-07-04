from pdb import set_trace as T
from collections import defaultdict

import torch
import copy

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, Boolean, String, Float, JSON, text, cast
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from pufferlib.rating import OpenSkillRating


Base = declarative_base()

class Policy(Base):
    __tablename__ = 'policies'

    id = Column(Integer, primary_key=True)
    model_path = Column(String)
    model_class = Column(String)
    name = Column(String, unique=True)
    mu = Column(Float)
    sigma = Column(Float)
    episodes = Column(Integer)
    additional_data = Column(JSON)

    def __init__(self, *args, model=None, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)
        if model:
            self.model = model

    def load_model(self, model):
        model.load_state_dict(torch.load(self.model_path))
        self.model = model
 
    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        self.model_class = str(type(model))
        self.model = model


class PolicyPool():
    def __init__(self, evaluation_batch_size, learner, name,
            sample_weights=[], active_policies=4, max_policies=100,
            path='pool', rank_update_episodes=10, policy_update_episodes=100, policy_add_episodes=100,
            mu=1000, anchor_mu=1000, sigma=100/3):

        self.learner = learner
        self.learner_name = name

        # Set up skill rating tournament
        self.tournament = OpenSkillRating(mu, anchor_mu, sigma)
        self.mu = mu
        self.anchor_mu = anchor_mu
        self.sigma = sigma

        # Set up scoring
        self.scores = defaultdict(list)
        self.rank_update_episodes = rank_update_episodes
        self.policy_update_episodes = policy_update_episodes
        self.policy_add_episodes = policy_add_episodes

        self.num_scores = 0
        self.num_active_policies = active_policies
        self.max_policies = max_policies
        self.active_policies = {}
        self.path = path
       
        # Set up the SQLite database and session
        engine = create_engine('sqlite:///policy_pool.db', echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.connection = engine.connect()
        self.connection.execute(text("PRAGMA journal_mode=WAL;"))

        # Assign policies used for evaluation
        self.add_policy(learner, name, tenured=True, mu=mu, sigma=sigma, anchor=False)
        self.update_active_policies()

        # Create indices for splitting data across policies
        chunk_size = sum(sample_weights)
        assert evaluation_batch_size % chunk_size == 0
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        self.sample_idxs = [[] for _ in range(len(sample_weights))]
        for idx in range(evaluation_batch_size):
            sublist_idx = pattern[idx % chunk_size]
            self.sample_idxs[sublist_idx].append(idx)

    def add_policy_copy(self, key, name, tenured=False, anchor=False):
        # Retrieve the policy from the database using the key
        original_policy = self.session.query(Policy).filter_by(name=key).first()
        
        # Use add_policy method to add the new policy
        self.add_policy(original_policy.model, name, tenured=tenured, mu=original_policy.mu, sigma=original_policy.sigma, anchor=anchor)

    def add_policy(self, model, name, tenured=False, mu=None, sigma=None, anchor=False, overwrite_existing=True):
        # Construct the model path by joining the model and name
        model_path = f"{self.path}/{name}"
        
        # Check if a policy with the same name already exists in the database
        existing_policy = self.session.query(Policy).filter_by(name=name).first()

        if existing_policy is not None:
            if overwrite_existing:
                self.session.delete(existing_policy)
                self.session.commit()
            else:
                raise ValueError(f"A policy with the name '{name}' already exists.")

        
        # Set default values for mu and sigma if they are not provided
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma

        # TODO: Eliminate need to deep copy
        model = copy.deepcopy(model)
        policy = Policy(
            model=model,
            model_path=model_path,
            model_class=str(type(model)),
            name=name,
            mu=mu,
            sigma=sigma,
            episodes=0,  # assuming new policies have 0 episodes
            additional_data={'tenured': tenured}
        )
        policy.save_model(model)
        
        # Add the new policy to the database
        self.session.add(policy)
        self.session.commit()

        # Add the policy to the tournament system
        # TODO: Figure out anchoring
        if anchor:
            self.tournament.set_anchor(name)
        else:
            self.tournament.add_policy(name)
            self.tournament.ratings[name].mu = mu
            self.tournament.ratings[name].sigma = sigma

        # If the maximum number of policies is reached, remove the worst policy
        num_policies = self.session.query(Policy).count()
        if num_policies > self.max_policies:
            # Query the worst policy that is not tenured
            worst_policy = self.session.query(Policy).filter(
                cast(Policy.additional_data['tenured'], Boolean) != 'true'
            ).order_by(Policy.mu.asc()).first()
            
            if worst_policy:
                # Remove the worst policy from the database
                self.session.delete(worst_policy)
                self.session.commit()


    def forwards(self, obs, lstm_state=None, dones=None):
        returns = []
        all_actions = []
        policy_keys = list(self.active_policies.keys())
        for idx in range(self.num_active_policies):
            key = policy_keys[0]
            if idx < len(self.active_policies):
                key = policy_keys[idx]

            policy = self.active_policies[key]
            samp = self.sample_idxs[idx]
            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.model.get_action_and_value(
                    obs[samp],
                    [lstm_state[0][:, samp], lstm_state[1][:, samp]],
                    dones[samp])
            else:
                atn, lgprob, _, val = policy.model.get_action_and_value(obs[samp])
            
            returns.append((atn, lgprob, val, lstm_state))
            all_actions.append(atn)
        
        return torch.cat(all_actions), self.sample_idxs, returns

    def load(self, path):
        '''Load all models in path'''
        records = self.session.query(Policy).all()
        for record in records:
            model = eval(record.model_class)
            model.load_state_dict(torch.load(record.model_path))
            
            policy = Policy(model, record.name, record.model_path,
                                      record.mu, record.sigma, ...) # additional attributes

            self.policies[record.name] = policy

    def update_scores(self, infos, info_key, step):
        # TODO: Check that infos is dense and sorted
        agent_infos = []
        for info in infos:
            agent_infos += list(info.values())

        policy_infos = defaultdict(list)
        policy_keys = list(self.active_policies.keys())
        for idx in range(self.num_active_policies):
            if idx >= len(self.active_policies):
                idx = 0

            add_policy = False
            update_policies = False
            update_ranks = False
            samp = self.sample_idxs[idx]
            policy = policy_keys[idx]
            pol_infos = np.array(agent_infos)[samp]
            policy_infos[policy] += list(pol_infos)
            for i in pol_infos:
                if info_key not in i:
                    continue 
            
                self.scores[policy].append(i[info_key])
                self.num_scores += 1

                if self.num_scores % self.policy_update_episodes == 0:
                    update_policies = True

                if self.num_scores % self.rank_update_episodes == 0:
                    update_ranks = True

                if self.num_scores % self.policy_update_episodes == 0:
                    add_policy = True

        if update_ranks:
            self.update_ranks()

        if update_policies:
            self.update_active_policies()

        if add_policy:
            self.add_policy_copy(
                    self.learner_name,
                    f'{self.learner_name}-{step}',
                    tenured=False,
                    anchor=False
            )

        return policy_infos

    def update_ranks(self):
        # Update the tournament rankings
        self.tournament.update(
            list(self.scores.keys()),
            list(self.scores.values())
        )
        
        # Update the mu and sigma values of each policy in the database
        for name, rating in self.tournament.ratings.items():
            policy = self.session.query(Policy).filter_by(name=name).first()
            if policy:
                policy.mu = rating.mu
                policy.sigma = rating.sigma
                self.session.commit()

        print(self.tournament)
        
        # Reset the scores
        self.scores = defaultdict(list)

    def update_active_policies(self):
        # Query the tenured policies
        tenured_policies = self.session.query(Policy).filter(
            cast(Policy.additional_data['tenured'], Boolean) == True
        ).all()

        # Extract names of the tenured policies
        active_policy_names = [policy.name for policy in tenured_policies]

        # Query the untenured policies
        untenured_policies = self.session.query(Policy).filter(
            cast(Policy.additional_data['tenured'], Boolean) != True
        ).all()

        # Randomly sample from untenured policies if needed
        if len(untenured_policies) > 0:
            additional_policy_names = np.random.choice(
                [policy.name for policy in untenured_policies],
                self.num_active_policies - len(active_policy_names),
                replace=True
            ).tolist()
            active_policy_names += additional_policy_names

        # Store the active policies in self.active_policies
        self.active_policies = {name: self.session.query(Policy).filter_by(name=name).first() for name in active_policy_names}
        for policy in self.active_policies.values():
            policy.load_model(copy.deepcopy(self.learner))

    def to_table(self):
        # Query all the policies
        policies = self.session.query(Policy).all()

        # Extract data
        data = []
        for policy in policies:
            model_name = policy.model_path.split('/')[-1]
            experiment = policy.model_path.split('/')[-2]
            checkpoint = int(model_name.split('.')[0])
            rank = self.tournament.ratings[policy.name].mu  # or however you are calculating rank
            num_samples = policy.episodes
            data.append([model_name, rank, num_samples, experiment, checkpoint])

        # Create DataFrame
        table = pd.DataFrame(data, columns=["Model", "Rank", "Num Samples", "Experiment", "Checkpoint"])

        table = table.sort_values(by='Rank', ascending=False)
        print(table[["Model", "Rank"]])
        