from pdb import set_trace as T
import numpy as np

import abc
import requests
import random
import time
import os
import sys

from collections import defaultdict

import ray
from ray.air.checkpoint import Checkpoint

from ray.serve import PredictorDeployment
from ray import serve

from pufferlib.rating import OpenSkillRating
from pufferlib.frameworks.rllib import RLPredictor, read_checkpoints


def group(obs, policy_mapping_fn, episode):
    groups = {}
    for k, v in obs.items():
        g = policy_mapping_fn(k, episode)
        if g not in groups:
            groups[g] = {}
        groups[g][k] = v
    return groups

def ungroup(groups):
    ungrouped = {}
    for g in groups.values():
        for k, v in g.items():
            assert k not in ungrouped
            ungrouped[k] = v
    return ungrouped

def serve_rl_model(checkpoint: Checkpoint, name="RLModel") -> str:
    """Serve a RL model and return deployment URI.
    This function will start Ray Serve and deploy a model wrapper
    that loads the RL checkpoint into a RLPredictor.
    """
    serve.start(detached=True)
    deployment = PredictorDeployment.options(name=name)
    deployment.deploy(RLPredictor, checkpoint)
    return deployment.url

def query_action(endpoint, obs: np.ndarray):
    """Perform inference on a served RL model.
    This will send a HTTP request to the Ray Serve endpoint of the served
    RL policy model and return the result.
    """
    vals = [v.tolist() for v in obs.values()]
    if type(endpoint) is RLPredictor:
        action_vals = endpoint.predict(np.array(vals))
    else:
        action_vals = requests.post(endpoint, json={"array": vals}).json()

    action_vals = np.array(action_vals)
    # TODO: Figure out data format for different envs
    return {key: action_vals[i] for i, key in enumerate(obs)}

def run_game(env_creator, policy_mapping_fn, endpoints, episode=0, horizon=1024, render=False):
    """Evaluate a served RL policy on a local environment.
    This function will create an RL environment and step through it.
    To obtain the actions, it will query the deployed RL model.
    """
    env = env_creator()
    obs = env.reset()

    if render:
        env.render()

    policy_rewards = defaultdict(float)
    for t in range(horizon):
        # Compute actions per policy
        grouped_actions = {}
        for idx, vals, in group(obs, policy_mapping_fn, episode).items():
            grouped_actions[idx] = query_action(endpoints[idx], vals)
        actions = ungroup(grouped_actions)

        # Centralized env step
        obs, rewards, dones, _ = env.step(actions)

        if render:
            env.render()

        # Compute policy rewards
        for key, val in rewards.items():
            policy = policy_mapping_fn(key, episode)
            policy_rewards[policy] += val

        if all(list(dones.values())):
            break

    return policy_rewards

class Tournament(abc.ABC):
    def __init__(self, num_policies, env_creator,
            policy_mapping_fn, policy_sampling_fn=random.sample,
            mu=1000, anchor_mu=1500, sigma=100/3):
        '''Runs matches for a pool of served policies'''
        self.num_policies = num_policies
        self.env_creator = env_creator
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_sampling_fn = policy_sampling_fn

        self.rating = OpenSkillRating(mu, anchor_mu, sigma)
        self.policies = {}

        self.init()

    def init(self):
        pass

    def add(self, name, endpoint, anchor):
        '''Add policy to pool of served models'''
        assert name not in self.policies
        self.policies[name] = endpoint

        if anchor:
            self.rating.set_anchor(name)
        else:
            self.rating.add_policy(name)

    def remove(self, name):
        '''Remove policy from pool of served models'''
        assert name in self.policies
        policy = self.policies[name]
        del self.policies[name]
        self.rating.remove_policy(name)
        return policy

    def server(self, tune_path, out_file=sys.stdout, sleep_seconds=10):
        episode = 0
        while True:
            for name, checkpoint in read_checkpoints(tune_path):
                if name not in self.policies:
                    self.add(name, checkpoint, anchor=len(self.policies)==0)

            if len(self.policies) >= self.num_policies:
                ratings = self.run_match(episode)
                out_file.write(str(ratings) + '\n')
                episode += 1
            else:
                time.sleep(sleep_seconds)

    def run_match(self, episode):
        '''Select participants and run a single game to update ratings

        policy_mapping_fn: Maps agent name to policy id
        policy_sampling_fn: Selects a subset of policies to run for the match
        num_policies: number of policies to use in this match'''

        def policy_mapping_fn(agent_name, episode):
            '''Convert agent name to key of sampled policy'''
            policy_idx = self.policy_mapping_fn(agent_name, episode)
            return list(self.policies)[policy_idx]

        rewards = run_game(
            self.env_creator,
            policy_mapping_fn,
            self.policies,
            episode,
        )

        self.rating.update(
                policy_ids=list(rewards),
                scores=list(rewards.values())
        )

        return self.rating


class LocalTournament(Tournament):
    def add(self, name, policy_checkpoint, anchor=False):
        '''Add policy to pool of served models'''
        endpoint = RLPredictor.from_checkpoint(policy_checkpoint)
        super().add(name, endpoint, anchor)


class ServedTournament(Tournament):
    def init(self):
        serve.start(detached=True)

    def add(self, name, policy_checkpoint, anchor=False):
        '''Add policy to pool of served models'''
        deployment = PredictorDeployment.options(name=name)
        deployment.deploy(RLPredictor, policy_checkpoint)
        super().add(name, deployment, anchor)

    def remove(self, name):
        endpoint = super().remove(name)
        endpoint.delete()
