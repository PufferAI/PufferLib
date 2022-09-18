from pdb import set_trace as T
import numpy as np

import gym
import requests
import random
import time
import os
import sys

from collections import defaultdict

import ray
from ray.air.checkpoint import Checkpoint
from ray.train.rl import RLCheckpoint
from ray.serve import PredictorDeployment
from ray import serve
from ray.rllib.policy.policy import PolicySpec

from pufferlib.rating import OpenSkillRating
from pufferlib.rllib import RLPredictor


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

def create_policies(n):
    return {f'policy_{i}': 
        PolicySpec(
            policy_class=None,
            observation_space=None,
            action_space=None,
            config={"gamma": 0.85},
        )
        for i in range(n)
    }

def serve_rl_model(checkpoint: Checkpoint, name="RLModel") -> str:
    """Serve a RL model and return deployment URI.
    This function will start Ray Serve and deploy a model wrapper
    that loads the RL checkpoint into a RLPredictor.
    """
    serve.start(detached=True)
    deployment = PredictorDeployment.options(name=name)
    deployment.deploy(RLPredictor, checkpoint)
    return deployment.url

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


def run_tournament(policy_mapping_fn, env_creator, endpoint_uri_list, num_games=5, horizon=16):
    agents = [i for i in range(len(endpoint_uri_list))]
    ratings = OpenSkillRating(agents, 0)

    for episode in range(num_games):
        rewards = run_game(episode, policy_mapping_fn, env_creator, endpoint_uri_list)

        ratings.update(
                policy_ids=list(rewards),
                scores=list(rewards.values())
        )

        print(ratings)

    return ratings

def query_action(endpoint, obs: np.ndarray):
    """Perform inference on a served RL model.
    This will send a HTTP request to the Ray Serve endpoint of the served
    RL policy model and return the result.
    """
    #action_dict = requests.post(endpoint_uri, json={"array": obs.tolist()}).json()
    #obs = {k: v.ravel().tolist() for k, v in obs.items()}
    #action_dict = requests.post(endpoint_uri, json=obs).json()
    vals = [v.tolist() for v in obs.values()]
    if type(endpoint) is RLPredictor:
        action_vals = endpoint.predict(np.array(vals))
    else:
        action_vals = requests.post(endpoint, json={"array": vals}).json()
    #action_vals = np.array(action_vals).reshape(8, -1)
    #action_dict = {key: action_vals[:, i] for i, key in enumerate(obs)}
    action_vals = np.array(action_vals)
    action_dict = {key: action_vals[i] for i, key in enumerate(obs)}
    #action_dict = {key: val for key, val in zip(list(obs.keys()), action_vals)}

    #action_dict = requests.post(endpoint_uri, json={"array": [[1]]}).json()
    return action_dict

class Tournament:
    def __init__(self, num_policies, env_creator,
            policy_mapping_fn, policy_sampling_fn=random.sample,
            mu=1000, anchor_mu=1500, sigma=100/3, deploy=False):
        '''Runs matches for a pool of served policies'''
        self.num_policies = num_policies
        self.env_creator = env_creator
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_sampling_fn = policy_sampling_fn
        self.deploy = deploy

        self.rating = OpenSkillRating(mu, anchor_mu, sigma)
        self.policies = {}

        if deploy:
            serve.start(detached=True)

    def server(self, checkpoint_path, out_file=sys.stdout, sleep_seconds=10):
        episode = 0
        anchored = False
        while True:
            files = os.listdir(checkpoint_path) 
            files = [e for e in files if e.startswith('checkpoint')]

            for f in files:
                if f in self.policies:
                    continue

                path = os.path.join(checkpoint_path, f)
                checkpoint = RLCheckpoint(path)
                if anchored:
                    self.add(f, checkpoint)
                else:
                    self.add(f, checkpoint, anchor=True)
                    anchored = True

            if len(self.policies) >= self.num_policies:
                ratings = self.run_match(episode)

                if out_file == sys.stdout:
                    #Avoids stdout glitches by writing directly
                    print(ratings)
                else:
                    out_file.write(str(ratings))
                episode += 1
            else:
                time.sleep(sleep_seconds)

    def add(self, name, policy_checkpoint, anchor=False):
        '''Add policy to pool of served models'''
        assert name not in self.policies

        if self.deploy:
            deployment = PredictorDeployment.options(name=name)
            deployment.deploy(RLPredictor, policy_checkpoint)
            endpoint = deployment
        else:
            endpoint = RLPredictor.from_checkpoint(policy_checkpoint)

        self.policies[name] = endpoint

        if anchor:
            self.rating.set_anchor(name)
        else:
            self.rating.add_policy(name)

    def remove(self, name):
        '''Remove policy from pool of served models'''
        assert name in self.policies
        endpoint = self.policies[name]

        if self.deploy:
            endpoint.delete()

        del self.policies[name]
        self.rating.remove_policy(name)

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

@ray.remote
class RemoteTournament(Tournament):
    pass