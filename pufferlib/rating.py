from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import openskill


def rank(policy_ids, scores):
    '''Compute policy rankings from per-agent scores'''
    agents = defaultdict(list)
    for policy_id, score in zip(policy_ids, scores):
        agents[policy_id].append(score)

    # Double argsort returns ranks
    return np.argsort(np.argsort(
        [-np.mean(vals) + 1e-8 * np.random.normal() for policy, vals in
        sorted(agents.items())])).tolist()


class OpenSkillRating:
    '''OpenSkill Rating wrapper for estimating relative policy skill

    Provides a simple method for updating skill estimates from raw
    per-agent scores as are typically returned by the environment.'''
    def __init__(self, mu, anchor_mu, sigma, agents=[], anchor=None):
        '''
        Args:
            agents: List of agent classes to rank
            anchor: Baseline policy name to anchor to mu
            mu: Anchor point for the baseline policy (cannot be exactly 0)
            sigma: 68/95/99.7 win rate against 1/2/3 sigma lower SR'''

        assert type(agents) != set, 'Agents must be ordered (e.g. list, not set)'

        self.ratings = {}
        self.mu        = mu
        self.anchor_mu = anchor_mu
        self.sigma     = sigma

        for e in agents:
            self.add_policy(e)

        self.anchor    = anchor
        self._anchor_baseline()

    def __str__(self):
        return ', '.join(f'{p}: {int(r.mu)}' for p, r in self.ratings.items())

    @property
    def stats(self):
        return {p: int(r.mu) for p, r in self.ratings.items()}

    def _anchor_baseline(self):
        '''Resets the anchor point policy to mu SR'''
        for agent, rating in self.ratings.items():
            rating.sigma = self.sigma
            if agent == self.anchor:
                rating.mu    = self.anchor_mu
                rating.sigma = self.sigma

    def set_anchor(self, name):
        '''TODO: multiple anchors'''
        if self.anchor is not None:
            self.remove_policy(self.anchor)
        self.add_policy(name)
        self.anchor = name
        self._anchor_baseline()

    def add_policy(self, name):
        assert name not in self.ratings, f'Policy {name} already added to ratings'
        self.ratings[name] = openskill.Rating(mu=self.mu, sigma=self.sigma)

    def remove_policy(self, name):
        assert name in self.ratings, f'Policy {name} not in ratings'
        del self.ratings[name]

    def update(self, policy_ids, ranks=None, scores=None):
        '''Updates internal skill rating estimates for each policy

        You should call this function once per simulated environment
        Provide either ranks OR policy_ids and scores

        Args:
            ranks: List of ranks in the same order as agents
            policy_ids: List of policy IDs for each agent episode
            scores: List of scores for each agent episode

        Returns:
            Dictionary of ratings keyed by agent names'''

        assert (ranks is None) != (scores is None), 'Specify either ranks or scores'
        assert self.anchor is not None, 'Set the anchor policy before updating ratings'

        # if ranks is None:
        #     ranks = rank(policy_ids, scores)

        teams = [[self.ratings[e]] for e in policy_ids]
        ratings = openskill.rate(teams, score=scores)

        #ratings = [openskill.create_rating(team[0]) for team in ratings]
        for idx, policy in enumerate(policy_ids):
            self.ratings[policy] = ratings[idx][0]

        self._anchor_baseline()

        return self.ratings
