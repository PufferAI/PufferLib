from pdb import set_trace as T

import numpy as np
import sqlite3
from typing import Dict

import pickle

from pufferlib.rating import OpenSkillRating
from pufferlib.policy_store import PolicySelector


class PolicyRanker():
    def update_ranks(self, scores: Dict[str, float], wandb_policies=[], step: int = 0):
        pass

class OpenSkillPolicySelector(PolicySelector):
    pass

class OpenSkillRanker(PolicyRanker):
    def __init__(self, db_path, anchor: str, mu: int = 1000, anchor_mu: int = 1000, sigma: float = 100/3):
        super().__init__()
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self._tournament = OpenSkillRating(mu, anchor_mu, sigma)
        self._anchor = anchor
        self._default_mu = mu
        self._default_sigma = sigma
        self._anchor_mu = anchor_mu
        self.add_policy(anchor, anchor=True)

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    policy TEXT PRIMARY KEY,
                    mu REAL,
                    sigma REAL
                );
            """)

    def update_ranks(self, scores: Dict[str, float], wandb_policies=[], step: int = 0):
        if len(scores) == 0:
            return

        # Ensuring policies exist
        for policy in scores.keys():
            if policy not in self._tournament.ratings:
                self.add_policy(policy, anchor=(policy == self._anchor))

        # Updating ranks
        if len(scores) > 1:
            T()
            self._tournament.update(
                policy_ids=list(scores.keys()),
                scores=np.array([v for v in scores.values()]),
            )

        # Log updated data (replacing the DataFrame logging)
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM ratings;")
            for row in cursor.fetchall():
                print(row)

        # Logging to wandb
        if len(wandb_policies) > 0:
            import wandb  # Assuming wandb is available
            for wandb_policy in wandb_policies:
                rating = self._tournament.ratings[wandb_policy]
                wandb.log({
                    f"skillrank/{wandb_policy}/mu": rating['mu'],
                    f"skillrank/{wandb_policy}/sigma": rating['sigma'],
                    f"skillrank/{wandb_policy}/score": scores[wandb_policy],
                    "agent_steps": step,
                    "global_step": step,
                })

    def add_policy(self, name: str, mu=None, sigma=None, anchor=False):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO ratings (policy, mu, sigma)
                VALUES (?, ?, ?);
            """, (name, mu if mu is not None else self._default_mu, sigma if sigma is not None else self._default_sigma))

        if name in self._tournament.ratings:
            raise ValueError(f"Policy with name {name} already exists")

        if anchor:
            self._tournament.set_anchor(name)
            self._tournament.ratings[name].mu = self._anchor_mu
        else:
            self._tournament.add_policy(name)
            self._tournament.ratings[name].mu = mu if mu is not None else self._default_mu
            self._tournament.ratings[name].sigma = sigma if sigma is not None else self._default_sigma

    def add_policy_copy(self, name: str, src_name: str):
        mu = self._default_mu
        sigma = self._default_sigma
        if src_name in self._tournament.ratings:
            mu = self._tournament.ratings[src_name].mu
            sigma = self._tournament.ratings[src_name].sigma
        self.add_policy(name, mu, sigma)

    def ratings(self):
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM ratings;")
            return {row[0]: {"mu": row[1], "sigma": row[2]} for row in cursor.fetchall()}

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