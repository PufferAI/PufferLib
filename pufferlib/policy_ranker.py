from pdb import set_trace as T
import numpy as np

import sqlite3


def win_prob(elo1, elo2):
    '''Calculate win probability such that a difference of
    50/100/150 elo corresponds to win probabilitit 68/95/99.7%'''
    return 1 / (1 + 10 ** ((elo2 - elo1) / 77.6))

def update_elos(elos: np.ndarray, scores: np.ndarray, k: float = 4.0):
    '''Update elos based on the result of a game

    The parameter k controls the magnitude of the update.
    A higher k means that the elo will change more after a game.
    This means that elos will converge faster but less precisely.
    In particular, low k cannot distinguish between players of
    similar skill, while a high k will just take longer to converge.

    The default is tuned for normally distributed player skill
    You should lower it if you have very similar players.
    Raise it if you are evaluating a diverse skill pool.
    '''
    num_players = len(elos)
    assert num_players == len(scores)

    elo_update = [[] for _ in range(num_players)]
    for i in range(num_players):
        for j in range(i+1, num_players):
            delta = scores[i] - scores[j]

            # Convert to elo scoring format
            if delta > 0:
                score_i = 1
            elif delta == 0:
                score_i = 0.5
            else:
                score_i = 0

            # Calculate elo update for pairs
            expected_i = win_prob(elos[i], elos[j])
            expected_j = 1 - expected_i
            score_j = 1 - score_i

            elo_update[i].append(k * (score_i - expected_i))
            elo_update[j].append(k * (score_j - expected_j))

    elo_update = [np.mean(e) for e in elo_update]
    return [elo + update for elo, update in zip(elos, elo_update)]

class Ranker:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    policy TEXT PRIMARY KEY,
                    elo REAL
                );
            """)

    def __repr__(self):
        return '\n'.join([
            f'Policy: {name}, Elo: {elo}'
            for name, elo in self.ratings
        ])

    @property
    def ratings(self):
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM ratings;")

        return {row[0]: row[1] for row in cursor.fetchall()}

    def update(self, scores: dict):
        if len(scores) < 2:
            return

        # Load all elos from DB
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM ratings;")

        elos = {row[0]: row[1] for row in cursor.fetchall()}

        flat_scores = []
        flat_elos = []

        for policy in scores.keys():
            flat_scores.append(scores[policy])
            if policy in elos:
                flat_elos.append(elos[policy])
            else:
                flat_elos.append(1000.0)

        flat_elos = update_elos(flat_elos, flat_scores)
        elos = zip(scores.keys(), flat_elos)
        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO ratings (policy, elo)
                VALUES (?, ?);
            """, elos)
