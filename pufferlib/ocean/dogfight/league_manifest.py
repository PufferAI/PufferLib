"""League Manifest: Data model and persistence for the league system.

Stores policy entries (active + frozen anchors), round records with win matrices
and ratings, and rating history. Uses atomic JSON save (write .tmp then rename).
"""
import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PolicyEntry:
    id: str
    model_path: str
    obs_scheme: int
    hidden_size: int
    config: dict
    status: str  # "active" | "frozen"
    generation: int
    rating: float
    rating_rd: float
    created_round: int
    best_checkpoint_path: str
    best_checkpoint_generation: int
    consecutive_rejections: int = 0
    flagged_for_review: bool = False
    wandb_run_id: Optional[str] = None
    wandb_run_name: Optional[str] = None
    source_policy: Optional[str] = None
    frozen_at_round: Optional[int] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RoundRecord:
    round: int
    type: str  # "initial_eval" | "train_verify_eval"
    timestamp: str
    win_matrix: dict  # {labels: [...], data: [[...]]}
    ratings: dict  # policy_id -> rating
    verification: Optional[dict] = None
    training_steps: Optional[int] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LeagueManifest:
    def __init__(self):
        self.version = 2
        self.source_project = ""
        self.created = ""
        self.policies: list[PolicyEntry] = []
        self.rounds: list[RoundRecord] = []
        self.rating_history: dict = {}  # policy_id -> [{round, rating, generation, decision}]

    @classmethod
    def load(cls, path: str) -> 'LeagueManifest':
        with open(path, 'r') as f:
            data = json.load(f)
        manifest = cls()
        manifest.version = data.get('version', 2)
        manifest.source_project = data.get('source_project', '')
        manifest.created = data.get('created', '')
        manifest.policies = [PolicyEntry.from_dict(p) for p in data.get('policies', [])]
        manifest.rounds = [RoundRecord.from_dict(r) for r in data.get('rounds', [])]
        manifest.rating_history = data.get('rating_history', {})
        return manifest

    def save(self, path: str):
        """Atomic save: write to temp file then rename."""
        data = {
            'version': self.version,
            'source_project': self.source_project,
            'created': self.created,
            'policies': [p.to_dict() for p in self.policies],
            'rounds': [r.to_dict() for r in self.rounds],
            'rating_history': self.rating_history,
        }
        dir_name = os.path.dirname(path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except:
            os.unlink(tmp_path)
            raise

    def get_active_policies(self) -> list[PolicyEntry]:
        return [p for p in self.policies if p.status == 'active']

    def get_frozen_anchors(self) -> list[PolicyEntry]:
        return [p for p in self.policies if p.status == 'frozen']

    def get_policies_by_scheme(self, scheme: int) -> list[PolicyEntry]:
        return [p for p in self.policies if p.obs_scheme == scheme]

    def get_policy_by_id(self, policy_id: str) -> Optional[PolicyEntry]:
        for p in self.policies:
            if p.id == policy_id:
                return p
        return None

    def add_policy(self, entry: PolicyEntry):
        self.policies.append(entry)
        if entry.id not in self.rating_history:
            self.rating_history[entry.id] = []

    def add_round(self, record: RoundRecord):
        self.rounds.append(record)

    def update_rating_history(self, policy_id: str, round_num: int, rating: float,
                              generation: int, decision: str = None):
        if policy_id not in self.rating_history:
            self.rating_history[policy_id] = []
        entry = {'round': round_num, 'rating': rating, 'generation': generation}
        if decision:
            entry['decision'] = decision
        self.rating_history[policy_id].append(entry)

    def next_round_number(self) -> int:
        if not self.rounds:
            return 0
        return max(r.round for r in self.rounds) + 1
