"""Regression checks for monster codegen parsing helpers."""

import importlib.util
from pathlib import Path


def load_generate_monsters_module():
    """Load the codegen module directly from the repo path."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "ocean/osrs/tools/generate_monsters.py"
    spec = importlib.util.spec_from_file_location("generate_monsters", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_max_hit_accepts_style_suffix():
    """Parses max hit strings that include a style label."""
    generate_monsters = load_generate_monsters_module()

    assert generate_monsters.parse_max_hit("46 (Ranged)") == 46


def test_parse_max_hit_accepts_malformed_html_suffix():
    """Parses the broken Zuk max-hit string from the reference dump."""
    generate_monsters = load_generate_monsters_module()

    assert generate_monsters.parse_max_hit("148&thinsp;\x7f\x7f") == 148
