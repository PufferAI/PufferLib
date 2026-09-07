from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ocean" / "fight_caves" / "viewer" / "eval_contract.py"
SPEC = importlib.util.spec_from_file_location("fight_caves_eval_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL_CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL_CONTRACT)


def test_checkpoint_format_accepts_exact_raw_weight_size(tmp_path):
    checkpoint = tmp_path / "raw.bin"
    checkpoint.write_bytes(b"\x00" * 64)
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) == "raw"


def test_checkpoint_format_accepts_pytorch_zip_container(tmp_path):
    checkpoint = tmp_path / "cpu.bin"
    checkpoint.write_bytes(b"PK\x03\x04state-dictionary-placeholder")
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) == "pytorch"


def test_checkpoint_format_rejects_unknown_or_missing_file(tmp_path):
    checkpoint = tmp_path / "wrong.bin"
    checkpoint.write_bytes(b"not a supported checkpoint")
    assert EVAL_CONTRACT.checkpoint_format(checkpoint, 64) is None
    assert EVAL_CONTRACT.checkpoint_format(tmp_path / "missing.bin", 64) is None
